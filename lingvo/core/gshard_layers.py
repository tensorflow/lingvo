# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Layers and utilities that facilitate building MOE models."""

from lingvo import compat as tf
from lingvo.core import activations
from lingvo.core import base_layer
from lingvo.core import gshard_utils
from lingvo.core import py_utils
from lingvo.core import recurrent
from lingvo.core import tpu_summary
# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
# pylint: enable=g-direct-tensorflow-import


Split = gshard_utils.Split
MeshSplit = gshard_utils.MeshSplit
ZigzagOrderOnDeviceMesh = gshard_utils.ZigzagOrderOnDeviceMesh
GetNonPod2dMesh = gshard_utils.GetNonPod2dMesh


class VarLayer(base_layer.BaseLayer):
  """Container for variables."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('weights', None, '[(name, WeightParams)..] list.')
    p.Define(
        'shared_var_collection_suffix', None,
        'Weights created with collection name ending with '
        'p.shared_var_collection_suffix are shared.')
    p.name = p.name or 'w'
    return p

  def _get_var_from_collection(self, vp):
    for collection in vp.collections:
      if self.params.shared_var_collection_suffix in collection:
        in_collection = tf.get_collection(collection)
        if in_collection:
          return in_collection[0]
    return None

  def __init__(self, params):
    super().__init__(params)
    for k, v in self.params.weights:
      vp = v.Copy()
      if vp.init is None:
        vp.init = self.params.params_init
      # Skip creation if it's already in some collection
      if (not self.params.shared_var_collection_suffix or
          self._get_var_from_collection(vp) is None):
        self.CreateVariable(k, vp)
    if self.params.shared_var_collection_suffix:
      self.InstantiateVariables()

  def FProp(self, theta, *args, **kwargs):

    def MaybeCastToFPropDtype(x):
      if x is None or not x.dtype.is_floating or x.dtype == self._params.fprop_dtype:
        return x
      if self._params.fprop_dtype is None:
        return x
      return tf.cast(x, self._params.fprop_dtype)

    # TODO(lepikhin): MoEBuilder.Embedding can not use '->emb' rule without
    # returning single element  of list of one element below.
    retval = []
    for k, vp in self.params.weights:
      # Try to get the variable value from tf.collection.
      var_value = None
      if self.params.shared_var_collection_suffix:
        var_value = self._get_var_from_collection(vp)
      if var_value is None:
        var_value = theta[k]
      if isinstance(var_value, tf.Variable):
        var_value = var_value.read_value()
      retval.append(MaybeCastToFPropDtype(var_value))
    return retval[0] if len(retval) == 1 else retval


def ShardedWeightParams(shape,
                        init=None,
                        dtype=None,
                        collections=None,
                        tensor_split_dims_mapping=None):
  """Returns a hyperparams for a weight variable with optional XLA sharding."""
  p = py_utils.WeightParams(
      shape,
      init,
      dtype,
      collections,
      tensor_split_dims_mapping=tensor_split_dims_mapping)
  return p


class ShardedVarLayer(VarLayer):
  """Container for variables whose values sharded across different devices."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Delete('weights')
    p.Define('weights', None, '[(name, ShardedWeightParams)..] list.')
    p.Define('cast_to_fprop_dtype', True,
             'Whether to cast variables to fprop_dtype')
    return p

  def InstantiateVariables(self):
    super().InstantiateVariables()
    p = self.params

    def MaybeWeightSplit(k, v):
      # In-place annotate the variable (no sharding op). This makes sure that
      # in some backend implementation, even if the following sharding is
      # optimized away, the backend can still infer the variable sharding.
      split_dims = v.tensor_split_dims_mapping
      if split_dims is not None:
        # Fix the rank difference between variable shape and annotation
        # due to variable shape prefix introduced in builder_layers.RepeatLayer.
        shape_prefix_len = len(self.vars[k].shape) - len(split_dims)
        split_dims = [-1] * shape_prefix_len + split_dims
      gshard_utils.MeshSplit(
          self.vars[k], p.device_mesh, split_dims, use_sharding_op=False)

    for k, v in p.weights:
      MaybeWeightSplit(k, v)

  def FProp(self, theta, *args, **kwargs):
    p = self.params

    # TODO(huangyp, lepikhin): Maybe cast to fprop dtype as well.
    def MaybeWeightSplitAndCastToFPropDtype(k, v):
      x = theta[k]
      if isinstance(x, tf.Variable):
        x = x.read_value()
      if x is None:
        return None

      # We annotate the read value again because some backend implementation
      # may only look at the neighbors of the variable during compilation.
      x = gshard_utils.MeshSplit(
          x, p.device_mesh, v.tensor_split_dims_mapping, use_sharding_op=True)
      if (p.cast_to_fprop_dtype and x.dtype.is_floating and
          x.dtype != p.fprop_dtype and p.fprop_dtype):
        x = tf.cast(x, p.fprop_dtype)
      return x

    retval = [MaybeWeightSplitAndCastToFPropDtype(k, v) for k, v in p.weights]
    return retval[0] if len(retval) == 1 else retval


def _ToTuple(x):
  return x if isinstance(x, tuple) else (x,)


class LayerwiseShardablePipelinedLayer(base_layer.BaseLayer):
  """A layer that implements pipelining across stages.

  It creates a loop over microbatches around a loop-body layer. The loop body
  has a leading num_stages dimension in the input/output data (required) and
  weights (to achieve real pipeline parallelism). This leading dimension can
  be added in 2 ways:

  1) Defined manually in the wrapped layer Params().stage_parallel_body.

  2) Automatically vectorized via tf.vectorized_map(). In this case, use
  Params().single_stage_body instead to define a single stage. This may fail if
  some ops or control flow patterns are not supported by tf.vectorized_map(),
  and use the first option in that case.

  It can run on a single core, or sharded using GShard annotations. If the stage
  dimension is sharded, GShard will produce a cross-core pipelining pattern.

  Inputs to LayerwiseShardablePipelinedLayer should have a leading
  num_microbatch dimension. Each microbatch will be send to each pipeline loop
  iteration.

  The high-level idea is to use a shifting buffer to communicate between stages,
  as shown below (although the real implementation uses recurrent.Recurrent() to
  manage accumulation buffers)::

      input = ...  # shape: [num_microbatches, ...]
      # Insert a num_stages dimension after num_microbatches, then pad to shape:
      #   [num_microbatches + num_stages - 1, num_stages, ...]
      padded_input = pad(expand_dim(input, 1), ...)

      # Shifting buffer
      state = tf.zeros([num_stages, ...])

      # Recurrent loop
      for i in range(num_microbatches + num_stages - 1):
        # shift state to the right by one stage
        shifted_state = tf.pad(state, [[1, 0], ...])[1:]
        in_mask = tf.equal(tf.range(num_stages), 0)
        stages_in = tf.where(in_mask, padded_input[i],  shifted_state)
        state = stage_parallel_body.FProp(theta.body, stages_in)
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_stages', 1, 'Number of pipeline stages.')
    p.Define(
        'stage_parallel_body', None,
        'The param for the main network layer. Its input data should have '
        'a leading dimension that corresponds to num_stages, and its '
        'computation should be parallel along this dimension to achieve '
        'real pipeline parallelism.')
    p.Define(
        'single_stage_body', None,
        'The param for a single stage, which will be automatically vectorized '
        'into a stage-parallel computation.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    if p.stage_parallel_body is not None:
      assert p.single_stage_body is None
      self.CreateChild('body', p.stage_parallel_body)
    else:
      assert p.single_stage_body is not None
      self.CreateChild('body', p.single_stage_body)

  def _CreateChildrenVariables(self):
    p = self.params
    if p.stage_parallel_body is None:
      with py_utils.VariableShapePrefixContext(p.num_stages):
        self.children.body.InstantiateVariables()
    else:
      super()._CreateChildrenVariables()

  def BodyFProp(self, theta, *args):
    p = self.params
    if p.stage_parallel_body is not None:
      return self.body.FProp(theta, *args)

    def _SingleStageFProp(x):
      return self.body.FProp(x.theta, *x.args)

    return tf.vectorized_map(
        _SingleStageFProp,
        py_utils.NestedMap(theta=theta, args=args),
        fallback_to_while_loop=False)

  def FProp(self, theta, *args):
    p = self.params

    # Adds a `stages` dimension after the leading num_microbatches to the inputs
    # which will be sharded. Also pad the leading num_microbatches dimension by
    # num_stages - 1 to match loop iteration count, which corresponds to the
    # bubbles between forward and backward passes.
    #
    # Inputs are not the loop state: they are not changed during the loop. The
    # state (shifting buffer) does not have a num_microbatches dimension.
    def _PadInput(inp):
      if inp is None:
        return None
      # Takes input tensor of shape [num_microbatches, ...] and returns padded
      # tensor of shape [num_microbatches + (num_stages - 1), num_stages,  ...],
      # where num_stages is a new dimension.
      with_new_dim = tf.expand_dims(inp, 1)
      padding = [[0, p.num_stages - 1], [0, p.num_stages - 1]]
      padding += [[0, 0]] * (len(inp.shape) - 1)
      padded = tf.pad(with_new_dim, padding)
      assert len(padded.shape) == len(inp.shape) + 1
      assert padded.shape[1] == p.num_stages
      return padded

    padded_inputs = tf.nest.map_structure(_PadInput, args)
    padded_shapes = tf.nest.map_structure(
        lambda x: None if x is None else x.shape, padded_inputs)
    remove_first_dim = lambda x: None if x is None else x[1:]
    state_shapes = tf.nest.map_structure(remove_first_dim, padded_shapes)

    def _ArgsToState(arg_list):
      """Returns a NestedMap from a list of FProp args."""
      state = py_utils.NestedMap()
      # Maintains a mapping from arg_idx to tensor. states cannot contain None
      # tensors.
      for idx in range(len(padded_inputs)):
        if isinstance(arg_list[idx], py_utils.NestedMap):
          # Make sure each value in the NestedMap is a tensor.
          if not all(isinstance(t, tf.Tensor) for t in arg_list[idx].Flatten()):
            raise ValueError(
                'Each value in the input NestedMap must be a tensor.')
        if arg_list[idx] is not None:
          state['_s{}'.format(idx)] = arg_list[idx]
      return state

    def _StateToArgs(state, shapes):
      """Returns a list of FProp args from a NestedMap."""
      arg_list = []
      for idx in range(len(padded_inputs)):
        attr = '_s{}'.format(idx)
        arg_list.append(state[attr] if attr in state else None)
        tf.nest.map_structure(lambda x, s: x.set_shape(s), arg_list[-1],
                              shapes[idx])
      return arg_list

    def _CellFn(theta, state0, inputs):
      """Recurrent cell function wrapper of body.FProp."""

      def _SelectInput(state, inp):
        # The state is aligned to previous stage. We shift it to the right by 1
        # stage. If the stage dimension is partitioned in GShard, this will
        # cause a collective-permute being added.
        padding = [[1, 0]] + [[0, 0]] * (len(state.shape) - 1)
        shifted_state = tf.pad(state, padding)[0:p.num_stages, ...]
        in_mask = tf.reshape(
            tf.equal(tf.range(p.num_stages), 0),
            [p.num_stages] + [1] * (len(inp.shape) - 1))
        return tf.where(
            tf.broadcast_to(in_mask, shifted_state.shape),
            tf.cast(inp, shifted_state.dtype), shifted_state)

      selected_inputs = tf.nest.map_structure(
          _SelectInput, _StateToArgs(state0, state_shapes),
          _StateToArgs(inputs, state_shapes))
      # Runs the actual body's FProp
      fprop_outputs = self.BodyFProp(theta, *selected_inputs)
      fprop_outputs = _ToTuple(fprop_outputs)
      assert len(fprop_outputs) == len(selected_inputs)

      # Passes fprop outputs to the next layer through state.
      state1 = _ArgsToState(fprop_outputs)
      return state1, py_utils.NestedMap()

    with tf.name_scope(p.name):
      inputs_nmap = _ArgsToState(padded_inputs)

      def _CreateInitState(inp):
        return tf.zeros(inp.shape[1:], dtype=inp.dtype)

      # Add FProp arg list to state0.
      state0 = tf.nest.map_structure(_CreateInitState, inputs_nmap)
      # Runs body.FProp k times using Recurrent where k = dim 0 of inputs_nmap.
      accum, _ = recurrent.Recurrent(
          theta=theta.body,
          state0=state0,
          inputs=inputs_nmap,
          cell_fn=_CellFn,
          allow_implicit_capture=p.allow_implicit_capture)

      # Retrieves fprop outputs.
      def _ExtractLastStage(outp):
        return outp[p.num_stages - 1:, -1, ...]

      output_tensors = tf.nest.map_structure(_ExtractLastStage,
                                             _StateToArgs(accum, padded_shapes))
      return output_tensors[0] if len(args) == 1 else tuple(output_tensors)


class StateLayer(base_layer.BaseLayer):
  """Container for recurrent state for incremental decoding.

  It has two operation modes.

  During training, it does nothing.
  It expects that FProp(x, t) is called with t=None, and returns x unchanged.

  During decoding, it expects:

    t: an int32 scalar and
    x: a tensor of shape `[batch, 1, ...]`.

  It updates state `x_full[:, t, :] <- x[:, 0, :]` and returns x_full.
  The shape of x_full is then `[batch, time, ...]`.

  The state is stored as theta.state attribute.

  To construct initial state, call InitState classmethod on the root layer.
  InitState() will traverse root layer children recursively, will initialize
  internal state for each StateLayer instance, and will return a nested
  tuple of states.

  For incremental iteration the static methods work as follows::

    dec = builder.DecoderLayerStack(...).Instantiate()
    state0 = StateLayer.InitState(dec, shape=[tgt_batch, max_len])
    theta0 = StateLayer.UpdateTheta(dec, dec.theta, state0, t=0)
    # (FProp in nested StateLayer now has access to 'state0' and 't')
    dec.FProp(theta0, ...)
    # FProp will  modify theta0 in-place
    state1 = state0.copy()
    state1 = StateLayer.UpdateState(dec, theta0, state1)
  """
  _use_flat_beam_search = False

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('shape', [None, None], 'batch, time, etc...')
    p.Define('use_xla_dynamic_update_slice', True, 'internal optimization')
    return p

  def NewState(self, shape):
    """Returns initial state.

    Args:
      shape: [batch, time] for beam_search_tpu_helper or [batch, beam, time] for
        flat_beam_search.

    Returns:
      zero-initialized state tensor with shape [batch, time, ...] for
        beam_search_tpu_helper or [time, batch, beam, ...] for flat_beam_search.

    Raises:
      ValueError: the length of shape is not 2 or 3.
    """
    p = self.params
    fprop_dtype = p.dtype or py_utils.FPropDtype(p)

    if len(shape) == 2:
      # For use with beam_search_tpu_helper batch_major_compute=1
      shape = tuple(shape) + tuple(p.shape[2:])
      state = tf.zeros(shape, fprop_dtype)
      return state
    elif len(shape) == 3:
      # For use with flat_beam_search
      self._use_flat_beam_search = True
      batch, beam, max_steps = shape
      if p.use_xla_dynamic_update_slice:
        state_shape = [batch, max_steps * beam]
        # Need to remember beam_size to correctly map 't' argument of Fprop
        # to the combined (max_steps * beam) dimension.
        self._beam = beam
      else:
        state_shape = [max_steps, batch, beam]
      state = tf.Empty(state_shape + p.shape[2:], fprop_dtype, init=True)
      return state
    else:
      raise ValueError('bad shape: %r' % shape)

  def FProp(self, theta, x):
    p = self.params
    if not hasattr(theta, 't'):
      return x
    t = theta.t
    if t is None:
      return x
    assert hasattr(theta, 'state')
    state = theta.state

    tf.logging.info('p.name=%r', p.name)
    tf.logging.info('state=%r', state)
    tf.logging.info('x=%r', x)
    tf.logging.info('t=%r', t)

    with tf.name_scope(p.name):
      if not self._use_flat_beam_search:
        z = tf.one_hot(t, tf.shape(state)[1])
        z = tf.expand_dims(z, 0)
        while len(z.shape) < len(x.shape):
          z = tf.expand_dims(z, -1)
        y = state = (1 - z) * state + z * x

      if self._use_flat_beam_search and not p.use_xla_dynamic_update_slice:
        state_slice_size = int(state.shape[2])
        update_slice_size = int(x.shape[1])
        if update_slice_size == state_slice_size:
          state = tf.InplaceUpdate(state, t, tf.cast(x, state.dtype))
        else:
          # With prefix decoding the first call to decoder can have
          # sequence length (N * beam_size) with N > 1.
          # In this special case state tensor update is implemented as multiple
          # InplaceUpdate ops each for a slice [batch_size, beam_size].
          div = int(update_slice_size / state_slice_size)
          assert update_slice_size == state_slice_size * div, (
              update_slice_size, state_slice_size)
          for i, x_i in enumerate(tf.split(x, div, 1)):
            state = tf.InplaceUpdate(state, t + i, tf.cast(x_i, state.dtype))
        tf.logging.info('state*=%r', state)
        # [T,B,L,...]
        y = tf.cast(state, x.dtype)
        # [T, B, L, ...] -> [B, T, L, ...]
        perm = list(range(len(y.shape)))
        perm[:2] = [1, 0]
        y = tf.transpose(y, perm)
        # [B, T, L, ...] -> [B, T*L, ...]
        y_shape = list(y.shape)
        y_shape[1:3] = [int(y_shape[1]) * int(y_shape[2])]
        y = tf.reshape(y, y_shape)

      if self._use_flat_beam_search and p.use_xla_dynamic_update_slice:
        update_index = [0] * len(state.shape)
        update_index[1] = (t * self._beam)
        state = xla.dynamic_update_slice(state, tf.cast(x, state.dtype),
                                         update_index)
        y = state

    theta.state = state

    tf.logging.info('y=%r', y)
    return y

  @classmethod
  def InitState(cls, layer, shape):
    """Returns new state with leading shape=[batch, time]."""

    def Rec(layer):  # pylint: disable=missing-docstring
      state = None
      if isinstance(layer, StateLayer):
        assert not layer.children
        state = layer.NewState(shape)
        return state
      for c_name, c in layer.children.items():
        c_state = Rec(c)
        if c_state is not None:
          if state is None:
            state = py_utils.NestedMap()
          state[c_name] = c_state
      return state

    state = Rec(layer)
    assert state is not None
    return state

  @classmethod
  def UpdateTheta(cls, layer, theta, state, t=None):
    """Returns theta with state."""

    def Rec(layer, theta, state):  # pylint: disable=missing-docstring
      if isinstance(layer, StateLayer):
        theta.state = state
        theta.t = t
        return
      for c_name, c in layer.children.items():
        if c_name in state:
          Rec(c, theta[c_name], state[c_name])

    Rec(layer, theta, state)
    return theta

  @classmethod
  def UpdateState(cls, layer, theta, state):
    """Returns updated state from theta."""

    def Rec(layer, theta, state):  # pylint: disable=missing-docstring
      for c_name, c in layer.children.items():
        if isinstance(c, StateLayer):
          state[c_name] = theta[c_name].state
        elif c_name in state:
          Rec(c, theta[c_name], state[c_name])

    Rec(layer, theta, state)
    return state


class OverrideLayer(base_layer.BaseLayer):
  """Allows to override arbitrary tensors in the graph.

  If key is not set in the global context, FProp does nothing.
  Otherwise it returns value associated to 'key'.

  To override a tensor during my_layer.FProp::

    OverrideLayer.Set(key, value)
    out_with_override = my_layer.FProp(...)
    OverrideLayer.Clear()
  """

  _OVERRIDE = {}

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('key', None, 'Context key')
    return p

  def FProp(self, theta, x):
    p = self.params
    if p.key and p.key in self._OVERRIDE:
      return self._OVERRIDE[p.key]
    else:
      return x

  @classmethod
  def Set(cls, k, v):
    cls._OVERRIDE[k] = v

  @classmethod
  def Clear(cls):
    cls._OVERRIDE.clear()


class ReshapeInputLayer(base_layer.BaseLayer):
  """Reshape input only for training."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_groups', None, 'Number of groups.')
    p.Define('num_devices', 1, 'Number of devices.')
    p.Define(
        'model_dims', None,
        'A list, the dimensions that M is reshaped into. If None, default'
        ' to the last dimension of fprop inputs')
    return p

  def FProp(self, unused_theta, inputs, segment_id):
    p = self.params
    paddings = tf.cast(tf.equal(segment_id, 0), inputs.dtype)
    orig_inputs = inputs
    # input size in tokens
    input_size = int(orig_inputs.shape[0] * orig_inputs.shape[1])
    group_size = input_size // p.num_groups
    assert (group_size * p.num_groups == input_size), (p.num_groups, input_size)
    model_dims = p.model_dims or [orig_inputs.shape[-1]]
    inputs = tf.reshape(
        orig_inputs, [p.num_groups, group_size] + model_dims,
        name='grouped_inputs')
    if p.num_devices > 1:
      inputs = gshard_utils.Split(inputs, 0, p.num_devices)
    paddings = tf.reshape(paddings, py_utils.GetShape(inputs)[:2])
    return inputs, paddings


class SharedEmbeddingSoftmaxLayer(base_layer.BaseLayer):
  """Shared weights for embemdding lookup and softmax."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('vocab_size', 0, 'Num tokens in vocab.')
    p.Define('max_len', 0, 'Num of token in the sequence.')
    p.Define('embedding_dim', 0, 'Depth of the output.')
    p.Define('z_loss_coef', 1e-4, 'Label smoothing.')
    p.Define('num_devices', 1, 'Number of devices for sharding.')
    p.Define('logits_abs_max', None, 'Logits clipping.')
    p.Define('label_smoothing', 0.1, 'Label smoothing.')
    p.Define(
        'use_tgt_labels_size_as_loss_denominator', True,
        'False to use total number of non-padding tokens instead of '
        'fixed tgt_labels tensor size.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    emb_p = py_utils.WeightParams(
        init=py_utils.WeightInit.Gaussian(),
        shape=[p.vocab_size, p.embedding_dim])
    pos_emb_p = py_utils.WeightParams(
        init=py_utils.WeightInit.Gaussian(), shape=[p.max_len, p.embedding_dim])
    self.CreateVariable('embedding', emb_p)
    self.CreateVariable('pos_emb', pos_emb_p)

  def _MaybeSplit(self, x):
    if True or self.params.num_devices <= 1:
      return x
    return gshard_utils.Split(x, 0, self.params.num_devices)

  def FProp(self, theta, ids, segment_pos):
    p = self.params
    fprop_dtype = py_utils.FPropDtype(p)

    ids = self._MaybeSplit(ids)
    segment_pos = self._MaybeSplit(segment_pos)

    one_hot_ids = tf.one_hot(ids, p.vocab_size, dtype=fprop_dtype)
    one_hot_ids = self._MaybeSplit(one_hot_ids)

    one_hot_pos = tf.one_hot(segment_pos, p.max_len, dtype=fprop_dtype)
    one_hot_pos = self._MaybeSplit(one_hot_pos)

    token_emb = tf.einsum('VH,BLV->BLH', theta.embedding, one_hot_ids)
    token_emb = self._MaybeSplit(token_emb)

    pos_emb = tf.einsum('VH,BLV->BLH', theta.pos_emb, one_hot_pos)
    pos_emb = self._MaybeSplit(pos_emb)
    return self._MaybeSplit(token_emb + pos_emb)

  def ComputeLoss(self, theta, activation, labels, segment_ids):
    p = self.params
    activation = self._MaybeSplit(
        self._MaybeSplit(activation) * (p.embedding_dim**-0.5))
    softmax_weights = theta.embedding
    if activation.dtype != softmax_weights.dtype:
      softmax_weights = tf.cast(softmax_weights, activation.dtype)
    logits = self._MaybeSplit(
        tf.einsum('BLM,VM->BLV', activation, softmax_weights))
    if p.logits_abs_max is not None:
      logits = self._MaybeSplit(
          py_utils.clip_by_value(logits, -p.logits_abs_max, p.logits_abs_max))

    off_value = p.label_smoothing / p.vocab_size
    on_value = 1.0 - p.label_smoothing + off_value
    soft_targets = self._MaybeSplit(
        tf.one_hot(
            labels, p.vocab_size, on_value=on_value, off_value=off_value))
    xent = self._MaybeSplit(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels, p.vocab_size), logits=logits))
    loss = self._MaybeSplit(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=soft_targets, logits=logits))
    soft_targets_xent = loss

    if p.z_loss_coef > 0.0:
      log_z = tf.math.reduce_logsumexp(logits, -1)
      z_loss_inc = p.z_loss_coef * tf.math.square(log_z)
      loss += z_loss_inc

    non_padding = self._MaybeSplit(
        tf.cast(tf.not_equal(segment_ids, 0), py_utils.FPropDtype(p)))

    per_token_loss = self._MaybeSplit(loss * non_padding)
    if p.z_loss_coef > 0.0:
      per_token_z_loss_inc = self._MaybeSplit(z_loss_inc * non_padding)

    if p.use_tgt_labels_size_as_loss_denominator:
      # E.g. loss is going to be tiny if inputs are not packed and only a
      # fraction of tgt_labels are non-padding.
      loss_denom = tf.reduce_sum(tf.ones_like(non_padding))
      per_example_loss_denom = tf.reduce_sum(tf.ones_like(non_padding), 1)
    else:
      loss_denom = tf.reduce_sum(non_padding)
      per_example_loss_denom = tf.reduce_sum(non_padding, 1)
    avg_loss = tf.reduce_sum(per_token_loss) / loss_denom
    avg_z_loss_inc = (tf.reduce_sum(per_token_z_loss_inc) /
                      loss_denom) if p.z_loss_coef > 0.0 else 0.0

    soft_targets_xent = (
        tf.reduce_sum(self._MaybeSplit(soft_targets_xent * non_padding)) /
        tf.reduce_sum(non_padding))

    # TODO(lepikhin): consider returning
    #   {'loss': (unnormalized per_token_loss, tf.reduce_sum(non_padding))}
    per_example_loss = {
        'loss': tf.reduce_sum(per_token_loss, 1) / per_example_loss_denom
    }
    return {
        'mean_xent': (tf.reduce_sum(self._MaybeSplit(xent * non_padding)) /
                      tf.reduce_sum(non_padding), tf.reduce_sum(non_padding)),
        'soft_targets_xent': (soft_targets_xent, tf.reduce_sum(non_padding)),
        'weight': (tf.reduce_sum(non_padding), 1.0),
        'loss': (avg_loss, 1.0),
        'avg_z_loss_inc': (avg_z_loss_inc, 1.0),
    }, per_example_loss


def Top2GatingOnLogits(inputs,
                       paddings,
                       logits,
                       num_devices,
                       experts_dim,
                       expert_capacity_dim,
                       fprop_dtype,
                       use_xla_sharding=True,
                       second_expert_policy='all',
                       second_expert_threshold=0.0,
                       legacy_mtf_behavior=True,
                       capacity_factor=None,
                       importance=None,
                       mask_dtype=None):
  """Computes Top-2 gating for Mixture-of-Experts.

  There are two expected usages of this function:

  1. used with xla_sharding. In this case, 'inputs' corresponds to a sharded
     tensor across multiple tpu cores. The operations within this function are
     automatically sharded/replicated across tpu cores.
  2. used within other projects where'inputs' is always local to one tpu
     core. All computations below are carried out on one tpu core only. This
     function tries to dispatch examples across tpu cores in such a way that
     each expert is assigned no more than 'expert_capacity_dim' number of
     examples.

  Below ` indicates common way of splitting along mesh dimension.

  Dimensions cheat sheet::

    G: group_dim
    S: group_size_dim
    E: number of experts
    C: capacity per expert
    M: model_dim (same as input_dim, same as output_dim)
    B: original batch_dim
    L: original sequence_length_dim

  Note that for local_dispatch original batch BLM is reshaped into GSM, each
  group `g = 0...G-1` is being dispatched independently.

  Args:
    inputs: G`SM Tensor.
    paddings: G`S Tensor.
    logits: G`SE Tensor.
    num_devices: number of MoE devices for local dispatch
    experts_dim: number of experts.
    expert_capacity_dim: number of examples per minibatch(group) per expert.
      Each example is typically a vector of size input_dim, representing
      embedded token or an element of Transformer layer output.
    fprop_dtype: activations datatype to use.
    use_xla_sharding: bool, True if this function is used for the xla_sharding
      case.
    second_expert_policy: 'all', 'sampling' or 'random'.

      - 'all': we greedily pick the 2nd expert.
      - 'sampling': we sample the 2nd expert from the softmax.
      - 'random': we optionally 'random'-ize dispatch to second-best expert
        proportional to (weight / second_expert_threshold).

    second_expert_threshold: threshold for probability normalization for
      second_expert_policy == 'random'.
    legacy_mtf_behavior: bool, True if to match legacy mtf behavior exactly.
    capacity_factor: if set, increases expert_capacity_dim to at least
      (group_size * capacity_factor) / experts_dim
      where `group_size` is the size of G dimension of `inputs`. If the
      value of expert_capacity_dim is already big enough no change is made.
    importance: input importance weights for routing (G`S Tensor or None).
    mask_dtype: using bfloat16 for fprop_dtype could be problematic for mask
      tensors, mask_dtype is a special dtype for such tensors.

  TODO(lepikhin): get rid of the legacy_mtf_behavior flag.

  Returns:
    A tuple (aux_loss, combine_tensor, dispatch_tensor).

    - aux_loss: auxiliary loss, for equalizing the expert assignment ratios.
    - combine_tensor: G`SEC Tensor for combining expert outputs.
    - dispatch_tensor: G`SEC Tensor, scattering/dispatching inputs to
      experts.
  """
  if mask_dtype is None:
    mask_dtype = fprop_dtype
  if use_xla_sharding:
    tf.logging.warning('Sharding propagation should be sufficient and Splits '
                       'within Top2GatingOnLogits are generally redundant.')
  del inputs  # inputs is currently not used.
  # logits.dtype could be tf.float32
  raw_gates = tf.nn.softmax(logits)  # along E dim
  if raw_gates.dtype != fprop_dtype:
    raw_gates = tf.cast(raw_gates, fprop_dtype)

  if capacity_factor is not None:
    # Determine expert capacity automatically depedning on the input size.
    group_size_dim = int(logits.shape[1])
    auto_expert_capacity = int((group_size_dim * capacity_factor) / experts_dim)
    if expert_capacity_dim < auto_expert_capacity:
      expert_capacity_dim = auto_expert_capacity
      # Round up to a multiple of 4 to avoid possible padding.
      while expert_capacity_dim % 4:
        expert_capacity_dim += 1
      tf.logging.info(
          'Setting expert_capacity_dim=%r (capacity_factor=%r '
          'group_size_dim=%r experts_dim=%r name_scope=%r)',
          expert_capacity_dim, capacity_factor, group_size_dim, experts_dim,
          tf.get_default_graph().get_name_scope())
    tpu_summary.scalar('expert_capacity', expert_capacity_dim)

  # top first and second gate value and expert index for each input
  #
  # GSK Tensors, K=2
  def _MaybeSplit(x):
    if use_xla_sharding:
      return gshard_utils.Split(x, 0, num_devices)
    else:
      return x

  def _CreateOverCapacityRatioSummary(mask, position_in_expert, capacity, name):
    with tf.name_scope('over_capacity'):
      ge_capacity = tf.greater_equal(mask * position_in_expert, capacity)
      over_capacity = tf.reduce_sum(tf.cast(ge_capacity, tf.float32))
      over_capacity_ratio = over_capacity / tf.maximum(
          tf.constant(1.0, dtype=tf.float32),
          tf.cast(tf.reduce_sum(mask), tf.float32))
      py_utils.AddTpuSummaryTensor(name, over_capacity_ratio)
      tpu_summary.scalar(name, over_capacity_ratio, while_loop_reduce='mean')

  # As pointed out by zhifengc@ this method needs to be refactored. lepikhin@
  # and krikun@ will:
  #   - expand moe_spmd_test to compare Adafactor updates, slots on TPU
  #   including 2x2 with sharding
  #
  #   - add more tests for policy="random"
  #
  #   - add single step test for full size WMT model on CPU
  #
  # and then break this function into modules.
  #
  # GS
  index_1 = tf.math.argmax(raw_gates, axis=-1, output_type=tf.int32)
  index_1 = _MaybeSplit(index_1)
  tpu_summary.tensor('index_1', index_1)

  # GSE
  mask_1 = tf.one_hot(index_1, experts_dim, dtype=mask_dtype)
  mask_1 = _MaybeSplit(mask_1)
  density_1_proxy = raw_gates

  if importance is not None:
    importance_is_one = tf.equal(importance, 1.0)
    mask_1 *= tf.expand_dims(tf.cast(importance_is_one, mask_1.dtype), -1)
    density_1_proxy *= tf.expand_dims(
        tf.cast(importance_is_one, density_1_proxy.dtype), -1)
  else:
    if len(mask_1.shape) == 3:
      importance = tf.ones_like(mask_1[:, :, 0])
    else:
      importance = tf.ones_like(mask_1[:, :, :, 0])
    if paddings is not None:
      nonpaddings = 1.0 - paddings
      mask_1 *= tf.expand_dims(tf.cast(nonpaddings, mask_1.dtype), -1)
      density_1_proxy *= tf.expand_dims(
          tf.cast(nonpaddings, density_1_proxy.dtype), -1)
      importance = nonpaddings

  gate_1 = tf.einsum('...GSE,...GSE->...GS', raw_gates,
                     tf.cast(mask_1, raw_gates.dtype))
  gates_without_top_1 = raw_gates * (1.0 - tf.cast(mask_1, raw_gates.dtype))

  if second_expert_policy == 'sampling':
    # We directly sample the 2nd expert index from the softmax over of the 2nd
    # expert by getting rid of the 1st expert already selected above. To do so,
    # we set a very negative value to the logit corresponding to the 1st expert.
    # Then we sample from the softmax (categorical) distribution using the
    # Gumbel max trick.
    noise = _MaybeSplit(tf.random.uniform(logits.shape, dtype=logits.dtype))
    # Generates standard Gumbel(0, 1) noise, GSE Tensors
    noise = -tf.math.log(-tf.math.log(noise))
    very_negative_logits = _MaybeSplit(
        (tf.ones_like(logits) * logits.dtype.max *
         tf.constant(-0.7, dtype=logits.dtype)))
    # Gets rid of the first expert by setting its logit to be very negative
    updated_logits = _MaybeSplit(
        tf.where(mask_1 > 0.0, very_negative_logits, logits))
    # Adds the Gumbel noise to the updated logits
    noised_logits = _MaybeSplit(updated_logits + noise)
    # Picks the index of the largest noised logit as the 2nd expert. This is
    # equivalent to sampling from the softmax over the 2nd experts.
    index_2 = tf.math.argmax(noised_logits, axis=-1, output_type=tf.int32)
  else:
    index_2 = tf.math.argmax(gates_without_top_1, axis=-1, output_type=tf.int32)

  index_2 = _MaybeSplit(index_2)
  mask_2 = tf.one_hot(index_2, experts_dim, dtype=mask_dtype)
  mask_2 = _MaybeSplit(mask_2)
  if paddings is not None:
    importance_is_nonzero = tf.greater(importance, 0.0)
    mask_2 *= tf.expand_dims(tf.cast(importance_is_nonzero, mask_2.dtype), -1)
  gate_2 = tf.einsum('...GSE,...GSE->...GS', gates_without_top_1,
                     tf.cast(mask_2, gates_without_top_1.dtype))

  if legacy_mtf_behavior:
    # cl/298510175 moved this branch for gate_{1,2} denom calculation here.
    #
    # For policy=random, it's better to nomalize gate_{1,2} before taking
    # capacity  into account and before potentially dropping second expert.
    #
    # According to mean_xent:
    #   MoE_512_102xen_PolicyAll_298510175
    #   MoE_512_102xen_PolicyRandom_298510175
    #
    # vs pre-cl/298510175
    #   MoE_512_102xen_PolicyRandom
    #   MoE_512_102xen_PolicyAll
    #
    # it substantially improves policy=random with threshold=0.5 which
    # historically was better than policy="all"
    #
    # Also confirmed this by decoding
    #   nmt_train/m4/data/es_en/test.txt
    #   nmt_train/m4/data/ru_en/test.txt
    #   nmt_train/m4/data/zh_en/test.txt
    # and improving BLEU
    #
    # moe_decode.MoE_512_102xen_PolicyRandom_298510175-160000.batch1024.beam4.c_dim4.ln0.8.rkv.mteval102
    #   0.421443
    #   0.327102
    #   0.315693
    # vs
    # moe_decode.feb18_non_fig_snapshot_2626_MoE_512_102xen_PolicyRandom-190000.batch1024.beam4.c_dim4.ln0.8.rkv.mteval102
    #   0.399232
    #   0.310606
    #   0.288229
    #
    # Additional comparison, see mean_xent with
    # legacy_mtf_behavior=False models
    #   3 - MoE_512_102xen_PolicyAll_LegacyFalse
    #   6 - MoE_512_102xen_PolicyRandom_LegacyFalse
    # shows that policy="random" gets worse with legacy_mtf_behavior=False, and
    # is similar to pre-cl/298510175
    #   4 - MoE_512_102xen_PolicyRandom
    #
    # gate_1 can become 0 due to Expert being out of capacity.
    #
    # gate_2 can become 0 due to
    #   second_expert_policy == 'random'
    # or "out of capacity" scenario.
    #
    # Here we renormalize regardless of cases above.
    denom = gate_1 + gate_2 + 1e-9
    gate_1 /= denom
    gate_2 /= denom

  # We reshape the mask as [X*S, E], and compute cumulative sums of
  # assignment indicators for each expert index e \in 0..E-1 independently.
  # First occurrence of assignment indicator is excluded, see exclusive=True
  # flag below.
  #
  # tf.cumsum over S dim: mask_1 is ...GSE tensor. Pontentially with outer_dim
  # O.
  position_in_expert_1 = tf.cumsum(mask_1, exclusive=True, axis=-2)

  # GS Tensor
  capacity = tf.cast(expert_capacity_dim, dtype=position_in_expert_1.dtype)

  # GE Tensor (reducing S out of GSE tensor mask_1)
  # density_1[:, e] represents assignment ratio (num assigned / total) to
  # expert e as top_1 expert without taking capacity into account.
  assert importance.dtype == fprop_dtype
  if legacy_mtf_behavior:
    density_denom = 1.0
  else:
    density_denom = tf.reduce_mean(importance, axis=(1))[:, tf.newaxis] + 1e-6
  density_1 = tf.reduce_mean(
      tf.cast(mask_1, fprop_dtype), axis=-2) / density_denom
  # density_1_proxy[:, e] represents mean of raw_gates for expert e, including
  # those of examples not assigned to e with top_k.
  assert density_1_proxy.dtype == fprop_dtype
  density_1_proxy = tf.reduce_mean(density_1_proxy, axis=-2) / density_denom

  with tf.name_scope('aux_loss'):
    # The MoE paper (https://arxiv.org/pdf/1701.06538.pdf) uses an aux loss of
    # reduce_mean(density_1_proxy * density_1_proxy). Here we replace one of
    # the density_1_proxy with the discrete density_1 following mesh_tensorflow.
    aux_loss = tf.reduce_mean(density_1_proxy * density_1)  # element-wise
    aux_loss *= experts_dim * experts_dim  # const coefficient

  # Add the over capacity ratio for expert 1
  _CreateOverCapacityRatioSummary(mask_1, position_in_expert_1, capacity,
                                  'over_capacity_1_ratio')

  mask_1 *= tf.cast(tf.less(position_in_expert_1, capacity), dtype=mask_1.dtype)
  position_in_expert_1 = tf.einsum('...GSE,...GSE->...GS', position_in_expert_1,
                                   mask_1)

  # How many examples in this sequence go to this expert
  mask_1_count = tf.einsum('...GSE->...GE', mask_1)
  # [batch, group] - mostly ones, but zeros where something didn't fit
  mask_1_flat = tf.einsum('...GSE->...GS', mask_1)
  assert mask_1_count.dtype == mask_dtype
  assert mask_1_flat.dtype == mask_dtype

  if second_expert_policy == 'all' or second_expert_policy == 'sampling':
    pass
  elif second_expert_policy == 'random':
    # gate_2 is between 0 and 1, reminder:
    #
    #   raw_gates = tf.nn.softmax(logits)
    #   index_1 = tf.math.argmax(raw_gates, axis=-1, output_type=tf.int32)
    #   mask_1 = tf.one_hot(index_1, experts_dim, dtype=fprop_dtype)
    #   gate_1 = tf.einsum('GSE,GSE->GS', raw_gates, mask_1)
    #
    # E.g. if gate_2 exceeds second_expert_threshold, then we definitely
    # dispatch to second-best expert. Otherwise we dispatch with probability
    # proportional to (gate_2 / threshold).
    #
    sampled_2 = tf.less(
        _MaybeSplit(tf.random.uniform(gate_2.shape, dtype=gate_2.dtype)),
        (gate_2 / max(second_expert_threshold, 1e-9)))
    gate_2 *= tf.cast(sampled_2, gate_2.dtype)
    mask_2 *= tf.cast(tf.expand_dims(sampled_2, -1), mask_2.dtype)
  else:
    raise ValueError(second_expert_policy)

  position_in_expert_2 = tf.cumsum(
      mask_2, exclusive=True, axis=-2) + tf.expand_dims(mask_1_count, -2)

  # Add the over capacity ratio for expert 2
  _CreateOverCapacityRatioSummary(mask_2, position_in_expert_2, capacity,
                                  'over_capacity_2_ratio')

  mask_2 *= tf.cast(tf.less(position_in_expert_2, capacity), mask_2.dtype)
  position_in_expert_2 = tf.einsum('...GSE,...GSE->...GS', position_in_expert_2,
                                   mask_2)
  mask_2_flat = tf.reduce_sum(mask_2, axis=-1)

  # Equivalent non-einsum implementation:
  #
  # position_in_expert_2 *= mask_2
  # position_in_expert_2 = tf.reduce_sum(
  #     position_in_expert_2, axis=-1, name='position_in_expert_2')

  gate_1 *= tf.cast(mask_1_flat, gate_1.dtype)
  gate_2 *= tf.cast(mask_2_flat, gate_2.dtype)

  if not legacy_mtf_behavior:
    denom = gate_1 + gate_2
    # To avoid divide by 0.
    denom = tf.where(denom > 0, denom, tf.ones_like(denom))
    gate_1 /= denom
    gate_2 /= denom

  # GSC Tensor
  assert position_in_expert_1.dtype == mask_dtype  # could be float32 in tests
  b = tf.one_hot(
      tf.cast(position_in_expert_1, dtype=tf.int32),
      expert_capacity_dim,
      dtype=fprop_dtype,
      name='one_hot_b_0')
  # GSE Tensor
  a = tf.expand_dims(gate_1 * tf.cast(mask_1_flat, fprop_dtype),
                     -1) * tf.one_hot(
                         index_1, experts_dim, dtype=fprop_dtype)
  # GSEC Tensor
  first_part_of_combine_tensor = tf.einsum(
      '...GSE,...GSC->...GSEC', a, b, name='first_part_of_combine_tensor')

  # GSC Tensor
  assert position_in_expert_2.dtype == mask_dtype  # could be float32 in tests
  b = tf.one_hot(
      tf.cast(position_in_expert_2, dtype=tf.int32),
      expert_capacity_dim,
      dtype=fprop_dtype,
      name='one_hot_b_1')
  # GSE Tensor
  a = tf.expand_dims(gate_2 * tf.cast(mask_2_flat, fprop_dtype),
                     -1) * tf.one_hot(
                         index_2, experts_dim, dtype=fprop_dtype)
  second_part_of_combine_tensor = tf.einsum(
      '...GSE,...GSC->...GSEC', a, b, name='second_part_of_combine_tensor')

  # GSEC Tensor
  combine_tensor = tf.math.add(
      first_part_of_combine_tensor,
      second_part_of_combine_tensor,
      name='combine_tensor')
  combine_tensor = _MaybeSplit(combine_tensor)

  # GSEC Tensor
  dispatch_tensor = tf.cast(
      tf.cast(combine_tensor, tf.bool), fprop_dtype, name='dispatch_tensor')
  dispatch_tensor = _MaybeSplit(dispatch_tensor)

  # TODO(yonghui): compute and return per-group aux_loss.
  return aux_loss, combine_tensor, dispatch_tensor


def Top2Gating(w,
               inputs,
               paddings,
               num_devices,
               experts_dim,
               expert_capacity_dim,
               local_dispatch,
               fprop_dtype,
               use_xla_sharding=True,
               second_expert_policy='all',
               second_expert_threshold=0.0,
               legacy_mtf_behavior=True,
               capacity_factor=None,
               model_dim_reshape_segments=None,
               mask_dtype=None,
               gating_logits_dtype=None):
  """Computes Top-2 gating for Mixture-of-Experts.

  See Top2GatingOnLogits for more details.

  Note that for local_dispatch original batch BLM is reshaped into GSM, each
  group `g = 0...G-1` is being dispatched independently.

  Args:
    w: gating weights for each experts with shape ME. w was reshaped accordingly
        if model_dim_reshape_segments is not None,
    inputs: G`SM Tensor.
    paddings: G`S Tensor.
    num_devices: number of MoE devices for local dispatch
    experts_dim: number of experts.
    expert_capacity_dim: number of examples per minibatch(group) per expert.
      Each example is typically a vector of size input_dim, representing
      embedded token or an element of Transformer layer output.
    local_dispatch: whether dispatch is local to the group (G dim)
    fprop_dtype: activations datatype to use.
    use_xla_sharding: bool, True if this function is used for the xla_sharding
        case.
    second_expert_policy: 'all' or 'random', we optionally 'random'-ize dispatch
      to second-best expert proportional to (weight / second_expert_threshold).
    second_expert_threshold: threshold for probability normalization for
      second_expert_policy == 'random'.
    legacy_mtf_behavior: True for legacy behavior with no re-normalization of
      expert assignment weights if we go over capacity or randomly decide to not
      dispatch to second expert.
    capacity_factor: if set, increases expert_capacity_dim to at least
      `(group_size * capacity_factor) / experts_dim`
      where `group_size` is the size of G dimension of `inputs`. If the
      value of expert_capacity_dim is already big enough no change is made.
    model_dim_reshape_segments: none or a list, reshaping model dimension M to
      that + [-1]
    mask_dtype: using bfloat16 for fprop_dtype could be problematic for mask
      tensors, mask_dtype is a special dtype for such tensors.
    gating_logits_dtype: using bfloat16 for fprop_dtype could be problematic for
      gating logits, gating_logits_dtype is a special dtype for such tensors.

  Returns:
    A tuple (dispatch_tensor, combine_tensor, aux_loss).

    - dispatch_tensor: G`SEC Tensor, scattering/dispatching inputs to
      experts.
    - combine_tensor: G`SEC Tensor.
      combining expert outputs.
    - aux_loss: auxiliary loss, equalizing the expert assignment ratios.
  """
  orig_inputs = inputs
  if not local_dispatch:
    inputs = tf.reshape(inputs, [1, inputs.shape[0] * inputs.shape[1], -1])
    if paddings is not None:
      paddings = tf.reshape(paddings, [1, -1])

  if gating_logits_dtype is None or gating_logits_dtype == fprop_dtype:
    logits = EinsumWithModelDim('GSM,ME->GSE', inputs, w,
                                model_dim_reshape_segments)
  else:
    logits = EinsumWithModelDim(
        'GSM,ME->GSE',
        tf.cast(inputs, gating_logits_dtype),
        tf.cast(w, gating_logits_dtype),
        model_dim_reshape_segments,
        name='gating_logits_with_custom_dtype')

  top1_expert_per_example = tf.math.argmax(logits, -1)

  tpu_summary.tensor('top1_expert', top1_expert_per_example)

  aux_loss, combine_tensor, dispatch_tensor = Top2GatingOnLogits(
      inputs, paddings, logits, num_devices, experts_dim, expert_capacity_dim,
      fprop_dtype, use_xla_sharding, second_expert_policy,
      second_expert_threshold, legacy_mtf_behavior, capacity_factor, None,
      mask_dtype)

  if not local_dispatch:
    dispatch_tensor = tf.reshape(
        dispatch_tensor, orig_inputs.shape[:2] + dispatch_tensor.shape[2:])
    combine_tensor = tf.reshape(
        combine_tensor, orig_inputs.shape[:2] + combine_tensor.shape[2:])

  return py_utils.NestedMap(
      combine_tensor=combine_tensor,
      dispatch_tensor=dispatch_tensor,
      aux_loss=aux_loss)


def FeedForwardNetworksApplyGating(gating,
                                   inputs,
                                   reshaped_inputs,
                                   wi_split,
                                   wo_split,
                                   num_devices,
                                   num_groups,
                                   bi_split=None,
                                   bo_split=None,
                                   dropout_rate=0.0,
                                   device_mesh=None,
                                   gsm_split=None,
                                   egcm_split=None,
                                   gecm_split=None,
                                   gsec_split=None,
                                   eah_split=None,
                                   eam_split=None,
                                   model_dim_reshape_segments=None,
                                   activation_name='RELU'):
  """Apply top_2 gating to feedforward networks.

  Args:
    gating: returns from Top2Gating consisting of: dispatch_tensor, G`SEC
      Tensor, scattering/dispatching inputs to experts. combine_tensor, G`SEC
      Tensor, combining expert outputs. aux_loss. auxiliary loss, equalizing the
      expert assignment ratios
    inputs: G`SM Tensor.
    reshaped_inputs: G`SM Tensor.
    wi_split: First projection weights [E, M, H] of the feedforward networks.
    wo_split: Last projection weights [E, H, M] of the feedforward networks.
    num_devices: number of devices.
    num_groups: number of groups (generally matches to or proportional to
      num_devices).
    bi_split: First projection bias [E, 1, H] of the feedforward networks.
    bo_split: Last projection bias [E, 1, M] of the feedforward networks.
    dropout_rate: Dropout rate.
    device_mesh: Device mesh as a numpy ND array of device IDs. Split arguments
      must be set if device_mesh is not None.
    gsm_split: Mesh split for GSM tensors.
    egcm_split: Mesh split for EGCM tensors.
    gecm_split: Mesh split for GECM tensors.
    gsec_split: Mesh split for GSEC tensors.
    eah_split: Mesh split for EAH tensors.
    eam_split: Mesh split for EAM tensors.
    model_dim_reshape_segments: Reshaping model dimension M to that + [-1]
    activation_name: Default: `RELU`. Activation function for feed-forward.

  Returns:
    outputs: G`SM Tensor.
    aux_loss: scalar auxiliary loss.
  """
  if device_mesh is not None:
    assert gsm_split is not None
    assert egcm_split is not None
    assert gecm_split is not None
    assert gsec_split is not None
    assert eah_split is not None
    assert eam_split is not None

  def _Einsum(eq, x, y, name=None):
    return EinsumWithModelDim(eq, x, y, model_dim_reshape_segments, name)

  def _NewOrHistoricSplit(t, t_split):
    if device_mesh is not None:
      tf.logging.info('MeshSplit %s %s %s', t, device_mesh.shape, t_split)
      return gshard_utils.MeshSplit(t, device_mesh, t_split)
    return gshard_utils.Split(t, 0, num_devices)

  # dispatch_tensor: G`SEC
  expert_inputs = _Einsum(
      'GSEC,GSM->EGCM',
      _NewOrHistoricSplit(gating.dispatch_tensor, gsec_split),
      _NewOrHistoricSplit(reshaped_inputs, gsm_split),
      name='expert_inputs_egcm')
  expert_inputs = _NewOrHistoricSplit(expert_inputs, egcm_split)

  # pylint: disable=invalid-name
  if model_dim_reshape_segments is None:
    M = py_utils.GetShape(reshaped_inputs)[-1:]
  else:
    M = py_utils.GetShape(reshaped_inputs)[2:]
  E = py_utils.GetShape(expert_inputs)[0]

  # combine_tensor: G`SEC
  G = py_utils.GetShape(gating.combine_tensor)[0]
  # allow evaler/decoder to run.
  del num_groups
  C = py_utils.GetShape(gating.combine_tensor)[-1]
  A = G * C
  # pylint: enable=invalid-name

  # Reshaping EGCM => EAM where A = G*C, e.g.
  #
  # with E=512, G=1024
  #
  # (512, 1024, 4, 1024) => (512, 4096, 1024)
  expert_inputs = tf.reshape(
      expert_inputs, [py_utils.GetShape(expert_inputs)[0], A] + M,
      name='expert_inputs_eam')
  expert_inputs = _NewOrHistoricSplit(expert_inputs, eam_split)

  h = _Einsum('EAM,EMH->EAH', expert_inputs, wi_split, name='h_eah')
  h = _NewOrHistoricSplit(h, eah_split)
  if bi_split is not None:
    h += Split(bi_split, 0, num_devices)
    h = Split(h, 0, num_devices)

  h = activations.GetFn(activation_name)(h)
  if dropout_rate:
    # we generally do not use stateless dropout in MoE since it introduces
    # large uint32 tensor broadcast (per dehao@ study)
    h = tf.nn.dropout(h, dropout_rate)

  expert_outputs = _Einsum(
      'EAH,EHM->EAM', h, wo_split, name='expert_outputs_eam')
  expert_outputs = _NewOrHistoricSplit(expert_outputs, eam_split)

  if bo_split is not None:
    expert_outputs = gshard_utils.Split(expert_outputs, 0, num_devices)
    expert_outputs += gshard_utils.Split(bo_split, 0, num_devices)
    expert_outputs = gshard_utils.Split(expert_outputs, 0, num_devices)
  expert_outputs = tf.reshape(expert_outputs, [E, G, C] + M)

  # same as tf.transpose
  expert_outputs = tf.einsum(
      _EinsumEqWithModelDim('EGCM->GECM', model_dim_reshape_segments),
      expert_outputs,
      name='expert_outputs_gecm')

  combined_outputs = _Einsum(
      'GSEC,GECM->GSM',
      _NewOrHistoricSplit(gating.combine_tensor, gsec_split),
      _NewOrHistoricSplit(expert_outputs, gecm_split),
      name='combined_outputs_gsm')
  outputs = _NewOrHistoricSplit(
      tf.reshape(combined_outputs, py_utils.GetShape(inputs)), gsm_split)
  aux_loss = gating.aux_loss
  return outputs, aux_loss


def GatherK(selected_pos, values, k, num_devices=1):
  """Gather up to k elements from given tensors at selected pos under SPMD.

  Example::

    # Input
    k = 3

    selected_pos = [
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],  # topk(k=3) largest indices are selected in this row.
    ]

    value_2d = [
        [1, 3, 5, 7],
        [9, 11, 13, 15],
        [17, 19, 21, 23],
        [25, 27, 29, 31],
        [33, 35, 37, 39],
    ]

    # Output:
    output = [
        [0, 5, 7],
        [0, 11, 13],
        [0, 0, 0],
        [25, 27, 29],
        [35, 37, 39],
    ]

    # Output padding:
    output_padding = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    ]

  Args:
    selected_pos: a 0/1 2D tf.int32 tensor of shape [batch, time].
    values: a list of tensors, the rank of each is at least rank=2. [batch,
      time, ...].
    k: a scalar tf.int32 tensor or a Python int. On TPU, k must be a
      compile-time constant.
    num_devices: number of TPU devices used in xla_sharding SPMD.

  Returns:
    A tuple (output, padding).

    - output: a list of tensors of shape [batch, k, ...].
    - padding: a 2D 0/1 tensor of shape [batch, k], '1's are padded locations.
  """
  global_batch, seq_len = py_utils.GetShape(selected_pos, 2)
  if num_devices:
    device_batch = global_batch // num_devices
  else:
    device_batch = global_batch

  for i in range(len(values)):
    # Assert the first 2 dim of values[i] is [global_batch, seq_len]
    values[i] = py_utils.HasShape(values[i], [global_batch, seq_len], 2)
  # indices are 1-based for now, to distinguish between padding and selected
  # locations.
  indices = 1 + tf.range(tf.shape(values[0])[1], dtype=tf.int32)
  # [1, seq_len]
  indices = tf.expand_dims(indices, axis=0)

  # if 0, the position is not selected.
  # [1, seq_len] * [global_batch, seq_len] => [global_batch, t]
  # -- topk --> [global_batch, k]
  topk_indices, _ = tf.math.top_k(
      indices * tf.cast(selected_pos, indices.dtype), k)

  # [global_batch, k], sorted in ascending order.
  indices = tf.reverse(topk_indices, [-1])
  # [global_batch, k], padded positions are '1's.
  padding = tf.cast(tf.equal(indices, 0), values[0].dtype)
  padding = gshard_utils.Split(padding, 0, num_devices)

  # [global_batch, k], zero_based_indices
  mp_idx = tf.maximum(0, indices - 1)
  mp_idx = gshard_utils.Split(mp_idx, 0, num_devices)

  # [device_batch, k]
  if num_devices > 1 and py_utils.use_tpu():
    mp_idx = xla_sharding.auto_to_manual_spmd_partition(
        mp_idx, xla_sharding.get_op_sharding(mp_idx.op))
  # [device_batch, k, 1]
  mp_idx = tf.expand_dims(mp_idx, -1)

  # [device_batch]
  batch_ids = tf.range(device_batch, dtype=tf.int32)
  # [device_batch, 1, 1]
  batch_ids = tf.reshape(batch_ids, [device_batch, 1, 1])
  # [device_batch, k, 1]
  batch_ids = tf.broadcast_to(batch_ids, [device_batch, k, 1])

  # [device_batch, k, 2]
  final_indices = tf.concat([batch_ids, mp_idx], axis=-1)

  output = []
  for v in values:
    # Begin manually partition gather.
    v = gshard_utils.Split(v, 0, num_devices)
    v_shape = v.shape.as_list()
    if num_devices > 1 and py_utils.use_tpu():
      op_sharding = xla_sharding.get_op_sharding(v.op)
      v = xla_sharding.auto_to_manual_spmd_partition(v, op_sharding)
    # Returns [global_batch, k, ...]
    v_out = tf.gather_nd(v, final_indices)

    if num_devices > 1 and py_utils.use_tpu():
      v_shape[1] = k
      v_out = xla_sharding.manual_to_auto_spmd_partition(
          v_out, op_sharding, full_shape=tf.TensorShape(v_shape))
    output.append(v_out)

  return output, padding


def GetSentenceEmbeddings(inputs, segment_id):
  """Returns the average sentence embedding to gate by.

  Example::

    inputs: <tf.Variable 'Variable:0' shape=(10, 3) dtype=float64, numpy=
             array([[0.41258181, 0.61071571, 0.63777673],
                    [0.65571443, 0.54297766, 0.10288261],
                    [0.8577837 , 0.81915847, 0.61996602],
                    [0.46897136, 0.92662692, 0.32942232],
                    [0.60162383, 0.3385829 , 0.3408632 ],
                    [0.40774807, 0.86139635, 0.00927162],
                    [0.56126334, 0.51748817, 0.07791397],
                    [0.06595223, 0.95529216, 0.34458149],
                    [0.1238971 , 0.49897169, 0.25216722],
                    [0.11221774, 0.50284604, 0.84106974]])>
    segment_id: <tf.Variable 'Variable:0' shape=(10,) dtype=int64,
                 numpy=array([1, 1, 2, 0, 0, 3, 3, 3, 3, 0])>

  Args:
    inputs: G`SM Tensor.
    segment_id: G`S Tensor.

  Returns:
    sentence_embeddings: GSM Tensor that is an average of the input embeddings
    per segment.
  """
  reshaped_inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])

  # We set num_segments to a large value so that shape is known at compile time.
  max_segments = py_utils.GetShape(reshaped_inputs)[0]
  # We change the padding to be max_segments - 1 instead of 0 because
  # tf.math.unsorted_segment_mean because it only accepts values between 1 and
  # max_segments.
  modified_segment_id = tf.cast(
      segment_id + max_segments * tf.cast(
          tf.equal(segment_id, 0), dtype=tf.dtypes.as_dtype(segment_id.dtype)) -
      1,
      dtype=tf.int32)
  reshaped_segment_id = tf.reshape(modified_segment_id, [-1])

  # Takes the mean of all segments, w/ 0s for the padding.
  params = tf.concat([
      tf.math.unsorted_segment_mean(reshaped_inputs, reshaped_segment_id,
                                    max_segments)[:-1],
      tf.zeros([1, reshaped_inputs.shape[-1]], dtype=reshaped_inputs.dtype)
  ],
                     axis=0)
  raw_sentence_embeddings = tf.gather(params, modified_segment_id)

  # sentence_embedding: <tf.Tensor: shape=(10, 3), dtype=float64, numpy=
  #                     array([[0.92657252, 0.40264503, 0.55494457],
  #                            [0.92657252, 0.40264503, 0.55494457],
  #                            [0.08002721, 0.02360659, 0.63688627],
  #                            [0.        , 0.        , 0.        ],
  #                            [0.        , 0.        , 0.        ],
  #                            [0.8138629 , 0.54451293, 0.48802852],
  #                            [0.8138629 , 0.54451293, 0.48802852],
  #                            [0.8138629 , 0.54451293, 0.48802852],
  #                            [0.8138629 , 0.54451293, 0.48802852],
  #                            [0.        , 0.        , 0.        ]])>
  sentence_embeddings = tf.reshape(raw_sentence_embeddings, inputs.shape)

  return sentence_embeddings


def SentenceTop2Gating(w,
                       inputs,
                       paddings,
                       segment_id,
                       num_devices,
                       experts_dim,
                       expert_capacity_dim,
                       local_dispatch,
                       fprop_dtype,
                       use_xla_sharding=True,
                       second_expert_policy='all',
                       second_expert_threshold=0.0,
                       legacy_mtf_behavior=True,
                       embedding_type='sentence',
                       capacity_factor=None):
  """Computes Top-2 sentence gating for Mixture-of-Experts.

  Instead of using the each token, this function uses embedding_type to return a
  sentence-wise embedding to create dispatch and combine tensors that gate
  the entire sentence.

  Note that for local_dispatch original batch BLM is reshaped into GSM, each
  group `g = 0...G-1` is being dispatched independently.

  Args:
    w: gating weights for each experts.
    inputs: G`SM Tensor.
    paddings: G`S Tensor.
    segment_id: G`SM Tensor used for differentiating different sentences in an
      input example.
    num_devices: number of MoE devices for local dispatch
    experts_dim: number of experts.
    expert_capacity_dim: number of examples per minibatch(group) per expert.
      Each example is typically a vector of size input_dim, representing
      embedded token or an element of Transformer layer output.
    local_dispatch: whether dispatch is local to the group (G dim)
    fprop_dtype: activations datatype to use.
    use_xla_sharding: bool, True if this function is used for the xla_sharding
      case.
    second_expert_policy: 'all' or 'random', we optionally 'random'-ize dispatch
      to second-best expert proportional to (weight / second_expert_threshold).
    second_expert_threshold: threshold for probability normalization for
      second_expert_policy == 'random'.
    legacy_mtf_behavior: True for legacy behavior with no re-normalization of
      expert assignment weights if we go over capacity or randomly decide to not
      dispatch to second expert.
    embedding_type: 'sentence' by default. Options: 'sentence'. Setting this
      option calls GetSentenceEmbeddings.
    capacity_factor: if set, increases expert_capacity_dim to at least
      (group_size * capacity_factor) / experts_dim where `group_size` is the
      size of G dimension of `inputs`. If the value of expert_capacity_dim is
      already big enough no change is made.

  Returns:
    A tuple (dispatch_tensor, combine_tensor, aux_loss).

    - dispatch_tensor: G`SEC Tensor, scattering/dispatching inputs to
      experts.
    - combine_tensor: G`SEC Tensor.
      combining expert outputs.
    - aux_loss: auxiliary loss, equalizing the expert assignment ratios.
  """
  assert embedding_type in ['sentence']
  orig_inputs = inputs

  if not local_dispatch:
    inputs = tf.reshape(inputs, [1, inputs.shape[0] * inputs.shape[1], -1])
    if paddings is not None:
      paddings = tf.reshape(paddings, [1, -1])

  if embedding_type == 'sentence':
    sentence_embeddings = GetSentenceEmbeddings(inputs, segment_id)

  logits = tf.einsum('GSM,ME->GSE', sentence_embeddings, w)
  aux_loss, combine_tensor, dispatch_tensor = Top2GatingOnLogits(
      sentence_embeddings, paddings, logits, num_devices, experts_dim,
      expert_capacity_dim, fprop_dtype, use_xla_sharding, second_expert_policy,
      second_expert_threshold, legacy_mtf_behavior, capacity_factor)

  if not local_dispatch:
    dispatch_tensor = tf.reshape(
        dispatch_tensor, orig_inputs.shape[:2] + dispatch_tensor.shape[2:])
    combine_tensor = tf.reshape(
        combine_tensor, orig_inputs.shape[:2] + combine_tensor.shape[2:])

  return py_utils.NestedMap(
      combine_tensor=combine_tensor,
      dispatch_tensor=dispatch_tensor,
      aux_loss=aux_loss)


def TaskTop2Gating(w,
                   inputs,
                   paddings,
                   task_embeddings,
                   num_devices,
                   experts_dim,
                   expert_capacity_dim,
                   local_dispatch,
                   fprop_dtype,
                   use_xla_sharding=True,
                   second_expert_policy='all',
                   second_expert_threshold=0.0,
                   legacy_mtf_behavior=True):
  """Computes Top-2 sentence gating for Mixture-of-Experts.

  Instead of using the each token, this function uses embedding_type to return a
  sentence-wise embedding to create dispatch and combine tensors that gate
  the entire sentence.

  Note that for local_dispatch original batch BLM is reshaped into GSM, each
  group `g = 0...G-1` is being dispatched independently.

  Args:
    w: gating weights for each experts.
    inputs: G`SM Tensor.
    paddings: G`S Tensor.
    task_embeddings: G`SM Tensor.
    num_devices: number of MoE devices for local dispatch
    experts_dim: number of experts.
    expert_capacity_dim: number of examples per minibatch(group) per expert.
      Each example is typically a vector of size input_dim, representing
      embedded token or an element of Transformer layer output.
    local_dispatch: whether dispatch is local to the group (G dim)
    fprop_dtype: activations datatype to use.
    use_xla_sharding: bool, True if this function is used for the xla_sharding
      case.
    second_expert_policy: 'all' or 'random', we optionally 'random'-ize dispatch
      to second-best expert proportional to (weight / second_expert_threshold).
    second_expert_threshold: threshold for probability normalization for
      second_expert_policy == 'random'.
    legacy_mtf_behavior: True for legacy behavior with no re-normalization of
      expert assignment weights if we go over capacity or randomly decide to not
      dispatch to second expert.

  Returns:
    A tuple (dispatch_tensor, combine_tensor, aux_loss):

    - dispatch_tensor: G`SEC Tensor, scattering/dispatching inputs to
      experts.
    - combine_tensor: G`SEC Tensor.
      combining expert outputs.
    - aux_loss: auxiliary loss, equalizing the expert assignment ratios.
  """
  orig_inputs = inputs
  if not local_dispatch:
    inputs = tf.reshape(inputs, [1, inputs.shape[0] * inputs.shape[1], -1])
    task_embeddings = tf.reshape(
        task_embeddings,
        [1, task_embeddings.shape[0] * task_embeddings.shape[1], -1])
    if paddings is not None:
      paddings = tf.reshape(paddings, [1, -1])

  logits = tf.einsum('GSM,ME->GSE', task_embeddings, w)
  aux_loss, combine_tensor, dispatch_tensor = Top2GatingOnLogits(
      task_embeddings, paddings, logits, num_devices, experts_dim,
      expert_capacity_dim, fprop_dtype, use_xla_sharding, second_expert_policy,
      second_expert_threshold, legacy_mtf_behavior)

  if not local_dispatch:
    dispatch_tensor = tf.reshape(
        dispatch_tensor, orig_inputs.shape[:2] + dispatch_tensor.shape[2:])
    combine_tensor = tf.reshape(
        combine_tensor, orig_inputs.shape[:2] + combine_tensor.shape[2:])

  return py_utils.NestedMap(
      combine_tensor=combine_tensor,
      dispatch_tensor=dispatch_tensor,
      aux_loss=aux_loss)


def _EinsumEqWithModelDim(equation, model_dim_reshape_segments):
  """Adjust Einsum equation according to model_dim_reshape_segments."""
  if model_dim_reshape_segments is None:
    return equation
  assert len(model_dim_reshape_segments) <= 2
  insert_chars = 'N' if len(model_dim_reshape_segments) == 1 else 'NO'
  new_equation = ''
  for c in equation:
    assert c not in insert_chars
    if c == 'M':
      new_equation += insert_chars
    new_equation += c
  return new_equation


def EinsumWithModelDim(equation, x, y, model_dim_reshape_segments, name=None):
  """Einsum with adjusted equation according to model_dim_reshape_segments.

  It changes each dimension named 'M' in the equation into two dimensions 'NM'
  if model_dim_reshape_segments is set in the params. Therefore the original
  equation should not have 'N', and only use 'M' when it is expected to be
  reshaped.

  For example, an input equation 'GSM,ME->GSE' and model_dim_reshape_segments
  # [16, 4] will be rewritten into the new equation 'GSNOM,NOME->GSE'.

  Args:
    equation: a string describing the contraction, in the same format as
      numpy.einsum.
    x: First input to einsum.
    y: second input to einsum.
    model_dim_reshape_segments: Reshaping model dimension M to that + [-1]
    name: optional name.

  Returns:
    tf.einsum(maybe_modified_equation, x, y)
  """
  if model_dim_reshape_segments is None:
    return tf.einsum(equation, x, y, name=name)
  new_equation = _EinsumEqWithModelDim(equation, model_dim_reshape_segments)
  return tf.einsum(new_equation, x, y, name=name)
