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
"""A generic repeat layer.

Adapted from builder_layers.RepeatLayer.

A repeat layer consists a stack of `repeat` identical sub layers, where
  * The variables are stacked across layers. Each stacked variable has shape
    [repeat, ...].
  * The computation is performed with a recurrent loop across layers.

Compared with a layer stack, a repeat layer's TF graph size does not grow
proportional to the number of layers. It also reduces HBM usage but incurs
additional computation through rematerialization.

GenericRepeatLayer._Repeat() allows its subclasses to describe arbitrary
computation across sub layers.

Inputs to repeat layer computation fall into three categories:

  * common_input: shared tensor or numeric inputs to all sub layers, e.g.,
      aux_vec for cross attention, batch_size for creating zero state.
  * layerwise_inputs: separate inputs for each sub layer, specified by tensors
      of shape [repeat, ...], where T[i, ...] is the input for sub layer i,
      e.g., cached_states for ExtendStep. In particular, 'theta' is also a type
      of layerwise input, but in API we treat it separately for convenience.
  * iterative_input_0: iterative input to the first sub layer, e.g., hidden
      vectors.

The output of a sub layer can include:

  * layerwise: layerwise outputs, to be stacked for the final output, e.g.,
      updated_states from ExtendStep.
  * iterative: iterative input for the next sub layer.

The final output of a repeat layer will include:

  * layerwise: stacked tensors of layerwise outputs, of shape [repeat, ...],
      where T[i, ...] is a layerwise output of sub layer i.
  * iterative: the iterative output of the final sub layer.

In pseudo code::

  def _Repeat(theta, fn, common_input, layerwise_inputs, iterative_input_0):
    iterative = iterative_input_0
    for i in range(p.repeat):
      fn_out = fn(theta.body[i, ...],
                  iterative=iterative,
                  common_input=common_input,
                  layerwise_input=layerwise_inputs[i, ...])
      layerwise_outputs[i, ...] = fn_out.layerwise_output
      iterative = fn_out.iterative
    return NestedMap(iterative=iterative, layerwise=layerwise_outputs)
"""

from lingvo import compat as tf

from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import recurrent


def _CopyShapes(src, dst):
  tf.nest.map_structure(lambda s, t: t.set_shape(s.shape), src, dst)
  return dst


class GenericRepeatLayer(base_layer.BaseLayer):
  """A layer which repeats itself sequentially using lingvo Recurrent."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('body', None, 'The param for the main network layer.')
    p.Define('per_layer_vars', False, 'Use separate variables for each layer')
    p.Define('repeat', 1,
             'Repeat layers specified in \'body\' this many times.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name
    assert p.repeat > 0
    if p.per_layer_vars:
      for i in range(p.repeat):
        self.CreateChild('body_iter_%05d' % i, p.body)
    else:
      self.CreateChild('body', p.body)

  @property
  def _body(self):
    """A child layer to be used as the loop body."""
    p = self.params
    if p.per_layer_vars:
      return self.body_iter_00000
    else:
      return self.body

  def _CreateChildrenVariables(self):
    p = self.params
    with tf.variable_scope(p.name):
      if p.per_layer_vars:
        for i in range(p.repeat):
          with tf.variable_scope('iter_%05d' % i):
            self.children['body_iter_%05d' % i].InstantiateVariables()
      else:
        with py_utils.VariableShapePrefixContext(p.repeat):
          self.body.InstantiateVariables()
    super()._CreateChildrenVariables()

  def _MaybeStackExtraTheta(self, theta):
    p = self.params
    var_set = set([key for key, _ in self.vars.FlattenItems()])
    values = []
    for key, value in theta.FlattenItems():
      if key not in var_set and value is not None:
        # Replicate non-variable theta by p.repeat times.
        value = tf.stack([value] * p.repeat)
      values.append(value)
    return theta.Pack(values)

  def _Repeat(self,
              theta,
              fn,
              *,
              common_input=None,
              layerwise_inputs=None,
              iterative_input_0=None,
              layerwise_output_0=None):
    """Invokes 'fn' for each sub-layer.

    Args:
      theta: layer parameters.
      fn: A function with args (theta, common_input, layerwise_input, iterative)
        returning a NestedMap(layerwise_output=..., iterative=...).
      common_input: a NestedMap with common inputs for every sub-layer.
      layerwise_inputs: a nested tensor with separate inputs for each sub-layer,
        where each leaf value T is a tensor of shape [p.repeat, ...] and T[i,
        ...] represents layerwise inputs to the i'th sub-layer.
      iterative_input_0: a nested tensor for the iterative input of the 0'th
        sub-layer.
      layerwise_output_0: a nested tensor representing the layerwise output of
        the first sub layer. The actual tensor values are not used. Only the
        structure and tensor shapes are. In particular, if layerwise_output_0 is
        None, calls fn(...) to compute the output.

    Returns:
      A NestedMap with:
      - iterative: a nested tensor with the same structure as iterative_input_0
          representing the iterative output of the last sub-layer.
      - layerwise: a nested tensor where each leaf value T is a tensor of shape
          [p.repeat, ...] and T[i, ...] represents layerwise output from the
          i'th sub-layer.
    """
    p = self.params

    if common_input is None:
      common_input = py_utils.NestedMap()
    if layerwise_inputs is None:
      layerwise_inputs = py_utils.NestedMap()
    if iterative_input_0 is None:
      iterative_input_0 = py_utils.NestedMap()

    if p.per_layer_vars:
      all_iters = [theta['body_iter_%05d' % i] for i in range(p.repeat)]
      theta_stack = py_utils.NestedMap(
          body=tf.nest.map_structure(lambda *t: tf.stack(list(t)), *all_iters))
    else:
      theta_stack = self._MaybeStackExtraTheta(theta)
    theta_0 = tf.nest.map_structure(lambda t: t[0, ...], theta_stack)
    layerwise_input_0 = tf.nest.map_structure(lambda t: t[0, ...],
                                              layerwise_inputs)

    if layerwise_output_0 is None:
      # Run 'fn' for sublayer 0 to get a template for layerwise output.
      layerwise_output_0 = fn(
          theta_0,
          common_input=common_input,
          layerwise_input=layerwise_input_0,
          iterative=iterative_input_0).layerwise_output

    # Split 'common_input' to static vs. tensor values.
    static_common_input = tf.nest.map_structure(
        lambda x: None if isinstance(x, tf.Tensor) else x, common_input)
    dummy_tensor = tf.zeros([], dtype=p.dtype)
    tensor_common_input = tf.nest.map_structure(
        lambda x: x if isinstance(x, tf.Tensor) else dummy_tensor, common_input)

    def _ToRecurState(*, iterative, layerwise_output):
      """Returns a recurrent state."""
      # Make sure each value in the NestedMap is a tensor.
      recur_state = py_utils.NestedMap(
          iterative=iterative, layerwise_output=layerwise_output)
      for key, value in recur_state.FlattenItems():
        if not isinstance(value, tf.Tensor) or value.shape.rank is None:
          raise ValueError('Each value in the recurrent state must be a tensor '
                           f'with a known rank: {key}={value}')
      return recur_state

    def _CellFn(recur_theta, recur_state0, recur_input_i):
      """Recurrent cell function wrapper of body.FProp."""
      # Retrieves fprop arguments from state and sets shapes.
      theta_i = _CopyShapes(theta_0, recur_input_i.theta)
      # Use non-tensor values from 'static_common_input' if available.
      merged_common_input = tf.nest.map_structure(
          (lambda x, s, t: t if isinstance(x, tf.Tensor) else s), common_input,
          static_common_input, recur_theta)

      # Call 'fn'.
      fn_results = fn(
          theta_i,
          common_input=merged_common_input,
          layerwise_input=_CopyShapes(layerwise_input_0,
                                      recur_input_i.layerwise),
          iterative=_CopyShapes(iterative_input_0, recur_state0.iterative))

      # Pack results.
      return _ToRecurState(**fn_results), py_utils.NestedMap()

    with tf.name_scope('repeat'):
      recur_state0 = _ToRecurState(
          iterative=iterative_input_0, layerwise_output=layerwise_output_0)
      # Runs body.FProp k times using Recurrent where k = dim 0 of var_nmap.
      acc_states, final_recur_state = recurrent.Recurrent(
          theta=tensor_common_input,
          state0=recur_state0,
          inputs=py_utils.NestedMap(
              theta=theta_stack, layerwise=layerwise_inputs),
          cell_fn=_CellFn,
          allow_implicit_capture=p.allow_implicit_capture)
      # Retrieves outputs from recur_state1 and sets shapes.
      iterative_output = _CopyShapes(iterative_input_0,
                                     final_recur_state.iterative)
      # Set shapes of layer-wise outputs according to layerwise_output_0.
      layerwise_outputs = acc_states.layerwise_output
      tf.nest.map_structure(
          lambda t_stack, t_0: t_stack.set_shape([p.repeat] + t_0.shape.as_list(
          )), layerwise_outputs, layerwise_output_0)
      return py_utils.NestedMap(
          iterative=iterative_output, layerwise=layerwise_outputs)
