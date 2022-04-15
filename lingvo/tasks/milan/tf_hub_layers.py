# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Milan layers."""

from typing import Collection, List

from lingvo import compat as tf
from lingvo.core import base_layer
import tensorflow_hub as hub


def _WrapNonLingvoVars(dest_layer: base_layer.BaseLayer,
                       variables: Collection[tf.Variable],
                       trainable_variables: Collection[tf.Variable] = ()):
  """Adds variables to the given lingvo layer and appropriate graph collections.

  This function helps wrap variables created outside of lingvo so they are
  correctly handled by lingvo's trainer and checkpointer. It does the following:

    - makes all `variables` trackable through `dest_layer.vars`;
    - ensures `variables` are in the `tf.global_variables()` graph collection so
      the trainer can initialize them;
    - adds the `trainable_variables` subset to the `tf.trainable_variables()`
      graph collection, so they are visible to the learner (i.e. can be
      trained).

  Args:
    dest_layer: Lingvo layer to add the `variables` to.
    variables: The non-lingvo variables to wrap.
    trainable_variables: The subset of `variables` to ensure are trainable.
  """

  global_collection = set(tf.global_variables())
  for v in variables:
    assert v in global_collection
    name = v.name.split(':')[0]
    # pylint: disable=protected-access
    dest_layer._private_vars[name] = v
    with tf.device(v.device):
      dest_layer._private_theta[name] = tf.identity(v)
    # pylint: enable=protected-access

  trainable_collection = set(tf.trainable_variables())
  for v in trainable_variables:
    if v not in trainable_collection:
      tf.logging.warning(
          'Wrapped var %s not in trainable collection; adding it.', v.name)
      tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                     v)


def _AddUpdateOpsToSignature(module, signature='default'):
  """Creates a copy of a TF1 module signature that triggers stateful update ops.

  This helper is intended for TF1-style hub modules (i.e. those loaded via
  `hub.load()`) that are to be fine-tuned.  It creates a copy of
  `module.signatures[signature]` that is functionally equivalent except that it
  also runs update ops (e.g. updates to batch norm statistics) as a side effect.

  (The signatures of TF1 hub modules typically don't run such ops: they're often
  not dependencies of the signature outputs, and are therefore pruned when the
  module is loaded.)

  Args:
    module: A TF1-style hub module, in the format returned by `hub.load()`.
    signature: Name of the signature of `module` to wrap.

  Returns:
    A callable with the same interface as `module.signatures[signature]`.
  """
  fn = module.signatures[signature]
  # Determine names of the function's input and output Tensors, plus those of
  # any update ops that were in the graph prior to pruning but now only exist in
  # the parent graph (module.graph). Then construct a new `WrappedFunction` that
  # computes the same outputs from the same inputs, but also runs the update ops
  # as a side effect.
  captured_input_names = {t.name for t in fn.captured_inputs}
  input_names = [
      t.name for t in fn.inputs if t.name not in captured_input_names
  ]
  structured_output_names = tf.nest.pack_sequence_as(
      fn.structured_outputs,
      [t.name for t in tf.nest.flatten(fn.structured_outputs)])
  update_op_names = [
      op.name
      for op in fn.graph.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
  ]
  fn_with_update_ops = module.prune(input_names,
                                    (structured_output_names, update_op_names))

  def _CallAndDropUpdateOpOutputs(*args):
    outputs, _ = fn_with_update_ops(*args)
    return outputs

  return _CallAndDropUpdateOpOutputs


class ImageModule(base_layer.BaseLayer):
  """Wraps a TF-hub image module, allowing it to be fine-tuned.

  NOTE: Input images are assumed to be [0, 1] normalized as per TF-hub standard
  (see https://www.tensorflow.org/hub/common_signatures/images).


  Example usage::

    inception_model = ImageModule.Params().Set(
        name='inception_v3',
        module_path=(
            'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4'),
        update_batch_norm=True).Instantiate()
    images = tf.random.uniform([2, 299, 299, 3], minval=0.0, maxval=1.0)
    features = inception_model(images)
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('module_path', '', 'Path / handle of tf-hub module to load.')
    p.Define('signature', 'default', 'Module signature to run.')
    p.Define(
        'training_graph_tags', {'train'},
        'Tags of the module graph to use for training. Conventionally '
        'the "train"-tagged graph runs components like batch norm and '
        'dropout in training mode, and has extra ops to update batch norm '
        'moving averages, etc.')
    p.Define('eval_graph_tags', set(),
             'Tags of the module graph to use for evaluation.')
    p.Define(
        'run_update_ops', True,
        'Run update ops (e.g. to update batch-norm statistics) during '
        'fine-tuning?')
    return p

  def __init__(self, params):
    if not params.module_path:
      raise ValueError('Required param "module_path" not set.')
    super().__init__(params)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    # In TF1 modules the convention is that the 'train'-tagged graph runs
    # batch norm (if in use) in "training" mode, and adds additional
    # UPDATE_OPS to the graph to update the moving averages.
    tags = p.eval_graph_tags if self.do_eval else p.training_graph_tags
    tf.logging.info('Loading module with tags %s', tags)
    self._module = hub.load(p.module_path, tags=tags)

    # Extract the signature, and create a variant that also runs update ops in
    # the graph.
    self._fwd = self._module.signatures[p.signature]
    self._fwd_with_update_ops = _AddUpdateOpsToSignature(
        self._module, p.signature)
    self._reg_losses_fn = self._module.prune(
        [], self._module.graph.get_collection('regularization_losses'))
    _WrapNonLingvoVars(self, variables=self._module.variables)

  @property
  def losses(self) -> List[tf.Tensor]:
    return self._reg_losses_fn()

  def FProp(self, _, images):
    images.shape.assert_has_rank(4)
    p = self.params

    if not self.do_eval and p.run_update_ops:
      tf.logging.info('Running module in training mode with update ops.')
      return self._fwd_with_update_ops(images)['default']
    else:
      return self._fwd(images)['default']


class ImageModuleV2(base_layer.BaseLayer):
  """Wraps a TF2 hub image module, allowing it to be fine-tuned.

  Any regularization losses attached to the hub module are accessible through
  this object's `losses` property.

  NOTE: Input images are assumed to be [0, 1] normalized as per TF-hub standard
  (see https://www.tensorflow.org/hub/common_signatures/images).

  Example usage::

    params = ImageModuleV2.Params().Set(
        name='efficientnet_b4',
        module_path=(
            'https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1'))
    model = params.Instantiate()
    images = tf.random.uniform([2, 380, 380, 3], minval=0.0, maxval=1.0)
    features = model(images)
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'image_module_v2'
    p.Define('module_path', '', 'Path/handle of tf-hub module to load.')
    p.Define(
        'use_training_mode', True,
        'Iff True, the module is run in "training" mode when being fine-tuned. '
        'This typically turns on dropout-like regularization methods and '
        'causes batch norm statistics to be updated during fine-tuning.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    for required_param_name in ('name', 'module_path'):
      if not p.Get(required_param_name):
        raise ValueError(f'Must set {required_param_name} param.')

    # This name_scope is for checkpoint backwards-compatibility.
    with tf.name_scope(self._self_variable_scope.original_name_scope):
      # NB: `trainable` merely controls whether the model *can* be run in
      # training mode.
      self._module = hub.KerasLayer(p.module_path, trainable=True)

    _WrapNonLingvoVars(
        self,
        variables=self._module.variables,
        trainable_variables=self._module.trainable_variables)

    # Functionalize the module's __call__ so train-mode update ops run eagerly.
    self._hub_module_fn = tf.function(
        lambda images, training: self._module(images, training=training))

  @property
  def losses(self) -> List[tf.Tensor]:
    return self._module.losses

  def FProp(self, _, images):
    return self._hub_module_fn(
        images, training=self.params.use_training_mode and not self.do_eval)


EFFICIENTNET_B4_INPUT_SHAPE = (380, 380)
EFFICIENTNET_B4_OUTPUT_FEATURE_DIM = 1792


def EfficientNetB4Params():
  """Sets up params for loading EfficientNet-B4 from tf-hub."""
  return ImageModuleV2.Params().Set(
      name='efficientnet_b4',
      module_path=(
          'https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1'))
