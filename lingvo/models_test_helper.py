# Lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Helper for models_test."""

import re
import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import bn_layers
from lingvo.core import cluster_factory
from lingvo.core import py_utils
from lingvo.core import test_utils


def _StubOutCreateVariable(variable_cache):
  """Stub out py_utils.CreateVariable to not spend time creating variables.

  Args:
    variable_cache: a dict from unique shapes to a dummy tensor of that shape.
  """

  def _CreateVariableStub(name,
                          params,
                          reuse=None,
                          trainable=True,
                          collections=None,
                          default_seed=None,
                          synchronization=None,
                          aggregation=None):
    """Return a zero tensor of the right shape instead of creating variable."""
    del reuse
    del default_seed
    del synchronization
    del aggregation
    dtype = params.dtype
    shape = py_utils.ToStaticShape(params.shape)
    # For total samples counters we have to actually create variables so that
    # we can access the 'value' attribute during construction.
    if 'total_samples' in name:
      var = tf.get_variable(
          name,
          shape,
          dtype,
          tf.constant_initializer(0),
          collections=collections,
          trainable=trainable,
          validate_shape=True)
    else:
      key = (tf.get_default_graph(), tuple(shape))
      if key in variable_cache:
        var = variable_cache[key]
      else:
        var = tf.zeros(shape, dtype)
        variable_cache[key] = var
    return var

  py_utils.CreateVariable = _CreateVariableStub


def TraverseLayer(layer, fn):
  """Traverses the layer tree and invokes fn(node) on each node.

  Args:
    layer: a BaseLayer.
    fn: a function of (layer, layer_theta) -> None.
  """
  if isinstance(layer, (list, tuple)):
    for layer_i in layer:
      TraverseLayer(layer_i, fn)
    return

  with tf.name_scope(layer.params.name):
    fn(layer)
    # Traverse all children in alphabetical order.
    for _, child in sorted(layer.children.items()):
      TraverseLayer(child, fn)


class BaseModelsTest(test_utils.TestCase):
  """Base model test class which does not define any test methods of its own."""

  def setUp(self):
    super().setUp()
    cluster_factory.SetRequireSequentialInputOrder(False).__enter__()
    self._variable_cache = {}
    _StubOutCreateVariable(self._variable_cache)

  def _ValidateEMA(self, name, mdl):
    if not mdl.ema:
      return
    self.assertIsInstance(mdl, base_model.SingleTaskModel)
    for task in mdl.tasks:
      tp = task.params.train
      # If the model has explicitly specified ema_decay_moving_vars to
      # True or False, then we assume they understand the implication
      # of that choice.
      if tp.ema_decay_moving_vars is not None:
        # ema_decay_moving_vars is set explicitly.
        continue
      # Otherwise the model should not contain any BatchNormLayer.
      #
      # If a model fails this test, the user should explicitly specify
      # ema_decay_moving_vars, or ensure no BatchNormLayers are in their model.
      all_layers = []
      TraverseLayer(task, all_layers.append)
      batch_norm_layers = [
          layer.path
          for layer in all_layers
          if isinstance(layer, bn_layers.BatchNormLayer)
      ]
      self.assertEqual([], batch_norm_layers)

  def _testOneModelParams(self, registry, name):
    with tf.Graph().as_default():
      p = registry.GetParams(name, 'Train')
      self.assertTrue(issubclass(p.cls, base_model.BaseModel))
      self.assertIsNot(p.model, None)
      p.cluster.mode = 'sync'
      p.cluster.job = 'decoder'
      p.cluster.decoder.replicas = 1
      with p.cluster.Instantiate():
        # Instantiate the params class, to help catch errors in layer
        # constructors due to misconfigurations.
        mdl = p.Instantiate()
        self._ValidateEMA(name, mdl)
        p = mdl.params

      for dataset in ('Train', 'Dev', 'Test'):
        try:
          input_p = registry.GetParams(name, dataset).input
        except base_model_params.DatasetError:
          # Dataset not defined.
          if dataset == 'Dev':  # Dev can be optional.
            pass
          else:
            raise
        if issubclass(p.cls, base_model.SingleTaskModel):
          self.assertTrue(
              issubclass(input_p.cls, base_input_generator.BaseInputGenerator),
              'Error in %s' % dataset)
          if (dataset != 'Train') and issubclass(
              input_p.cls,
              base_input_generator.BaseSequenceInputGenerator) and (
                  input_p.num_samples != 0):
            self.assertEqual(
                input_p.num_batcher_threads, 1,
                'num_batcher_threads too large in %s. Decoder '
                'or eval runs over this set might not span '
                'exactly one epoch.' % dataset)
        else:
          self.assertTrue(issubclass(p.cls, base_model.MultiTaskModel))
          for _, v in input_p.IterParams():
            self.assertTrue(
                issubclass(v.cls, base_input_generator.BaseInputGenerator),
                'Error in %s' % dataset)

  @classmethod
  def CreateTestMethodsForAllRegisteredModels(cls,
                                              registry,
                                              task_regexes=None,
                                              exclude_regexes=None):
    """Programmatically defines test methods for each registered model."""
    task_regexes = task_regexes or []
    exclude_regexes = exclude_regexes or []
    model_names = list(registry.GetAllRegisteredClasses().keys())
    for model_name in sorted(model_names):
      if not any([re.search(regex, model_name) for regex in task_regexes]):
        tf.logging.info('Skipping tests for registered model: %s', model_name)
        continue
      if any([re.search(regex, model_name) for regex in exclude_regexes]):
        tf.logging.info(
            'Explicitly excluding tests for registered model: %s', model_name)
        continue

      def _Test(self, name=model_name):
        self._testOneModelParams(registry, name)  # pylint: disable=protected-access

      setattr(cls, 'testModelParams_%s' % model_name, _Test)
