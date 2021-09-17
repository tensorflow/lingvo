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
from lingvo import datasets
import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import bn_layers
from lingvo.core import cluster_factory
from lingvo.core import hyperparams
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
        var.trainable = trainable
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
      model_params = registry.GetClass(name)()
      try:
        all_datasets = model_params.GetAllDatasetParams()
      except datasets.GetAllDatasetParamsNotImplementedError:
        all_datasets = {}
        for dataset_name in datasets.GetDatasets(model_params):
          try:
            all_datasets[dataset_name] = getattr(model_params, dataset_name)()
          except NotImplementedError:
            pass

      p = model_params.Model()
      p.input = all_datasets['Train']
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

      for dataset, input_p in all_datasets.items():
        if issubclass(p.cls, base_model.SingleTaskModel):
          if (not isinstance(input_p, hyperparams.InstantiableParams) or
              not issubclass(input_p.cls,
                             base_input_generator.BaseInputGenerator)):
            # Assume this function is not a dataset function but some helper.
            continue
          if (dataset != 'Train' and issubclass(
              input_p.cls, base_input_generator.BaseSequenceInputGenerator) and
              input_p.num_samples != 0):
            self.assertEqual(
                input_p.num_batcher_threads, 1,
                f'num_batcher_threads too large in {dataset}. Decoder or eval '
                f'runs over this set might not span exactly one epoch.')
        else:
          self.assertTrue(issubclass(p.cls, base_model.MultiTaskModel))

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
        print(f'Skipping tests for registered model {model_name}')
        continue
      if any([re.search(regex, model_name) for regex in exclude_regexes]):
        print(f'Explicitly excluding tests for registered model {model_name}')
        continue

      def _Test(self, name=model_name):
        self._testOneModelParams(registry, name)  # pylint: disable=protected-access

      setattr(cls, 'testModelParams_%s' % model_name.replace('.', '_'), _Test)
