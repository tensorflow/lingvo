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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lingvo.core import base_input_generator
from lingvo.core import base_model
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
                          init_wrapper=None,
                          collections=None):
    """Return a zero tensor of the right shape instead of creating variable."""
    del reuse
    dtype = params.dtype
    if init_wrapper:
      var = init_wrapper(dtype, tf.constant_initializer(0, dtype=dtype))
    # For total samples counters we have to actually create variables so that
    # we can access the 'value' attribute during construction.
    elif 'total_samples' in name:
      var = tf.get_variable(
          name,
          params.shape,
          dtype,
          tf.constant_initializer(0, dtype=dtype),
          collections=collections,
          trainable=trainable,
          validate_shape=True)
    else:
      key = hash(tuple(params.shape))
      if key in variable_cache:
        var = variable_cache[key]
      else:
        var = tf.zeros(params.shape, dtype)
        variable_cache[key] = var
    return var, var

  py_utils.CreateVariable = _CreateVariableStub


class BaseModelsTest(test_utils.TestCase):
  """Base model test class which does not define any test methods of its own."""

  def setUp(self):
    self._variable_cache = {}
    _StubOutCreateVariable(self._variable_cache)

  def _testOneModelParams(self, registry, name):
    p = registry.GetParams(name, 'Train')
    self.assertTrue(issubclass(p.cls, base_model.BaseModel))
    self.assertTrue(p.model is not None)
    p.cluster.mode = 'sync'
    p.cluster.job = 'decoder'
    p.cluster.decoder.replicas = 1
    with p.cluster.cls(p.cluster), tf.Graph().as_default():
      # Instantiate the params class, to help catch errors in layer constructors
      # due to misconfigurations.
      p = p.cls(p).params

    for dataset in ('Train', 'Dev', 'Test'):
      input_p = registry.GetClass(name).GetDatasetParams(dataset)
      if issubclass(p.cls, base_model.SingleTaskModel):
        self.assertTrue(
            issubclass(input_p.cls, base_input_generator.BaseInputGenerator),
            'Error in %s' % dataset)
        if (dataset != 'Train') and issubclass(
            input_p.cls, base_input_generator.BaseSequenceInputGenerator) and (
                input_p.num_samples != 0):
          self.assertEquals(
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
                                              task_prefix_filter='',
                                              exclude_prefix=''):
    """Programmatically defines test methods for each registered model."""
    model_names = registry.GetAllRegisteredClasses().keys()
    for model_name in sorted(model_names):
      if task_prefix_filter and not model_name.startswith(task_prefix_filter):
        tf.logging.info('Skipping tests for registered model: %s', model_name)
        continue
      if exclude_prefix and model_name.startswith(exclude_prefix):
        tf.logging.info('Explicitly excluding tests for registered model: %s',
                        model_name)
        continue

      def _Test(self, name=model_name):
        self._testOneModelParams(registry, name)  # pylint: disable=protected-access

      setattr(cls, 'testModelParams_%s' % model_name, _Test)
