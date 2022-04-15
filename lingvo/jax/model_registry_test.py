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
"""Tests for model_registry."""

from absl.testing import absltest
from lingvo.jax import base_model_params
from lingvo.jax import layers
from lingvo.jax import model_registry
from lingvo.jax.tasks.test.params import synthetic  # pylint: disable=unused-import


@model_registry.register_model
class DummyModel(base_model_params.BaseModelParams):

  def datasets(self):
    return []

  def task(self):
    act_p = layers.Activation.Params()
    return act_p


class ModelRegistryTest(absltest.TestCase):

  def test_get_model(self):
    # Module name is `__main__` when registering locally like here.
    dummy_model_cls = model_registry.get_model('__main__.DummyModel')
    dummy_model = dummy_model_cls()
    self.assertEmpty(dummy_model.datasets())
    self.assertIsNotNone(dummy_model.task())

  def test_get_lingvo_model(self):
    # Module name is `__main__` when registering locally like here.
    dummy_model_cls = model_registry.get_model(
        'test.synthetic.SyntheticClassifier')
    dummy_model = dummy_model_cls()
    self.assertEmpty(dummy_model.datasets())
    self.assertIsNotNone(dummy_model.task())

  def test_unregister_register_model(self):
    model_name = '__main__.DummyModel'
    model_registry.unregister_model(model_name)

    dummy_model_cls = model_registry.get_model(model_name)
    self.assertIsNone(dummy_model_cls)

    # Call unregister a second time: No op.
    model_registry.unregister_model(model_name)

    model_registry.register_model(DummyModel)

    dummy_model_cls = model_registry.get_model(model_name)
    dummy_model = dummy_model_cls()
    self.assertIsInstance(dummy_model, DummyModel)


if __name__ == '__main__':
  absltest.main()
