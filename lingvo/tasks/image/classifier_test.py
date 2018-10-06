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
"""Tests for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil

import tensorflow as tf
from lingvo import model_registry
from lingvo.core import cluster_factory
from lingvo.core import py_utils
from lingvo.core.test_utils import CompareToGoldenSingleFloat
from lingvo.tasks.image import classifier
from lingvo.tasks.image import input_generator
from lingvo.tasks.image.params import mnist


class ClassifierTest(tf.test.TestCase):

  def setUp(self):
    self._tmpdir, self.data_path = input_generator.FakeMnistData(train_size=0)

  def tearDown(self):
    shutil.rmtree(self._tmpdir)

  def _runOneStep(self, model, sess):
    f_loss = sess.run(model.GetTask().loss)
    sess.run(model.GetTask().train_op)
    return f_loss

  def testMnistLeNet5(self):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(1618)
      p = model_registry.GetParams('image.mnist.LeNet5', 'Test')
      p.random_seed = 73234288
      p.input.ckpt = self.data_path
      p.task.params_init = py_utils.WeightInit.Uniform(0.1, seed=73234288)
      with cluster_factory.ForTestingWorker(mode='sync', job='trainer_client'):
        model = p.cls(p)
        model.ConstructFPropBPropGraph()
    with self.session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      CompareToGoldenSingleFloat(self, 2.302583, self._runOneStep(model, sess))
      CompareToGoldenSingleFloat(self, 2.302405, self._runOneStep(model, sess))


if __name__ == '__main__':
  tf.test.main()
