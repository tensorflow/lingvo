# Lint as: python2, python3
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
"""Train models on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.tasks.image import classifier
from lingvo.tasks.image import input_generator


class Base(base_model_params.SingleTaskModelParams):
  """Input params for MNIST."""

  @property
  def path(self):
    # Generated using lingvo/tools:keras2ckpt.
    return '/tmp/mnist/mnist'

  def Train(self):
    p = input_generator.MnistTrainInput.Params()
    p.ckpt = self.path
    return p

  def Test(self):
    p = input_generator.MnistTestInput.Params()
    p.ckpt = self.path
    return p

  def Dev(self):
    return self.Test()


@model_registry.RegisterSingleTaskModel
class LeNet5(Base):
  """LeNet params for MNIST classification."""

  BN = False
  DROP = 0.2

  def Task(self):
    p = classifier.ModelV1.Params()
    p.name = 'lenet5'
    # Overall architecture:
    #   conv, maxpool, conv, maxpool, fc, softmax.
    p.filter_shapes = [(5, 5, 1, 20), (5, 5, 20, 50)]
    p.window_shapes = [(2, 2), (2, 2)]
    p.batch_norm = self.BN
    p.dropout_prob = self.DROP
    p.softmax.input_dim = 300
    p.softmax.num_classes = 10
    p.train.save_interval_seconds = 10  # More frequent checkpoints.
    p.eval.samples_per_summary = 0  # Eval the whole set.
    return p
