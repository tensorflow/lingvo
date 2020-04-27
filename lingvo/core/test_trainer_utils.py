# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Helpers for trainer-based tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
from lingvo import trainer as trainer_lib
import lingvo.compat as tf
from lingvo.core import test_utils

FLAGS = tf.flags.FLAGS


def _ModelTuples(model_classes):
  return [(model_class.__name__, model_class) for model_class in model_classes]


def MakeModelValidatorTestCase(model_classes):
  """Returns a TestCase that validates the training for `model_classes`.

  Usage in a test:

  MakeModelValidatorTestCase([path.to.module.MyRegisteredModel1,
                              path.to.module.MyRegisteredModel2])

  Args:
    model_classes: A list of model classes.
  """

  class _ModelValidator(parameterized.TestCase, test_utils.TestCase):
    """TestCase template for validating training and decoding models."""

    def TrainerBuilds(self, model):
      tmpdir = os.path.join(FLAGS.test_tmpdir, model.__name__)

      # Trainer should probably create these directories in the future.
      tf.io.gfile.makedirs(os.path.join(tmpdir, 'train'))

      model_params = model()
      cfg = model_params.Model()
      cfg.input = model_params.GetDatasetParams('Train')
      cfg.cluster.mode = 'sync'
      cfg.cluster.job = 'trainer_client'
      _ = trainer_lib.Trainer(cfg, '', tmpdir, tf_master='')

    def DecoderBuilds(self, model):
      tmpdir = os.path.join(FLAGS.test_tmpdir, model.__name__)
      tf.io.gfile.makedirs(tmpdir)

      model_params = model()
      cfg = model_params.Model()
      cfg.input = model_params.GetDatasetParams('Train')
      cfg.cluster.mode = 'sync'
      cfg.cluster.job = 'decoder'
      cfg.cluster.task = 0
      cfg.cluster.decoder.replicas = 1
      _ = trainer_lib.Decoder('train', cfg, '', tmpdir, tf_master='')

    # Each of these take about 10-20 seconds to run, to build the model and
    # execute the forward and backward pass building.
    @parameterized.named_parameters(_ModelTuples(model_classes))
    def testTrain(self, model):
      self.TrainerBuilds(model)

    @parameterized.named_parameters(_ModelTuples(model_classes))
    def testDecoder(self, model):
      self.DecoderBuilds(model)

  return _ModelValidator
