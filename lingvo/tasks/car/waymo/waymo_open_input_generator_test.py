# Lint as: python3
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
"""Tests for WaymoOpenInputGenerator."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car.waymo import waymo_open_input_generator
import numpy as np

FLAGS = tf.flags.FLAGS


class WaymoOpenInputGeneratorTest(test_utils.TestCase):

  def testFilterNLZ(self):
    num_points = 32
    num_features = 3
    points_feature = np.random.rand(num_points, num_features).astype(np.float32)
    # Mark one point as being in nlz.
    points_feature[-1, 2] = 1.
    lasers_np = py_utils.NestedMap(
        points_xyz=np.random.rand(num_points, 3).astype(np.float32),
        points_feature=points_feature)
    features_np = py_utils.NestedMap(lasers=lasers_np)
    features = features_np.Transform(tf.constant)

    preprocessor_p = waymo_open_input_generator.FilterNLZPoints.Params()
    processor = preprocessor_p.Instantiate()

    features = processor.TransformFeatures(features)
    with self.session():
      actual_features = self.evaluate(features)
      # one point dropped because it was in nlz
      self.assertEqual((num_points - 1, 3),
                       actual_features.lasers.points_xyz.shape)


if __name__ == '__main__':
  tf.test.main()
