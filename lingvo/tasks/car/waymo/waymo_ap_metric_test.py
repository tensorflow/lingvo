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
"""Tests for waymo_ap_metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car.waymo import waymo_ap_metric
from lingvo.tasks.car.waymo import waymo_metadata
import numpy as np
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

FLAGS = tf.flags.FLAGS


class WaymoAveragePrecisionMetrics3DTest(test_utils.TestCase):

  def testWaymoAPConfig(self):
    metadata = waymo_metadata.WaymoMetadata()
    # Use 2D metric.
    config_str = waymo_ap_metric._BuildWaymoMetricConfig(metadata, '2d')
    config = metrics_pb2.Config()
    config.ParseFromString(config_str)
    vehicle_idx = label_pb2.Label.Type.Value('TYPE_VEHICLE')
    ped_idx = label_pb2.Label.Type.Value('TYPE_PEDESTRIAN')
    cyc_idx = label_pb2.Label.Type.Value('TYPE_CYCLIST')

    thresholds_meta = metadata.IoUThresholds()
    self.assertNear(config.iou_thresholds[vehicle_idx],
                    thresholds_meta['Vehicle'], 1e-6)
    self.assertNear(config.iou_thresholds[ped_idx],
                    thresholds_meta['Pedestrian'], 1e-6)
    self.assertNear(config.iou_thresholds[cyc_idx], thresholds_meta['Cyclist'],
                    1e-6)

  def testPerfectBox(self):
    metadata = waymo_metadata.WaymoMetadata()
    params = waymo_ap_metric.WaymoAPMetrics.Params(metadata)
    m = params.Instantiate()
    # Make one update with a perfect box.
    update_dict = py_utils.NestedMap(
        groundtruth_labels=np.array([1]),
        groundtruth_bboxes=np.ones(shape=(1, 7)),
        groundtruth_difficulties=np.zeros(shape=(1)),
        groundtruth_num_points=None,
        detection_scores=np.ones(shape=(5, 1)),
        detection_boxes=np.ones(shape=(5, 1, 7)),
        detection_heights_in_pixels=np.ones(shape=(5, 1)))

    m.Update('1234', update_dict)

    waymo_ap = m._AveragePrecisionByDifficulty()
    self.assertAllClose(waymo_ap['default'][0], 1.)
    self.assertTrue(np.isnan(waymo_ap['default'][1]))


if __name__ == '__main__':
  tf.test.main()
