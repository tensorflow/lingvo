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
"""Tests for waymo_ap_metric."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car.waymo import waymo_ap_metric
from lingvo.tasks.car.waymo import waymo_metadata
import numpy as np
from waymo_open_dataset import label_pb2

FLAGS = tf.flags.FLAGS


class APTest(test_utils.TestCase):

  def testWaymoAPConfig(self):
    metadata = waymo_metadata.WaymoMetadata()
    # Use 2D metric.
    config = waymo_ap_metric.BuildWaymoMetricConfig(metadata, '2d', [])
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

    waymo_ap = m.value
    self.assertAllClose(waymo_ap, 1. / 3.)

    # Write a summary.
    summary = m.Summary('foo')
    # Check that both AP and APH are in the tags.
    tags = [v.tag for v in summary.value]
    self.assertIn('foo/Pedestrian/AP_LEVEL_1', tags)
    self.assertIn('foo/Pedestrian/APH_LEVEL_1', tags)
    self.assertIn('foo/Pedestrian/AP_LEVEL_2', tags)
    self.assertIn('foo/Pedestrian/APH_LEVEL_2', tags)

  def testWaymoBreakdowns(self):
    metadata = waymo_metadata.WaymoMetadata()
    params = waymo_ap_metric.WaymoAPMetrics.Params(metadata)
    params.waymo_breakdown_metrics = ['RANGE', 'VELOCITY']

    m = params.Instantiate()
    # Make one update with a perfect box.
    update_dict = py_utils.NestedMap(
        groundtruth_labels=np.array([1]),
        groundtruth_bboxes=np.ones(shape=(1, 7)),
        groundtruth_difficulties=np.zeros(shape=(1)),
        groundtruth_num_points=None,
        groundtruth_speed=np.zeros(shape=(1, 2)),
        detection_scores=np.ones(shape=(5, 1)),
        detection_boxes=np.ones(shape=(5, 1, 7)),
        detection_heights_in_pixels=np.ones(shape=(5, 1)))

    m.Update('1234', update_dict)

    # Write a summary.
    summary = m.Summary('foo')
    # Check that the summary value for default ap and
    # a waymo breakdown version by range is the same.
    for v in summary.value:
      if v.tag == 'foo/Vehicle/AP_LEVEL_1':
        default_val = v.simple_value
      elif v.tag == 'foo/Vehicle/APH_LEVEL_1':
        aph_default_val = v.simple_value
      elif v.tag == 'foo_extra/AP_RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1':
        ap_bd_val_l1 = v.simple_value
      elif v.tag == 'foo_extra/AP_RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2':
        ap_bd_val_l2 = v.simple_value
      elif v.tag == 'foo_extra/APH_RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1':
        aph_bd_val_l1 = v.simple_value
      elif v.tag == 'foo_extra/APH_RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2':
        aph_bd_val_l2 = v.simple_value
      elif v.tag == 'foo_extra/AP_VELOCITY_TYPE_VEHICLE_STATIONARY_LEVEL_1':
        vbd_val_l1 = v.simple_value
      elif v.tag == 'foo_extra/AP_VELOCITY_TYPE_VEHICLE_STATIONARY_LEVEL_2':
        vbd_val_l2 = v.simple_value

    self.assertEqual(ap_bd_val_l1, default_val)
    self.assertEqual(ap_bd_val_l2, default_val)
    self.assertEqual(aph_bd_val_l1, aph_default_val)
    self.assertEqual(aph_bd_val_l2, aph_default_val)
    self.assertEqual(vbd_val_l1, default_val)
    self.assertEqual(vbd_val_l2, default_val)

    # Check that eval classes not evaluated are not present.
    tags = [v.tag for v in summary.value]
    self.assertNotIn('foo_extra/APH_RANGE_TYPE_SIGN_[0, 30)_LEVEL_1', tags)
    self.assertNotIn('foo_extra/APH_RANGE_TYPE_SIGN_[0, 30)_LEVEL_2', tags)


if __name__ == '__main__':
  tf.test.main()
