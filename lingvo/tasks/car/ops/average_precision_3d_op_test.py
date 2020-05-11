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
"""Tests for lingvo.tasks.car.ops.car_metrics_ops."""

from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.car import ops
import numpy as np


class ImageMetricsOpsTest(test_utils.TestCase):

  def _GenerateRandomBBoxes(self, num_images, num_bboxes):
    xyz = np.random.uniform(low=-1.0, high=1.0, size=(num_bboxes, 3))
    dimension = np.random.uniform(low=0.1, high=1.0, size=(num_bboxes, 3))
    rotation = np.random.uniform(low=-np.pi, high=np.pi, size=(num_bboxes, 1))
    bboxes = np.concatenate([xyz, dimension, rotation], axis=-1)
    imageid = np.random.randint(0, num_images, size=[num_bboxes])
    scores = np.random.uniform(size=[num_bboxes])
    return bboxes, imageid, scores

  def _GetAP(self, gt_bbox, gt_imgid, pd_bbox, pd_imgid, pd_score, algorithm):
    g = tf.Graph()
    with g.as_default():
      iou, pr, score_and_hit = ops.average_precision3d(
          iou_threshold=0.5,
          groundtruth_bbox=gt_bbox,
          groundtruth_imageid=gt_imgid,
          groundtruth_ignore=tf.zeros_like(gt_imgid, dtype=tf.int32),
          prediction_bbox=pd_bbox,
          prediction_imageid=pd_imgid,
          prediction_score=pd_score,
          prediction_ignore=tf.zeros_like(pd_imgid, dtype=tf.int32),
          num_recall_points=41,
          algorithm=algorithm)
    with self.session(graph=g):
      val = self.evaluate([iou, pr, score_and_hit])
    return val

  def testAPKITTI(self):
    k, n, m = 10, 100, 20
    gt_bbox, gt_imgid, _ = self._GenerateRandomBBoxes(k, n)
    pd_bbox, pd_imgid, pd_score = self._GenerateRandomBBoxes(k, m)
    # IoU between two set of random boxes;
    iou, _, score_and_hit = self._GetAP(
        gt_bbox, gt_imgid, pd_bbox, pd_imgid, pd_score, algorithm='KITTI')
    self.assertAllEqual(score_and_hit.shape, (m, 2))
    self.assertTrue(0 <= iou and iou <= 1.0)

    # Make the predictions be a duplicate of the ground truth to emulate
    # perfect detection.
    iou, _, score_and_hit = self._GetAP(
        gt_bbox, gt_imgid, gt_bbox, gt_imgid, np.ones(n), algorithm='KITTI')
    self.assertAllEqual(score_and_hit.shape, (n, 2))
    self.assertAllEqual(score_and_hit[:, 1], np.ones(n))
    self.assertEqual(1, iou)

    # Ditto as above but make the detection scores unique so that one can test
    # that the scores are correctly returned.
    iou, _, score_and_hit = self._GetAP(
        gt_bbox,
        gt_imgid,
        gt_bbox,
        gt_imgid,
        np.linspace(0, 1, n),
        algorithm='KITTI')
    self.assertAllEqual(score_and_hit.shape, (n, 2))
    self.assertAllClose(score_and_hit[:, 0], np.linspace(0, 1, n))
    self.assertAllEqual(score_and_hit[:, 1], np.ones(n))
    self.assertEqual(1, iou)

    # IoU of empty detection
    iou, _, score_and_hit = self._GetAP(
        gt_bbox, gt_imgid, pd_bbox, pd_imgid + n, pd_score, algorithm='KITTI')
    self.assertAllEqual(score_and_hit.shape, (m, 2))
    self.assertAllEqual(score_and_hit[:, 1], np.zeros(m))
    self.assertEqual(0, iou)

  def testAPVOC(self):
    k, n, m = 10, 100, 20
    gt_bbox, gt_imgid, _ = self._GenerateRandomBBoxes(k, n)
    pd_bbox, pd_imgid, pd_score = self._GenerateRandomBBoxes(k, m)
    # IoU between two set of random boxes;
    iou, _, _ = self._GetAP(
        gt_bbox, gt_imgid, pd_bbox, pd_imgid, pd_score, algorithm='VOC')
    self.assertTrue(0 <= iou and iou <= 1.0)
    # IoU of perfect detection
    iou, _, score_and_hit = self._GetAP(
        gt_bbox, gt_imgid, gt_bbox, gt_imgid, np.ones(n), algorithm='VOC')
    # Just check that dummy values are returned.
    self.assertAllEqual(score_and_hit.shape, (n, 2))
    self.assertAllEqual(score_and_hit, -1.0 * np.ones(shape=(n, 2)))

    self.assertEqual(1, iou)
    # IoU of empty detection
    iou, _, _ = self._GetAP(
        gt_bbox, gt_imgid, pd_bbox, pd_imgid + n, pd_score, algorithm='VOC')
    self.assertEqual(0, iou)

  def testAllZeroValue(self):
    k, n, m = 10, 100, 20
    gt_bbox, gt_imgid, _ = self._GenerateRandomBBoxes(k, n)
    pd_bbox, pd_imgid, pd_score = self._GenerateRandomBBoxes(k, m)
    # IoU between two set of random boxes;
    iou, pr, _ = self._GetAP(
        gt_bbox * 0,
        gt_imgid * 0,
        pd_bbox * 0,
        pd_imgid * 0,
        pd_score * 0,
        algorithm='KITTI')
    self.assertEqual(0, iou)
    self.assertAllEqual(pr.shape, (41, 2))
    self.assertAllEqual(np.zeros(41), pr[:, 0])


if __name__ == '__main__':
  tf.test.main()
