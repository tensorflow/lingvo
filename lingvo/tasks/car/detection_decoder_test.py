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
"""Tests for detection_decoder."""

from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.car import detection_decoder


class DetectionDecoderTest(test_utils.TestCase):
  """Tests for detection_decoder library."""

  def testDecoderWithOrientedPerClassNMS(self):
    batch_size = 4
    num_preds = 8
    num_classes = 10

    # An example of setting the score threshold high and IOU threshold low
    # for classes we don't care about
    score_threshold = [1.0] * num_classes
    score_threshold[1] = 0.05

    nms_iou_threshold = [0.0] * num_classes
    nms_iou_threshold[1] = 0.5

    with tf.Graph().as_default():
      tf.random.set_seed(12345)
      predicted_bboxes = tf.random.normal([batch_size, num_preds, 7])
      classification_scores = tf.random.uniform(
          [batch_size, num_preds, num_classes], minval=0, maxval=1)

      idxs, bboxes, bbox_scores, valid_mask = detection_decoder.DecodeWithNMS(
          predicted_bboxes,
          classification_scores,
          nms_iou_threshold=nms_iou_threshold,
          score_threshold=score_threshold,
          use_oriented_per_class_nms=True)

      with self.session():
        outputs = self.evaluate([
            predicted_bboxes, classification_scores, idxs, bboxes, bbox_scores,
            valid_mask
        ])
        (input_bboxes, input_scores, output_idxs, output_bboxes, output_scores,
         mask) = outputs

        self.assertEqual((batch_size, num_preds, 7), input_bboxes.shape)
        self.assertEqual((batch_size, num_classes, num_preds),
                         output_idxs.shape)
        self.assertEqual((batch_size, num_classes, num_preds, 7),
                         output_bboxes.shape)
        self.assertEqual((batch_size, num_preds, num_classes),
                         input_scores.shape)
        self.assertEqual((batch_size, num_classes, num_preds),
                         output_scores.shape)
        self.assertEqual((batch_size, num_classes, num_preds), mask.shape)

        # Assert that NMS did some kind of filtering for each class
        for cls_idx in range(num_classes):
          self.assertEqual(
              mask[:, cls_idx, :].sum(),
              (input_scores[:, :, cls_idx] > score_threshold[cls_idx]).sum())
          self.assertEqual(
              mask[:, cls_idx, :].sum(),
              (output_scores[:, cls_idx, :] > score_threshold[cls_idx]).sum())

  def testDecoderSingleClassNMS(self):
    batch_size = 4
    num_preds = 8
    num_classes = 10

    score_threshold = 0.05
    nms_iou_threshold = 0.5
    with tf.Graph().as_default():
      tf.random.set_seed(12345)
      predicted_bboxes = tf.random.normal([batch_size, num_preds, 7])
      classification_scores = tf.random.uniform(
          [batch_size, num_preds, num_classes], minval=0, maxval=1)

      idxs, bboxes, bbox_scores, valid_mask = detection_decoder.DecodeWithNMS(
          predicted_bboxes,
          classification_scores,
          nms_iou_threshold=nms_iou_threshold,
          score_threshold=score_threshold,
          use_oriented_per_class_nms=False)

      with self.session():
        outputs = self.evaluate([
            predicted_bboxes, classification_scores, idxs, bboxes, bbox_scores,
            valid_mask
        ])
        (input_bboxes, input_scores, output_idxs, output_bboxes, output_scores,
         mask) = outputs

        self.assertEqual((batch_size, num_preds, 7), input_bboxes.shape)
        self.assertEqual((batch_size, num_preds), output_idxs.shape)
        self.assertEqual((batch_size, num_classes, num_preds, 7),
                         output_bboxes.shape)
        self.assertEqual((batch_size, num_preds, num_classes),
                         input_scores.shape)
        self.assertEqual((batch_size, num_classes, num_preds),
                         output_scores.shape)
        self.assertEqual((batch_size, num_classes, num_preds), mask.shape)


if __name__ == '__main__':
  tf.test.main()
