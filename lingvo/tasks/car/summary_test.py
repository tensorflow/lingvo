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
"""Tests for summary."""

from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.car import summary
import numpy as np


class SummaryTest(test_utils.TestCase):
  """Tests for Summary utilities."""

  def testDrawBBoxes(self):
    bs = 5
    nbboxes = 2
    class_id_to_name = {0: 'foo', 1: 'bar'}

    loc_weights = np.ones(shape=(bs, nbboxes))
    labels = np.ones(shape=(bs, nbboxes))
    cbboxes = np.zeros(shape=(bs, nbboxes, 4))
    # Ensures ymin/xmin/ymax/ymin.
    cbboxes[:, :, :2] = 100. * np.random.normal(size=(bs, nbboxes, 2))
    cbboxes[:, :, 2:4] = cbboxes[:, :, :2] + 10. * np.random.uniform(
        size=(bs, nbboxes, 2))

    images = np.zeros(shape=(bs, 100, 100, 1), dtype=np.uint8)
    summary.DrawBBoxesOnImages(images, cbboxes, loc_weights, labels,
                               class_id_to_name, True)

  def testTransformBBoxesToTopDown(self):
    bs = 5
    nbboxes = 2
    cbboxes = np.zeros(shape=(bs, nbboxes, 4))
    # Ensures ymin/xmin/ymax/ymin.
    cbboxes[:, :, :2] = 100. * np.random.normal(size=(bs, nbboxes, 2))
    cbboxes[:, :, 2:4] = cbboxes[:, :, :2] + 10. * np.random.uniform(
        size=(bs, nbboxes, 2))

    image_bboxes = summary.TransformBBoxesToTopDown(cbboxes)
    self.assertAllEqual(cbboxes.shape, image_bboxes.shape)

  def testExtractRunIds(self):

    def PythonExtractRunIds(run_segments):
      """Extract the RunIds from the run_segments feature field."""
      num_segments = run_segments.shape[0]
      run_ids = []
      for i in range(num_segments):
        # Run segment is a serialized RunSegmentProto. We don't have access to
        # it, but it's pretty simple so just manually parse the string for now
        # to get the run id.
        #
        # One can add a duplicate proto to make parsing more explicit.
        run_segment = run_segments[i][0]
        run_id = run_segment.split('\n')[0].split('"')[1]
        start_time = str(int(float(run_segment.split('\n')[1].split(' ')[1])))
        run_ids.append(run_id + '_' + start_time)
      run_ids = [tf.compat.as_bytes(r) for r in run_ids]
      return np.stack(run_ids)

    example_run_segments = [
        [
            'run: "20170903_161642_C00844"\nstart_offset:'
            ' 6876.0795650482178\nend_offset: 6896.0795650482178\nstart_ts:'
            ' 1504462278.6018586\nend_ts: 1504462298.6018586\n'
        ],
        [
            'run: "20001029_0123458_C00844"\nstart_offset:'
            ' 66666.0795650482178\nend_offset: 77777.0795650482178\nstart_ts:'
            ' 1504462278.6018586\nend_ts: 1504462298.6018586\n'
        ],
    ]
    expected_ids = PythonExtractRunIds(np.array(example_run_segments))
    with self.session():
      run_ids = self.evaluate(
          summary.ExtractRunIds(tf.constant(example_run_segments)))
      self.assertAllEqual(expected_ids, run_ids)

  def testTrajectory(self):
    # Test that the trajectory code can execute.  A test that validates the
    # output is challenging
    bs = 2
    steps = 5
    nbboxes = 8
    gt_bboxes = np.zeros(shape=(bs, steps, nbboxes, 5))
    gt_masks = np.ones(shape=(bs, steps, nbboxes))

    pred_bboxes = np.zeros(shape=(bs, steps, nbboxes, 5))
    pred_masks = np.ones(shape=(bs, steps, nbboxes))
    labels = np.ones(shape=(bs, steps, nbboxes)).astype(np.int32)

    image = summary.GetTrajectoryComparison(gt_bboxes, gt_masks, labels,
                                            pred_bboxes, pred_masks, labels)
    self.assertEqual((bs, 1000, 500, 3), image.shape)


if __name__ == '__main__':
  tf.test.main()
