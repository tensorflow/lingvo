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
"""Average Precision metric class for KITTI."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.tasks.car import ap_metric
from lingvo.tasks.car import ops
import numpy as np


class KITTIAPMetrics(ap_metric.APMetrics):
  """The KITTI implementation of AP metric."""

  def _GetData(self,
               classid,
               difficulty=None,
               distance=None,
               num_points=None,
               rotation=None):
    """Returns groundtruth and prediction for the classid in a NestedMap.

    Args:
      classid: int32 specifying the class
      difficulty: String in [easy, moderate, hard]. If None specified, all
        difficulty levels are permitted.
      distance: int32 specifying a binned Euclidean distance of the ground truth
        bounding box. If None is specified, all distances are selected.
      num_points: int32 specifying a binned number of laser points within the
        ground truth bounding box. If None is specified, all boxes are selected.
      rotation: int32 specifying a binned rotation within the ground truth
        bounding box. If None is specified, all boxes are selected.

    Returns:
      NestedMap containing iou_threshold, groundtruth and predictions for
      specified, classid, difficulty level and binned distance. If no bboxes
      are found with these parameters, returns None.
    """
    assert classid > 0 and classid < self.metadata.NumClasses()
    g = self._LoadBoundingBoxes(
        'groundtruth',
        classid,
        distance=distance,
        num_points=num_points,
        rotation=rotation)
    # Note that we do not specify num_points for predictions because only
    # groundtruth boxes contain points.
    p = self._LoadBoundingBoxes(
        'prediction', classid, distance, num_points=None, rotation=rotation)
    if g is None or p is None:
      return None

    # KITTI ignores detections that are too small.
    min_height = 0
    if difficulty:
      min_height = self.metadata.MinHeight2D()[difficulty]
    pd_ignore = p.heights_in_pixels < min_height
    # We first fetch the ground truth boxes for the specified class id
    # into gt_boxes.
    gt_boxes = g.boxes
    gt_imgids = g.imgids
    # Ignore bboxes that are more difficult than this levels by setting their
    # gt_ignore to 1 (IgnoreOneMatch).
    if difficulty:
      gt_ignore = g.difficulties < self.metadata.DifficultyLevels()[difficulty]
      gt_ignore = gt_ignore.astype(np.int32)
    else:
      gt_ignore = np.zeros_like(g.difficulties, dtype=np.int32)

    # Extract the bounding boxes from the groundtruth for 'similar'
    # categories.  Add these to the list of ground truth bounding boxes
    # with an ignore setting of 1 (IgnoreOneMatch).
    if classid in self.metadata.IgnoreClassIndices():
      for class_id_to_ignore in self.metadata.IgnoreClassIndices()[classid]:
        g_ignore = self._LoadBoundingBoxes(
            'groundtruth', class_id=class_id_to_ignore)

        if g_ignore is not None:
          n_ignore_boxes = g_ignore.boxes.shape[0]
          gt_imgids = np.concatenate([gt_imgids, g_ignore.imgids])
          gt_boxes = np.concatenate([gt_boxes, g_ignore.boxes])
          gt_ignore = np.concatenate([gt_ignore, np.ones(n_ignore_boxes)])

    # Extract the DontCare bounding boxes from the data, and add
    # these to the list of bounding boxes to evaluate with an ignore
    # setting of 2 (IgnoreAllMatches).  Only relevant to KITTI.
    if 'DontCare' in self.metadata.ClassNames():
      g_ignore = self._LoadBoundingBoxes(
          'groundtruth', class_id=self.metadata.ClassNames().index('DontCare'))
      if g_ignore is not None:
        n_ignore_boxes = g_ignore.boxes.shape[0]
        gt_imgids = np.concatenate([gt_imgids, g_ignore.imgids])
        gt_boxes = np.concatenate([gt_boxes, g_ignore.boxes])
        gt_ignore = np.concatenate([gt_ignore, 2 * np.ones(n_ignore_boxes)])

    iou_threshold = self._iou_thresholds[self.metadata.ClassNames()[classid]]
    return py_utils.NestedMap(
        iou_threshold=iou_threshold,
        gt=py_utils.NestedMap(imgid=gt_imgids, bbox=gt_boxes, ignore=gt_ignore),
        pd=py_utils.NestedMap(
            imgid=p.imgids, bbox=p.boxes, score=p.scores, ignore=pd_ignore))

  def _BuildMetric(self, feed_data, classid):
    """Construct tensors and the feed_dict for KITTI metric op.

    Args:
      feed_data: a NestedMap returned by _GetData()
      classid: integer. Unused in this implementation.

    Returns:
      A tuple of 3 dicts:

      - scalar_metrics: a dict mapping all the metric names to fetch tensors.
      - curves: a dict mapping all the curve names to fetch tensors.
      - feed_dict: a dict mapping the tensors in feed_tensors to feed values.
    """
    if feed_data is None:
      dummy_scalar = tf.constant(np.nan)
      dummy_calibration = tf.constant(np.nan)
      dummy_curve = tf.zeros([self.metadata.NumberOfPrecisionRecallPoints(), 2],
                             tf.float32)
      scalar_metrics = {'ap': dummy_scalar}
      curve_metrics = {'pr': dummy_curve}
      calibration_metrics = {'calibrations': dummy_calibration}

      return py_utils.NestedMap(
          feed_dict={},
          scalar_metrics=scalar_metrics,
          curve_metrics=curve_metrics,
          calibration_metrics=calibration_metrics)

    feed_dict = {}

    f_iou = tf.placeholder(tf.float32)
    feed_dict[f_iou] = feed_data.iou_threshold

    f_gt_bbox = tf.placeholder(tf.float32)
    feed_dict[f_gt_bbox] = feed_data.gt.bbox

    f_gt_imgid = tf.placeholder(tf.int32)
    feed_dict[f_gt_imgid] = feed_data.gt.imgid

    f_gt_ignore = tf.placeholder(tf.int32)
    feed_dict[f_gt_ignore] = feed_data.gt.ignore

    f_pd_bbox = tf.placeholder(tf.float32)
    feed_dict[f_pd_bbox] = feed_data.pd.bbox

    f_pd_imgid = tf.placeholder(tf.int32)
    feed_dict[f_pd_imgid] = feed_data.pd.imgid

    f_pd_ignore = tf.placeholder(tf.int32)
    feed_dict[f_pd_ignore] = feed_data.pd.ignore

    f_pd_score = tf.placeholder(tf.float32)
    feed_dict[f_pd_score] = feed_data.pd.score

    # TODO(shlens): The third returned argument contain statistics for measuring
    # the calibration error. Use it.
    ap, pr, calibration = ops.average_precision3d(
        iou_threshold=f_iou,
        groundtruth_bbox=f_gt_bbox,
        groundtruth_imageid=f_gt_imgid,
        groundtruth_ignore=f_gt_ignore,
        prediction_bbox=f_pd_bbox,
        prediction_imageid=f_pd_imgid,
        prediction_ignore=f_pd_ignore,
        prediction_score=f_pd_score,
        num_recall_points=self.metadata.NumberOfPrecisionRecallPoints())

    scalar_metrics = {'ap': ap}
    curve_metrics = {'pr': pr}
    calibration_metrics = {'calibrations': calibration}
    return py_utils.NestedMap(
        feed_dict=feed_dict,
        scalar_metrics=scalar_metrics,
        curve_metrics=curve_metrics,
        calibration_metrics=calibration_metrics)

  def _ComputeFinalMetrics(self,
                           classids=None,
                           difficulty=None,
                           distance=None,
                           num_points=None,
                           rotation=None):
    """Compute precision-recall curves as well as average precision.

    Args:
      classids: A list of N int32.
      difficulty: String in [easy, moderate, hard]. If None specified, all
        difficulty levels are permitted.
      distance: int32 specifying a binned Euclidean distance of the ground truth
        bounding box. If None is specified, all distances are selected.
      num_points: int32 specifying a binned number of laser points within the
        ground truth bounding box. If None is specified, all boxes are selected.
      rotation: int32 specifying a binned rotation within the ground truth
        bounding box. If None is specified, all boxes are selected.

    Returns:
      dict. Each entry in the dict is a list of C (number of classes) dicts
      containing mapping from metric names to individual results. Individual
      entries may be the following items.
      - scalars: A list of C (number of classes) dicts mapping metric
      names to scalar values.
      - curves: A list of C dicts mapping metrics names to np.float32
      arrays of shape [NumberOfPrecisionRecallPoints()+1, 2]. In the last
      dimension, 0 indexes precision and 1 indexes recall.
      - calibrations: A list of C dicts mapping metrics names to np.float32
      arrays of shape [number of predictions, 2]. The first column is the
      predicted probability and the second column is 0 or 1 indicating that the
      prediction matched a ground truth item.
    """
    tf.logging.info('Computing final KITTI metrics.')
    assert classids is not None, 'classids must be supplied.'
    feed_dict = {}
    g = tf.Graph()
    scalar_fetches = []
    curve_fetches = []
    calibration_fetches = []
    with g.as_default():
      for classid in classids:
        data = self._GetData(
            classid,
            difficulty=difficulty,
            distance=distance,
            num_points=num_points,
            rotation=rotation)
        metrics = self._BuildMetric(data, classid)
        scalar_fetches += [metrics.scalar_metrics]
        curve_fetches += [metrics.curve_metrics]
        calibration_fetches += [metrics.calibration_metrics]
        feed_dict.update(metrics.feed_dict)

    with tf.Session(graph=g) as sess:
      results = sess.run([scalar_fetches, curve_fetches, calibration_fetches],
                         feed_dict=feed_dict)
    tf.logging.info('Finished computing final KITTI metrics.')
    return {
        'scalars': results[0],
        'curves': results[1],
        'calibrations': results[2]
    }
