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
"""Average Precision metric class for Waymo open dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.tasks.car import ap_metric
import numpy as np
from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import breakdown_pb2
from waymo_open_dataset.protos import metrics_pb2


def _BuildWaymoMetricConfig(metadata, box_type):
  """Build the Config proto for Waymo's metric op."""
  config = metrics_pb2.Config()
  # config.num_desired_score_cutoffs = metadata.NumberOfPrecisionRecallPoints()
  num_pr_points = metadata.NumberOfPrecisionRecallPoints()
  config.score_cutoffs.extend(
      [i * 1.0 / (num_pr_points - 1) for i in range(num_pr_points)])
  config.matcher_type = metrics_pb2.MatcherProto.Type.TYPE_HUNGARIAN
  if box_type == '2d':
    config.box_type = label_pb2.Label.Box.Type.TYPE_2D
  else:
    config.box_type = label_pb2.Label.Box.Type.TYPE_3D
  # Default values
  config.iou_thresholds[:] = [0.7, 0.7, 0.7, 0.7, 0.7]
  for class_name, threshold in metadata.IoUThresholds().items():
    cls_idx = metadata.ClassNames().index(class_name)
    config.iou_thresholds[cls_idx] = threshold
  config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.ONE_SHARD)
  config.difficulties.append(metrics_pb2.Difficulty())
  return config.SerializeToString()


class WaymoAPMetrics(ap_metric.APMetrics):
  """The Waymo Open Dataset implementation of AP metric."""

  def __init__(self, params):
    super(WaymoAPMetrics, self).__init__(params)
    self._waymo_metric_config = _BuildWaymoMetricConfig(self.metadata,
                                                        self.params.box_type)

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

    gt_boxes = g.boxes
    gt_imgids = g.imgids
    iou_threshold = self._iou_thresholds[self.metadata.ClassNames()[classid]]

    return py_utils.NestedMap(
        iou_threshold=iou_threshold,
        gt=py_utils.NestedMap(imgid=gt_imgids, bbox=gt_boxes),
        pd=py_utils.NestedMap(imgid=p.imgids, bbox=p.boxes, score=p.scores))

  def _BuildMetric(self, feed_data, classid):
    """Construct tensors and the feed_dict for Waymo metric op.

    Args:
      feed_data: a NestedMap returned by _GetData().
      classid: integer.

    Returns:
      A tuple of 3 dicts:

      - scalar_metrics: a dict mapping all the metric names to fetch tensors.
      - curves: a dict mapping all the curve names to fetch tensors.
      - feed_dict: a dict mapping the tensors in feed_tensors to feed values.
    """

    if feed_data is None:
      dummy_scalar = tf.constant(np.nan)
      dummy_curve = tf.zeros([self.metadata.NumberOfPrecisionRecallPoints(), 2],
                             tf.float32)
      scalar_metrics = {'ap': dummy_scalar, 'ap_ha_weighted': dummy_scalar}
      curve_metrics = {'pr': dummy_curve, 'pr_ha_weighted': dummy_curve}
      return scalar_metrics, curve_metrics, {}

    feed_dict = {}

    f_gt_bbox = tf.placeholder(tf.float32)
    feed_dict[f_gt_bbox] = feed_data.gt.bbox

    f_gt_imgid = tf.placeholder(tf.int32)
    feed_dict[f_gt_imgid] = feed_data.gt.imgid

    f_pd_bbox = tf.placeholder(tf.float32)
    feed_dict[f_pd_bbox] = feed_data.pd.bbox

    f_pd_imgid = tf.placeholder(tf.int32)
    feed_dict[f_pd_imgid] = feed_data.pd.imgid

    f_pd_score = tf.placeholder(tf.float32)
    feed_dict[f_pd_score] = feed_data.pd.score

    num_gt_bboxes = feed_data.gt.imgid.shape[0]
    num_pd_bboxes = feed_data.pd.imgid.shape[0]
    gt_class_ids = tf.constant(classid, dtype=tf.uint8, shape=[num_gt_bboxes])
    pd_class_ids = tf.constant(classid, dtype=tf.uint8, shape=[num_pd_bboxes])
    ap, ap_ha, pr, pr_ha, _ = py_metrics_ops.detection_metrics(
        prediction_bbox=f_pd_bbox,
        prediction_type=pd_class_ids,
        prediction_score=f_pd_score,
        prediction_frame_id=tf.cast(f_pd_imgid, tf.int64),
        prediction_overlap_nlz=tf.zeros_like(f_pd_imgid, dtype=tf.bool),
        ground_truth_bbox=f_gt_bbox,
        ground_truth_type=gt_class_ids,
        ground_truth_frame_id=tf.cast(f_gt_imgid, tf.int64),
        ground_truth_difficulty=tf.zeros_like(f_gt_imgid, dtype=tf.uint8),
        config=self._waymo_metric_config)
    # All tensors returned by Waymo's metric op have a leading dimension
    # B=number of breakdowns. At this moment we always use B=1 to make
    # it compatible to the python code.
    scalar_metrics = {'ap': ap[0], 'ap_ha_weighted': ap_ha[0]}
    curve_metrics = {'pr': pr[0], 'pr_ha_weighted': pr_ha[0]}
    return scalar_metrics, curve_metrics, feed_dict
