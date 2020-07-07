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
"""Average precision metric interface."""

import functools
from lingvo import compat as tf
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core.metrics import BaseMetric
from lingvo.tasks.car import breakdown_metric
import numpy as np


# TODO(shlens): Consider making this internal data structure a list of NestedMap
# to simplify the implementation.
class Boxes3D:
  """A container for a list of 7-DOF 3D boxes.

  Each 3D box is represented by the 7-tuple [x, y, z, dx, dy, dz, phi],
  where x, y, z is the center of the box, dx, dy, dz represent the width,
  length, and height of the box, and phi is the rotation of the box.
  """

  def __init__(self):
    self._capacity = 0
    self._size = 0
    self._buf = py_utils.NestedMap(
        imgids=np.empty([0]),
        scores=np.empty([0]),
        boxes=np.empty([0, 7]),
        difficulties=np.empty([0]),
        distances=np.empty([0]),
        num_points=np.empty([0]),
        rotations=np.empty([0]),
        heights_in_pixels=np.empty([0]),
        speeds=np.empty([0, 2]))

  def Add(self, img_id, score, box, difficulty, distance, num_points, rotation,
          height_in_pixels, speed):
    """Adds a bbox.

    Args:
      img_id: A unique image identifier.
      score: The confidence score.
      box: [1 x 7] numpy array.
      difficulty: The difficulty of a box. On KITTI this is one of [0, 1, 2, 3].
      distance: The binned distance of a box.
      num_points: Number of laser points in box.
      rotation: The binned rotation of a box.
      height_in_pixels: The height of the 2D bbox of this object in the camera
        image.
      speed: A [1 x 2] numpy array with speed of object in world frame.
    """
    if self._size >= self._capacity:
      # resize
      if self._capacity:
        # Increase the capacity exponentially.
        self._capacity += self._capacity // 4
      else:
        self._capacity = 100
      self._buf = self._buf.Transform(self._Resize)
    self._buf.imgids[self._size] = img_id
    self._buf.scores[self._size] = score
    self._buf.boxes[self._size] = box
    self._buf.difficulties[self._size] = difficulty
    self._buf.distances[self._size] = distance
    self._buf.num_points[self._size] = num_points
    self._buf.rotations[self._size] = rotation
    self._buf.heights_in_pixels[self._size] = height_in_pixels
    self._buf.speeds[self._size] = speed
    self._size += 1

  def _Resize(self, arr):
    n = self._capacity
    ret = np.empty([n] + list(arr.shape)[1:], dtype=arr.dtype)
    ret[:arr.shape[0]] = arr
    return ret

  @property
  def imgids(self):
    return self._buf.imgids[:self._size]

  @property
  def scores(self):
    return self._buf.scores[:self._size]

  @property
  def boxes(self):
    return self._buf.boxes[:self._size]

  @property
  def difficulties(self):
    return self._buf.difficulties[:self._size]

  @property
  def distances(self):
    return self._buf.distances[:self._size]

  @property
  def num_points(self):
    return self._buf.num_points[:self._size]

  @property
  def rotations(self):
    return self._buf.rotations[:self._size]

  @property
  def heights_in_pixels(self):
    return self._buf.heights_in_pixels[:self._size]

  @property
  def speeds(self):
    return self._buf.speeds[:self._size]


class APMetrics(BaseMetric):
  """Measure an assortment of precision-recall metrics on a dataset."""

  @classmethod
  def Params(cls, metadata):
    """Params builder for APMetrics."""
    p = hyperparams.InstantiableParams(cls)
    p.Define(
        'metadata', metadata,
        'Instance of class obeying EvaluationMetadata interface consisting of '
        'parameters specifying the details of the evaluation.')
    p.Define(
        'breakdown_metrics', [],
        'List of extra breakdown metrics when computing AP.  Valid values '
        'include: ["num_points", "distance", "rotation"].  See '
        'breakdown_metric.py:ByName for the full list. ByDifficulty is '
        'always used.')
    p.Define(
        'metric_weights', None,
        'For metrics that have multiple breakdown metrics, '
        'a user may want the value() function to be a function '
        'of the various breakdowns.  If provided, this specifies '
        'the weights assigned to each class\'s metric contribution. '
        'metric_weights should be a dictionary mapping every difficulty '
        'level to a numpy vector of weights whose values corresponds '
        'to the order of metadata.EvalClassIndices() weighting.')
    p.Define(
        'box_type', '3d', 'Specifies what kind of box evaluation will '
        'be performed (only supported by waymo metric).  '
        'One of ["2d", "3d"]: 3d means to do 3D AP calculation, and 2d '
        'means to do a top down Birds-Eye-View AP calculation.')
    return p

  def __init__(self, params):
    """Initialize the metrics."""
    self.params = params.Copy()
    self._is_eval_complete = False
    self._groundtruth = {}  # keyed by class id.
    self._prediction = {}  # keyed by class id.
    self._str_to_imgid = {}
    self._iou_thresholds = self.params.metadata.IoUThresholds()

    self.metadata = self.params.metadata
    assert self.params.box_type in ['2d', '3d']

    # We must always include ByDifficulty.
    metrics_params = breakdown_metric.ByDifficulty.Params().Set(
        metadata=self.metadata)
    self._breakdown_metrics = {
        'difficulty': breakdown_metric.ByDifficulty(metrics_params)
    }

    for breakdown_metric_name in self.params.breakdown_metrics:
      self._breakdown_metrics[breakdown_metric_name] = (
          breakdown_metric.ByName(breakdown_metric_name)(metrics_params))

  def _GetImageId(self, str_id):
    if str_id in self._str_to_imgid:
      return self._str_to_imgid[str_id]
    else:
      imgid = len(self._str_to_imgid)
      self._str_to_imgid[str_id] = imgid
      return imgid

  def _AddGroundtruth(self, box):
    """Record a ground truth box."""
    (str_imgid, classid, score, box, difficulty, distance, num_points, rot,
     speed) = box
    imgid = self._GetImageId(str_imgid)
    assert classid > 0 and classid < self.metadata.NumClasses(), (
        '{} vs. {}'.format(classid, self.metadata.NumClasses()))

    boxes = self._groundtruth.get(classid)
    if boxes is None:
      boxes = Boxes3D()
      self._groundtruth[classid] = boxes
    boxes.Add(imgid, score, box, difficulty, distance, num_points, rot, -1,
              speed)
    # Invalidate the evaluation.
    self._is_eval_complete = False

  def _LoadBoundingBoxes(self,
                         box_type,
                         class_id,
                         distance=None,
                         num_points=None,
                         rotation=None):
    """Load a specified set of bounding boxes.

    If no boxes are found, return None.

    Note that we do *not* specify 'difficulty' as an option due to how the KITTI
    evaluates bounding boxes across difficulty levels.

    Args:
      box_type: string. Either 'groundtruth' or 'prediction'
      class_id: int32 specifying the class
      distance: int32 specifying a binned Euclidean distance of the ground truth
        bounding box. If None is specified, all distances are selected.
      num_points: int32 specifying a binned number of laser points within the
        ground truth bounding box. If None is specified, all boxes are selected.
      rotation: int32 specifying a binned rotation within the ground truth
        bounding box. If None is specified, all boxes are selected.

    Returns:
      Boxes3D containing bounding boxes or None if no bounding boxes available.
    """
    assert box_type in ['groundtruth', 'prediction']
    if box_type == 'groundtruth':
      boxes_by_class = self._groundtruth
    else:
      boxes_by_class = self._prediction

    if class_id not in boxes_by_class:
      return None
    boxes = boxes_by_class[class_id]

    if boxes is not None and distance is not None:
      # Filter bounding boxes based a binned (integer) distance.
      filtered_boxes = None
      for i, s, b, d, dist, n, r, h, v in zip(boxes.imgids, boxes.scores,
                                              boxes.boxes, boxes.difficulties,
                                              boxes.distances, boxes.num_points,
                                              boxes.rotations,
                                              boxes.heights_in_pixels,
                                              boxes.speeds):
        if dist == distance:
          if filtered_boxes is None:
            filtered_boxes = Boxes3D()
          filtered_boxes.Add(i, s, b, d, dist, n, r, h, v)
      boxes = filtered_boxes

    if boxes is not None and num_points is not None:
      # Filter bounding boxes based a binned (integer) number of points.
      filtered_boxes = None
      for i, s, b, d, dist, n, r, h, v in zip(boxes.imgids, boxes.scores,
                                              boxes.boxes, boxes.difficulties,
                                              boxes.distances, boxes.num_points,
                                              boxes.rotations,
                                              boxes.heights_in_pixels,
                                              boxes.speeds):
        if n == num_points:
          if filtered_boxes is None:
            filtered_boxes = Boxes3D()
          filtered_boxes.Add(i, s, b, d, dist, n, r, h, v)
      boxes = filtered_boxes

    if boxes is not None and rotation is not None:
      # Filter bounding boxes based a binned (integer) rotation.
      filtered_boxes = None
      for i, s, b, d, dist, n, r, h, v in zip(boxes.imgids, boxes.scores,
                                              boxes.boxes, boxes.difficulties,
                                              boxes.distances, boxes.num_points,
                                              boxes.rotations,
                                              boxes.heights_in_pixels,
                                              boxes.speeds):
        if r == rotation:
          if filtered_boxes is None:
            filtered_boxes = Boxes3D()
          filtered_boxes.Add(i, s, b, d, dist, n, r, h, v)
      boxes = filtered_boxes

    return boxes

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
    raise NotImplementedError('_GetData must be implemented')

  def _BuildMetric(self, feed_data, classid):
    raise NotImplementedError('_BuildMetric must be implemented')

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
      predicted probabilty and the second column is 0 or 1 indicating that the
      prediction matched a ground truth item.
    """
    raise NotImplementedError('_ComputeFinalMetric must be implemented')

  def Update(self, str_id, result):
    """Update this metric with a newly evaluated image.

    Args:
      str_id: A string. Unique identifier of an image.
      result: A NestedMap with these fields:
          groundtruth_labels - [N]. Groundtruth boxes' labels.

          groundtruth_bboxes - [N, 7]. Groundtruth boxes coordinates.

          groundtruth_difficulties - [N]. Groundtruth boxes difficulties.

          groundtruth_num_points - [N]. Number of laser points in bounding
          boxes.

          groundtruth_speed - [N, 2] Speed in (vx, vy) of the ground truth
            object.

          detection_scores - [C, M] - For each class (C classes) we have (up to)
          M predicted boxes.

          detection_boxes - [C, M, 7] - [i, j, :] are coordinates for i-th
          classes j-th predicted box.

          detection_heights_in_pixels - [C, M]. [i, j] is the height of i-th
          classes j-th boxes 2D image coordinate (camera view).

    """
    n = result.groundtruth_labels.shape[0]
    assert result.groundtruth_bboxes.shape == (n, 7)

    if 'groundtruth_speed' not in result:
      result.groundtruth_speed = np.zeros((n, 2), dtype=np.float32)

    groundtruth_result = py_utils.NestedMap(
        bboxes=result.groundtruth_bboxes,
        num_points=result.groundtruth_num_points,
        difficulties=result.groundtruth_difficulties,
        labels=result.groundtruth_labels)
    for m in self._breakdown_metrics.values():
      m.AccumulateHistogram(groundtruth_result)
      m.AccumulateCumulative(groundtruth_result)

    # TODO(vrv): The Boxes3D structure currently expects a fixed
    # number of arguments corresponding to every possible
    # breakdown metric, even if not requested or present, requiring
    # dummy values in the latter case.  We should figure
    # out how to avoid requiring these dummy values by making
    # the Boxes3D object take a dynamic set of attributes.
    num_points = [0] * n
    rotations = [0] * n
    distances = [0] * n
    if 'num_points' in self._breakdown_metrics:
      num_points = self._breakdown_metrics['num_points'].Discretize(
          result.groundtruth_num_points)
    if 'rotation' in self._breakdown_metrics:
      rotations = self._breakdown_metrics['rotation'].Discretize(
          result.groundtruth_bboxes)
    if 'distance' in self._breakdown_metrics:
      distances = self._breakdown_metrics['distance'].Discretize(
          result.groundtruth_bboxes)

    for label, bbox, difficulty, distance, num_points, rotations, speed in zip(
        result.groundtruth_labels, result.groundtruth_bboxes,
        result.groundtruth_difficulties, distances, num_points, rotations,
        result.groundtruth_speed):
      self._AddGroundtruth((str_id, label, 1., bbox, difficulty, distance,
                            num_points, rotations, speed))

    c = result.detection_scores.shape[0]
    assert c == self.metadata.NumClasses(), '%s vs. %s' % (
        c, self.metadata.NumClasses())

    str_imgid = self._GetImageId(str_id)

    # Iterate first by class.
    for class_id in range(1, c):
      assert class_id > 0 and class_id < self.metadata.NumClasses(), (
          '{} vs. {}'.format(class_id, self.metadata.NumClasses()))

      # Get or create the box list for the class.  This is done for speed -- we
      # do it once per class to avoid dictionary lookups to fetch the field for
      # every box.
      boxes_for_class = self._prediction.get(class_id)
      if boxes_for_class is None:
        boxes_for_class = Boxes3D()
        self._prediction[class_id] = boxes_for_class

      # Get boxes and scores for this class.
      bboxes = result.detection_boxes[class_id, :]
      scores = result.detection_scores[class_id, :]
      heights_in_pixels = result.detection_heights_in_pixels[class_id, :]
      # Select bboxes where scores > 0.
      non_zero_bboxes = bboxes[scores > 0]
      non_zero_scores = scores[scores > 0]
      non_zero_heights_in_pixels = heights_in_pixels[scores > 0]

      rotations = [0] * len(non_zero_bboxes)
      distances = [0] * len(non_zero_bboxes)
      if 'distance' in self._breakdown_metrics:
        # Compute all distances for non-zero-bboxes in one shot.
        distances = self._breakdown_metrics['distance'].Discretize(
            non_zero_bboxes)
      if 'rotation' in self._breakdown_metrics:
        rotations = self._breakdown_metrics['rotation'].Discretize(
            non_zero_bboxes)

      # Add each box to the list.
      #
      # NOTE: The length of this loop can be large (e.g., for an early
      # checkpoint), so any code inside of this for loop should be
      # double-checked for efficiency.
      dummy_speed = np.zeros((1, 2), dtype=np.float32)
      for box_id in range(non_zero_bboxes.shape[0]):
        boxes_for_class.Add(
            img_id=str_imgid,
            score=non_zero_scores[box_id],
            box=non_zero_bboxes[box_id],
            difficulty=0,
            distance=distances[box_id],
            num_points=0,
            rotation=rotations[box_id],
            height_in_pixels=non_zero_heights_in_pixels[box_id],
            speed=dummy_speed)

  def _EvaluateIfNecessary(self):
    """Evaluate all precision recall metrics."""
    if self._is_eval_complete:
      return
    compute_metrics_fn = functools.partial(
        self._ComputeFinalMetrics, classids=self.metadata.EvalClassIndices())
    for metric_class in self._breakdown_metrics.values():
      metric_class.ComputeMetrics(compute_metrics_fn)
    self._is_eval_complete = True

  @property
  def value(self):
    if self.params.metric_weights is None:
      # Choose a backwards compatible default.  The default assumes:
      #
      # if KITTI, metric_weights is all zeros except for car/moderate.
      #
      # if not KITTI, metric_weights are all equal.
      metric_weights = {}
      keys = list(self.metadata.DifficultyLevels().keys())
      for difficulty in keys:
        if 'moderate' in keys and len(keys) == 3:
          # The KITTI case.
          if difficulty == 'moderate':
            metric_weights[difficulty] = np.array([1., 0., 0.])
          else:
            metric_weights[difficulty] = np.array([0., 0., 0.])
        else:
          # Every difficulty and every class is weighted equally.
          metric_weights[difficulty] = np.array(
              [1.] * len(self.metadata.EvalClassIndices()))
    else:
      metric_weights = self.params.metric_weights

    # Compute weighted average of AP scores across all classes and difficulties.
    num_sum = 0.0
    denom_sum = 0.0
    for difficulty, value in self._AveragePrecisionByDifficulty().items():
      metric_weight = metric_weights[difficulty]
      denom = np.sum(metric_weight)
      if denom == 0.0:
        continue
      num_sum += np.sum(np.nan_to_num(value) * metric_weight)
      denom_sum += denom

    # Ensure at least some weights are specified
    assert denom_sum > 0., 'All AP metric weights were 0!'
    return num_sum / denom_sum

  def _AveragePrecisionByDifficulty(self):
    """Special case to identify mAP versus difficulty."""
    self._EvaluateIfNecessary()
    return self._breakdown_metrics['difficulty']._average_precisions  # pylint:disable=protected-access

  def Summary(self, name):
    self._EvaluateIfNecessary()

    ret = tf.Summary()

    # Put '.value' first (so it shows up in logs / summaries, etc).
    ret.value.add(tag='{}/weighted_mAP'.format(name), simple_value=self.value)

    average_precision_by_difficulty = self._AveragePrecisionByDifficulty()
    for i, j in enumerate(self.metadata.EvalClassIndices()):
      classname = self.metadata.ClassNames()[j]
      for difficulty in self.metadata.DifficultyLevels():
        tag_str = '{}/{}/AP_{}'.format(name, classname, difficulty)
        ap_value = average_precision_by_difficulty[difficulty][i]
        ret.value.add(tag=tag_str, simple_value=ap_value)

    for metric_class in self._breakdown_metrics.values():
      image_summaries = metric_class.GenerateSummaries(name)
      for image_summary in image_summaries:
        ret.value.extend(image_summary.value)
    return ret
