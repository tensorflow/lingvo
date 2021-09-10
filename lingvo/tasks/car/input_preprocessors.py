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
"""Input preprocessors."""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.tasks.car import car_lib
from lingvo.tasks.car import detection_3d_lib
from lingvo.tasks.car import geometry
from lingvo.tasks.car import ops
import numpy as np
# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.ops import inplace_ops
# pylint:enable=g-direct-tensorflow-import


def _ConsistentShuffle(tensors, seed):
  """Shuffle multiple tensors with the same shuffle order."""
  shuffled_idx = tf.range(tf.shape(tensors[0])[0])
  shuffled_idx = tf.random.shuffle(shuffled_idx, seed=seed)
  return tuple([tf.gather(t, shuffled_idx) for t in tensors])


def _GetApplyPointMaskFn(points_mask):
  """Returns a function that applies a mask to one of our points tensors."""

  def _ApplyPointMaskFn(points_tensor):
    """Applies a mask to the points tensor."""
    if points_tensor is None:
      return points_tensor
    return tf.boolean_mask(points_tensor, points_mask)

  return _ApplyPointMaskFn


def _Dense(sparse):
  return tf.sparse_to_dense(
      sparse_indices=sparse.indices,
      output_shape=sparse.dense_shape,
      sparse_values=sparse.values,
      default_value=0)


class Preprocessor(base_layer.BaseLayer):
  """Base class for input preprocessor.

  Input preprocessors expect the combined output of all extractors and performs
  a transformation on them. Input preprocessors can add/edit/remove fields
  from the NestedMap of features.

  Note: Features correspond to that for one example (no batch dimension).

  Sub-classes need to implement the following three functions:

  1) TransformFeatures(features): Given a NestedMap of features representing the
     output of all the extractors, apply a transformation on the features.

  2) TransformShapes(shapes): Given a corresponding NestedMap of shapes,
     produce a NestedMap of shapes that corresponds to the transformation of the
     features after TransformFeatures.

  3) TransformDTypes(dtypes): Given a corresponding NestedMap of dtypes,
     produce a NestedMap of dtypes that corresponds to the transformation of the
     features after TransformFeatures.

  The preprocessor is expected to explicitly pass through untouched fields.
  For example, a preprocessor that does data augmentation should modify the
  features NestedMap on the fields it cares about augmenting, and then return
  the features NestedMap.
  """

  @classmethod
  def Params(cls):
    """Default params."""
    p = super().Params()
    p.name = cls.__name__
    return p

  def FProp(self, theta, features):
    """Performs TransformFeatures."""
    del theta  # unused
    return self.TransformFeatures(features)

  def TransformFeatures(self, features):
    """Transforms the features for one example.

    Args:
      features: A `NestedMap` of tensors.

    Returns:
      A `NestedMap` of tensors corresponding.
    """
    raise NotImplementedError()

  def TransformBatchedFeatures(self, features):
    """Transforms the features for a batch of examples.

    Args:
      features: A `NestedMap` of batched tensors.

    Returns:
      A `NestedMap` of tensors corresponding.
    """
    dtypes = features.Transform(lambda v: v.dtype)
    dtypes = self.TransformDTypes(dtypes)
    # Default impl uses map_fn.
    result = tf.map_fn(
        self.TransformFeatures, elems=features, dtype=dtypes, back_prop=False)
    return result

  def TransformShapes(self, shapes):
    """Sets correct shapes corresponding to TransformFeatures.

    Args:
      shapes: A `NestedMap` of TensorShapes, corresponding to the
        pre-transformed features.

    Returns:
      A `NestedMap` of TensorShapes corresponding to the transformed features.
    """
    raise NotImplementedError()

  def TransformDTypes(self, dtypes):
    """Sets correct dtypes corresponding to TransformFeatures.

    Args:
      dtypes: A `NestedMap` of DTypes, corresponding to the pre-transformed
        features.

    Returns:
      A `NestedMap` of DTypes corresponding to the transformed features.
    """
    raise NotImplementedError()


class EntryPreprocessor(Preprocessor):
  """A Preprocessor that transforms a NestedMap sub-structure.

  Some preprocessors want to apply a function to any NestedMap whose key matches
  a specific prefix. An EntryPreprocessor provides an interface for specifying
  the function transformation for a NestedMap of inputs, adding, modifying, or
  deleting the entries in that NestedMap.

  For example, if an input contains a nested structure such as:
    - lasers.front.xyz
                  .features
    - lasers.side.xyz
                 .features

  and one wants to apply a transform that modifies the .xyz features
  on both structures, one can define an EntryPreprocessor that implements:

    UpdateEntry(entry):
    UpdateEntryShape(shapes):
    UpdateEntryDType(dtypes):

  and set self.params.prefixes = ['lasers.front', 'lasers.side']
  where the prefixes refer to a fully-qualified NestedMap sub-structure.

  The arguments to these functions will contain just the NestedMap structure
    whose key prefix can be found in self.params.prefixes.  One can then modify
    these structures as desired.

  Example:
    def UpdateEntry(self, entry):
       # entry is a NestedMap.
       assert 'xyz' in entry
       entry.xyz = self._ApplyFn(entry.xyz)
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('prefixes', ['pseudo_ri'], 'List of keys to apply to.')
    return p

  def _ApplyToMatchingStructure(self, nested_map, fn):
    """Apply fn to any NestedMap sub-structure whose prefix is in p.prefixes."""
    p = self.params
    # Don't mutate the original.
    nested_map = nested_map.DeepCopy()
    updated_entries = []
    for prefix in p.prefixes:
      entry = nested_map.GetItem(prefix)
      if not isinstance(entry, py_utils.NestedMap):
        raise TypeError('Prefix key {} selected a {}, not a NestedMap!'.format(
            prefix, type(entry)))
      fn(entry)
      updated_entries.append(entry)
    return nested_map, updated_entries

  def UpdateEntry(self, entry):
    """Update the Tensors in a NestedMap entry.

    Args:
      entry: A NestedMap of Tensors.
    """
    raise NotImplementedError()

  def UpdateEntryShape(self, shapes):
    """Update the shapes in a NestedMap entry.

    Args:
      shapes: A NestedMap of TensorShapes.
    """
    raise NotImplementedError()

  def UpdateEntryDType(self, dtypes):
    """Transform the dtypes in a NestedMap entry.

    Args:
      dtypes: A NestedMap of dtypes.
    """
    raise NotImplementedError()

  def TransformFeatures(self, features):
    features, _ = self._ApplyToMatchingStructure(features, self.UpdateEntry)
    return features

  def TransformShapes(self, shapes):
    shapes, _ = self._ApplyToMatchingStructure(shapes, self.UpdateEntryShape)
    return shapes

  def TransformDTypes(self, dtypes):
    dtypes, _ = self._ApplyToMatchingStructure(dtypes, self.UpdateEntryDType)
    return dtypes


class CreateDecoderCopy(Preprocessor):
  """Creates references to current lasers, images, and labels.

  This is useful if the data is further transformed.

  If desired, the keys that are copied can be customized by overriding the
  default keys param.

  This preprocessor expects features to optionally contain the following keys:
  - lasers - a NestedMap of tensors
  - images - a NestedMap of tensors
  - labels - a NestedMap of tensors

  Adds the following features (if the features existed):
    - decoder_copy.lasers - a copy of the lasers NestedMap
    - decoder_copy.images - a copy of the images NestedMap
    - decoder_copy.labels - a copy of the labels NestedMap

  The processor also by default pads the laser features; this can be disabled
  by setting the pad_lasers param to None.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('keys', ['lasers', 'labels', 'images'],
             'Keys to look for and copy if exists.')
    p.Define('parent_key', 'decoder_copy', 'The key to nest the copies under.')
    p.Define('pad_lasers', PadLaserFeatures.Params(),
             'Params for a layer that pads the laser features.')
    p.name = 'create_decoder_copy'
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.pad_lasers is not None:
      self.CreateChild('pad_lasers', p.pad_lasers)

  def _DeepCopyIfExists(self, keys, nested_map, parent_key):
    """Deep copy a specific key to a parent key if it exists."""
    for key in keys:
      if key in nested_map:
        if parent_key not in nested_map:
          nested_map[parent_key] = py_utils.NestedMap()
        nested_map[parent_key][key] = nested_map[key].DeepCopy()
    return nested_map

  def TransformFeatures(self, features):
    p = self.params
    features = self._DeepCopyIfExists(p.keys, features, p.parent_key)
    if p.pad_lasers is not None:
      features[p.parent_key] = self.pad_lasers.TransformFeatures(
          features[p.parent_key])
    return features

  def TransformShapes(self, shapes):
    p = self.params
    shapes = self._DeepCopyIfExists(p.keys, shapes, p.parent_key)
    if p.pad_lasers is not None:
      shapes[p.parent_key] = self.pad_lasers.TransformShapes(
          shapes[p.parent_key])
    return shapes

  def TransformDTypes(self, dtypes):
    p = self.params
    dtypes = self._DeepCopyIfExists(p.keys, dtypes, p.parent_key)
    if p.pad_lasers is not None:
      dtypes[p.parent_key] = self.pad_lasers.TransformDTypes(
          dtypes[p.parent_key])
    return dtypes


class FilterByKey(Preprocessor):
  """Filters features to keep only specified keys.

  This keeps only feature entries that are specified. This allows us to reduce
  the number of fields returned. For example, during training, one may not
  need the actual laser points if training with a pillars based model that
  has a preprocessor that already maps the points to grid.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'keep_key_prefixes', [''], 'Prefixes of keys to keep. If this '
        'contains the empty string, then it will keep all the keys.')
    return p

  def _FilterFn(self, key, entry):
    """Filter a nested map."""
    del entry  # unused
    p = self.params
    for prefix in p.keep_key_prefixes:
      if key.startswith(prefix):
        return True
    return False

  def TransformFeatures(self, features):
    return features.FilterKeyVal(self._FilterFn)

  def TransformShapes(self, shapes):
    return shapes.FilterKeyVal(self._FilterFn)

  def TransformDTypes(self, dtypes):
    return dtypes.FilterKeyVal(self._FilterFn)


class FilterGroundTruthByNumPoints(Preprocessor):
  """Removes ground truth boxes with less than params.min_num_points points.

  This preprocessor expects features to contain the following keys::
    labels.labels of shape [..., L]
    labels.bboxes_3d of shape [..., L, 7]
    labels.bboxes_3d_mask of shape [..., L]
    labels.unfiltered_bboxes_3d_mask of shape [..., L]
    labels.bboxes_3d_num_points of shape [..., L].

  Modifies the bounding box data to turn off ground truth objects that don't
  meet the params.min_num_points point filter:

    labels.labels: Boxes with less than params.min_num_points have their label
    set to params.background_id (defaults to 0).

    labels.bboxes_3d_mask: Boxes with less than params.min_num_points are set
    to 0.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'min_num_points', 1, 'The minimum number of points allowed before '
        'the associated ground truth box is turned off. Defaults to 1.')
    p.Define(
        'background_id', 0, 'The ID of the background class we set '
        'filtered boxes to. Defaults to 0.')
    return p

  def TransformFeatures(self, features):
    p = self.params
    bbox_is_valid = tf.greater_equal(features.labels.bboxes_3d_num_points,
                                     p.min_num_points)
    features.labels.labels = tf.where(
        bbox_is_valid, features.labels.labels,
        p.background_id * tf.ones_like(features.labels.labels))
    features.labels.bboxes_3d_mask *= tf.cast(bbox_is_valid, tf.float32)
    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class FilterGroundTruthByDifficulty(Preprocessor):
  """Removes groundtruth boxes based on detection difficulty.

  This preprocessor expects features to contain the following keys::
    labels.single_frame_detection_difficulties of shape [..., L]
    labels.labels of shape [..., L]
    labels.bboxes_3d_mask of shape [..., L]
    labels.unfiltered_bboxes_3d_mask of shape [..., L]

  The preprocessor masks out the bboxes_3d_mask / labels based on whether
  single_frame_detection_difficulties is greater than p.difficulty_threshold.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'background_id', 0, 'The ID of the background class we set '
        'filtered boxes to. Defaults to 0.')
    p.Define(
        'difficulty_threshold', 1,
        'Filter groundtruth bounding boxes whose detection difficulty is '
        'greater than `difficulty_threshold`')
    return p

  def TransformFeatures(self, features):
    p = self.params
    bbox_is_valid = tf.less_equal(
        features.labels.single_frame_detection_difficulties,
        p.difficulty_threshold)
    features.labels.labels = tf.where(
        bbox_is_valid, features.labels.labels,
        p.background_id * tf.ones_like(features.labels.labels))
    features.labels.bboxes_3d_mask *= tf.cast(bbox_is_valid, tf.float32)
    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class CountNumberOfPointsInBoxes3D(Preprocessor):
  """Computes bboxes_3d_num_points.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]
  - labels.bboxes_3d of shape [L, 7]
  - labels.bboxes_3d_mask of shape [L]

  and optionally points_padding of shape [P] corresponding to the padding.
  if points_padding is None, then all points are considered valid.

  Adds the following features:
    labels.bboxes_3d_num_points: [L] - integer tensor containing the number of
      laser points for each corresponding bbox.
  """

  def TransformFeatures(self, features):
    points_xyz = features.lasers.points_xyz
    if 'points_padding' in features.lasers:
      points_mask = 1 - features.lasers.points_padding
      points_xyz = tf.boolean_mask(points_xyz, points_mask)

    points_in_bboxes_mask = geometry.IsWithinBBox3D(points_xyz,
                                                    features.labels.bboxes_3d)
    bboxes_3d_num_points = tf.reduce_sum(
        tf.cast(points_in_bboxes_mask, tf.int32), axis=0, keepdims=False)
    bboxes_3d_num_points *= tf.cast(features.labels.bboxes_3d_mask, tf.int32)

    features.labels.bboxes_3d_num_points = bboxes_3d_num_points
    return features

  def TransformShapes(self, shapes):
    num_bboxes = shapes.labels.bboxes_3d[0]
    shapes.labels.bboxes_3d_num_points = tf.TensorShape([num_bboxes])
    return shapes

  def TransformDTypes(self, dtypes):
    dtypes.labels.bboxes_3d_num_points = tf.int32
    return dtypes


class AddPerPointLabels(Preprocessor):
  """Computes the class and bbox id of each point.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]
  - labels.bboxes_3d of shape [L, 7]
  - labels.labels of shape [L]

  This makes an assumption that each point is only in 1 box, which should
  almost always true in 3D. In cases where this is not true, the largest
  label integer and largest bbox_id will be assigned.

  NOTE: Be very careful that this is performed after any modifications
  to the semantic labels of each point in the pointcloud. Examples of this
  would be operators like GroundTruthAugmentation, or DropBoxesOutOfRange.

  Adds the following features:
    lasers.points_label: [P] - integer tensor containing the class id of each
      point.
    lasers.points_bbox_id: [P] - integer tensor containing box id of each
      point from 0 to num_bboxes, where an id of num_bboxes indicates a
      background point.
    lasers.points_bbox_3d: [P, 7] - float tensor containing bounding box of
      each point.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'per_dimension_adjustment', None,
        'A list of len 3 of floats with the amount (in meters) to add to '
        'each dimension of the box before using it to select points. '
        'If enabled, this is designed to protect against overly tight box '
        'annotations that appear in KITTI.')
    return p

  def TransformFeatures(self, features):
    p = self.params
    points_xyz = features.lasers.points_xyz
    bboxes_3d = features.labels.bboxes_3d
    num_points, _ = py_utils.GetShape(points_xyz)
    num_bboxes, _ = py_utils.GetShape(bboxes_3d)

    if p.per_dimension_adjustment:
      if len(p.per_dimension_adjustment) != 3:
        raise ValueError(
            'param `per_dimension_adjustment` expected to be len 3.')
      dims_adjustment = tf.constant([0, 0, 0] + p.per_dimension_adjustment +
                                    [0])
      bboxes_3d = bboxes_3d + dims_adjustment

    # Find which points are in each box and what class each box is.
    points_in_bboxes_mask = geometry.IsWithinBBox3D(points_xyz, bboxes_3d)
    points_in_bboxes_mask = tf.cast(points_in_bboxes_mask, tf.int32)
    points_in_bboxes_mask = py_utils.HasShape(points_in_bboxes_mask,
                                              [num_points, num_bboxes])

    # points_in_bboxes_mask is a [num_points, num_bboxes] 0/1 tensor
    # indicating whether that point is in a given box.
    # Each point should only be in one box, so after broadcasting the label
    # across the binary mask, we do a reduce_max to get the max label id
    # for each point. Since each point only belongs to one box, it will be
    # the only non-zero (background) label in that box.
    # Note: We assume background to be class_id == 0
    points_label = tf.reduce_max(
        points_in_bboxes_mask * features.labels.labels, axis=1)
    points_bbox_id = tf.argmax(
        points_in_bboxes_mask, axis=1, output_type=tf.int32)
    # If the class is background, make its id == num_bboxes
    points_bbox_id = tf.where(points_label > 0, points_bbox_id,
                              tf.broadcast_to(num_bboxes, [num_points]))

    # For each point, get the bbox_3d data.
    dummy_bbox = tf.constant([[0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
    bboxes_3d = tf.concat([bboxes_3d, dummy_bbox], axis=0)
    points_bbox_3d = tf.gather(bboxes_3d, points_bbox_id)

    points_label = tf.reshape(points_label, [num_points])
    points_bbox_id = tf.reshape(points_bbox_id, [num_points])
    features.lasers.points_label = points_label
    features.lasers.points_bbox_id = points_bbox_id
    features.lasers.points_bbox_3d = points_bbox_3d
    return features

  def TransformShapes(self, shapes):
    num_points = shapes.lasers.points_xyz[0]
    shapes.lasers.points_label = tf.TensorShape([num_points])
    shapes.lasers.points_bbox_id = tf.TensorShape([num_points])
    shapes.lasers.points_bbox_3d = tf.TensorShape([num_points, 7])
    return shapes

  def TransformDTypes(self, dtypes):
    dtypes.lasers.points_label = tf.int32
    dtypes.lasers.points_bbox_id = tf.int32
    dtypes.lasers.points_bbox_3d = tf.float32
    return dtypes


class PointsToGrid(Preprocessor):
  """Bins points to a 3D-grid using custom op: ops.point_to_grid.

  Expects features to have keys:
  - lasers.points_xyz of shape [P, 3]

  and optionally points_padding of shape [P] corresponding to the padding.
  if points_padding is None, then all points are considered valid.

  If normalizing the labels is enabled, then also expects:
  - labels.weights
  - labels.bboxes_td
  - labels.bboxes_td_mask
  - labels.bboxes_3d_mask

  Let:
    gx, gy, gz = p.grid_size
    F = 3 + num_laser_features

  Adds the following features:
    grid_centers: [gx, gy, gz, 3]: For each grid cell, the (x,y,z)
      floating point coordinate of its center.
    grid_num_points: [gx, gy, gz]: The number of points in each grid
      cell (integer).
    laser_grid: [gx, gy, gz, num_points_per_cell, F] - A 5D floating
      point Tensor containing the laser data placed into a fixed grid.

  Modifies the bboxes in labels to also be within the grid range x/y by default.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_points_per_cell', 100,
             'The maximum number of points per cell.')
    p.Define('grid_size', (40, 40, 1), 'Grid size along x,y,z axis.')

    # The max range of x and y is [-80, 80].
    p.Define('grid_range_x', (-80, 80), 'The X-axis Range covered by the grid')
    p.Define('grid_range_y', (-80, 80), 'The Y-axis Range covered by the grid')
    p.Define('grid_range_z', (-2, 4), 'The Z-axis Range covered by the grid')

    p.Define('normalize_td_labels', True,
             'Whether to clip the labels to the grid limits.')
    return p

  def _NormalizeLabels(self, ymin, xmin, ymax, xmax, x_range, y_range):
    """Normalizes the bboxes within a given range."""
    assert x_range, 'Must specify x_range if clipping.'
    assert y_range, 'Must specify y_range if clipping.'
    assert len(x_range) == 2, 'x_range %s must be 2 elements.' % x_range
    assert len(y_range) == 2, 'y_range %s must be 2 elements.' % y_range

    x_range_min = x_range[0]
    x_range_len = x_range[1] - x_range[0]
    y_range_min = y_range[0]
    y_range_len = y_range[1] - y_range[0]

    xmin = tf.cast(xmin - x_range_min, tf.float32) / tf.cast(
        x_range_len, tf.float32)
    xmax = tf.cast(xmax - x_range_min, tf.float32) / tf.cast(
        x_range_len, tf.float32)
    ymin = tf.cast(ymin - y_range_min, tf.float32) / tf.cast(
        y_range_len, tf.float32)
    ymax = tf.cast(ymax - y_range_min, tf.float32) / tf.cast(
        y_range_len, tf.float32)

    return ymin, xmin, ymax, xmax

  def TransformFeatures(self, features):
    p = self.params

    points_xyz = features.lasers.points_xyz
    points_feature = features.lasers.points_feature
    if ('points_padding' in features.lasers and
        features.lasers.points_padding is not None):
      points_mask = 1 - features.lasers.points_padding
      points_xyz = tf.boolean_mask(points_xyz, points_mask)
      points_feature = tf.boolean_mask(points_feature, points_mask)

    points_full = tf.concat([points_xyz, points_feature], axis=-1)
    points_grid_full, grid_centers, num_points = ops.point_to_grid(
        points_full, p.num_points_per_cell, p.grid_size[0], p.grid_size[1],
        p.grid_size[2], p.grid_range_x, p.grid_range_y, p.grid_range_z)

    features.laser_grid = points_grid_full
    features.grid_centers = grid_centers
    features.grid_num_points = num_points

    if p.normalize_td_labels:
      # Normalize bboxes_td w.r.t grid range.
      obb = features.labels
      x_range = p.grid_range_x
      y_range = p.grid_range_y
      ymin, xmin, ymax, xmax = tf.unstack(obb.bboxes_td[..., :4], axis=-1)
      ymin, xmin, ymax, xmax = self._NormalizeLabels(
          ymin, xmin, ymax, xmax, x_range=x_range, y_range=y_range)
      obb.bboxes_td = tf.concat(
          [tf.stack([ymin, xmin, ymax, xmax], axis=-1), obb.bboxes_td[..., 4:]],
          axis=-1)

    return features

  def TransformShapes(self, shapes):
    p = self.params
    shapes.grid_centers = tf.TensorShape(list(p.grid_size) + [3])
    shapes.grid_num_points = tf.TensorShape(list(p.grid_size))
    shapes.laser_grid = tf.TensorShape(
        list(p.grid_size) +
        [p.num_points_per_cell, 3 + shapes.lasers.points_feature[-1]])
    return shapes

  def TransformDTypes(self, dtypes):
    dtypes.grid_centers = tf.float32
    dtypes.grid_num_points = tf.int32
    dtypes.laser_grid = tf.float32
    return dtypes


class _PointPillarGridSettings:
  """Settings for PointPillars model defined in paper.

  https://arxiv.org/abs/1812.05784
  """
  # Chooses grid sizes that are a multiple of 16 to support point pillars
  # model requirements.  These also happen to match the values
  # in the PointPillars paper (voxel width of 0.16m in x, y)
  GRID_X = 432
  GRID_Y = 496
  GRID_Z = 1

  # These fields are set in the subclasses.
  GRID_X_RANGE = None
  GRID_Y_RANGE = None
  GRID_Z_RANGE = None

  @classmethod
  def UpdateGridParams(cls, grid_params):
    """Apply PointPillars settings to grid_params."""
    grid_params.grid_size = (cls.GRID_X, cls.GRID_Y, cls.GRID_Z)
    grid_params.grid_range_x = cls.GRID_X_RANGE
    grid_params.grid_range_y = cls.GRID_Y_RANGE
    grid_params.grid_range_z = cls.GRID_Z_RANGE

  @classmethod
  def UpdateAnchorGridParams(cls, anchor_params, output_stride=2):
    """Apply PointPillars settings to anchor_params."""
    # Set anchor settings to match grid settings.
    # Grid size for anchors is half the resolution.
    anchor_params.grid_size = (cls.GRID_X // output_stride,
                               cls.GRID_Y // output_stride, cls.GRID_Z)
    anchor_params.grid_range_x = cls.GRID_X_RANGE
    anchor_params.grid_range_y = cls.GRID_Y_RANGE
    # Grid along z axis should be pinned to 0.
    anchor_params.grid_range_z = (0, 0)


def MakeGridSettings(grid_x_range, grid_y_range, grid_z_range, grid_x, grid_y,
                     grid_z):
  """Returns configured class for PointPillar grid settings."""

  class GridSettings(_PointPillarGridSettings):
    GRID_X_RANGE = grid_x_range
    GRID_Y_RANGE = grid_y_range
    GRID_Z_RANGE = grid_z_range
    GRID_X = grid_x
    GRID_Y = grid_y
    GRID_Z = grid_z

  return GridSettings


PointPillarGridCarSettings = MakeGridSettings(
    grid_x_range=(0, 69.12),
    grid_y_range=(-39.68, 39.68),
    grid_z_range=(-3, 1),
    grid_x=432,
    grid_y=496,
    grid_z=1)

PointPillarGridPedCycSettings = MakeGridSettings(
    grid_x_range=(0, 47.36),
    grid_y_range=(-19.84, 19.84),
    grid_z_range=(-2.5, 0.5),
    grid_x=432,
    grid_y=496,
    grid_z=1)


class GridToPillars(Preprocessor):
  """Create pillars from a grid of points.

  Expects features to have keys:
    grid_centers: [gx, gy, gz, 3]

    grid_num_points: [gx, gy, gz]

    laser_grid: [gx, gy, gz, num_points_per_cell, F]

  Adds the following features:
    point_count: [num_pillars]. The number of points in the pillar.

    point_locations: [num_pillars, 3]. The grid location of each pillar.

    pillar_points: [num_pillars, num_points_per_cell, F]. Points of each
    pillar.

  Drops the following features by default:
    laser_grid
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_points_per_cell', 100,
             'The maximum number of points per cell.')
    p.Define('num_pillars', 12000, 'The maximum number of pillars to produce.')
    p.Define('drop_laser_grid', True, 'Whether to drop the laser_grid feature.')
    # The density based sampler is more expensive.
    p.Define('use_density_sampler', False,
             'Use a density based sampler during pillar selection.')
    return p

  def _GumbelTransform(self, probs):
    """Adds gumbel noise to log probabilities for multinomial sampling.

    This enables fast sampling from a multinomial distribution without
    replacement. See https://arxiv.org/abs/1611.01144 for details.
    A colab that demonstrates this in practice is here:
    http://colab/drive/1iuMt2n_r7dKPQG9T0UVMuK3fkbBayKjd

    Args:
      probs: A 1-D float tensor containing probabilities, summing to 1.

    Returns:
      A 1-D float tensor of the same size of probs, with gumbel noise added to
      log probabilities. Taking the top k elements from this provides a
      multinomial sample without replacement.
    """
    p = self.params
    log_prob = tf.math.log(probs)
    probs_shape = tf.shape(probs)
    uniform_samples = tf.random.uniform(
        shape=probs_shape,
        dtype=probs.dtype,
        seed=p.random_seed,
        name='uniform_samples')
    gumbel_noise = -tf.math.log(-tf.math.log(uniform_samples))
    return gumbel_noise + log_prob

  def _DensitySample(self, num_points):
    p = self.params

    # Flatten to [nx * ny * nz] for convenience during sampling.
    num_grid_points = np.prod(p.grid_size)
    flattened_num_points = tf.reshape(num_points, [num_grid_points])

    # Normalize flattened_num_points to sum to 1.
    flattened_num_points = tf.cast(flattened_num_points, tf.float32)
    flattened_num_points /= tf.reduce_sum(flattened_num_points)

    # TODO(jngiam): Consider generalizing this to enable other methods of
    # sampling: e.g., use largest deviation in z-axis. The gumbel transform
    # can still be applied regardless.

    # Add gumbel noise for multinomial sampling.
    sampling_logits = self._GumbelTransform(flattened_num_points)
    _, locations = tf.nn.top_k(
        sampling_logits, k=min(p.num_pillars, num_grid_points))

    # Unravel coordinates back to grid locations.
    locations = tf.unravel_index(locations, p.grid_size)

    # Unravel index will return a 3 x num_locations tensor, this needs to be
    # transposed so that we have it as num_locations x 3.
    locations = py_utils.HasShape(locations, [3, -1])
    locations = tf.transpose(locations)

    return locations

  def TransformFeatures(self, features):
    p = self.params

    num_points = features.grid_num_points
    if p.use_density_sampler:
      locations = self._DensitySample(num_points)
    else:
      # Select non-empty cells uniformly at random.
      locations = tf.random.shuffle(tf.cast(tf.where(num_points > 0), tf.int32))

    num_features = py_utils.GetShape(features.laser_grid)[-1]

    # [nx, ny, nz, np, 4] (x, y, z, f)
    points = features.laser_grid
    # [K, np, 4] (x, y, z, f)
    points = tf.gather_nd(points, locations)
    # [nx, ny, nz, 1, 3] (cx, cy, cz)
    centers = features.grid_centers[..., tf.newaxis, :]
    # [K, 1, 3] (cx, cy, cz)
    centers = tf.gather_nd(centers, locations)
    # NOTE: If there are fewer pillars than p.num_pillars, the following
    # padding creates many 'fake' pillars at grid cell (0, 0, 0) with
    # an all-zero pillar. Hopefully, the model can learn to ignore these.
    #
    # pillar_points[i, :, :] is the pillar located at pillar_locations[i, :3],
    # and pillar_points[i, :, :] == points_grid_full[pillar_locations[i, :3]].
    #   for 0 <= i < pillar_count;
    # pillar_locations[i, :3] are zero-ed, for i >= pillar_count.
    features.pillar_count = tf.shape(locations)[0]
    features.pillar_locations = py_utils.PadOrTrimTo(locations,
                                                     [p.num_pillars, 3])
    features.pillar_points = py_utils.PadOrTrimTo(
        points, [p.num_pillars, p.num_points_per_cell, num_features])
    features.pillar_centers = py_utils.PadOrTrimTo(centers,
                                                   [p.num_pillars, 1, 3])

    if p.drop_laser_grid:
      del features['laser_grid']

    return features

  def TransformShapes(self, shapes):
    p = self.params
    num_features = shapes.laser_grid[-1]
    shapes.pillar_count = tf.TensorShape([])
    shapes.pillar_locations = tf.TensorShape([p.num_pillars, 3])
    shapes.pillar_points = tf.TensorShape(
        [p.num_pillars, p.num_points_per_cell, num_features])
    shapes.pillar_centers = tf.TensorShape([p.num_pillars, 1, 3])
    if p.drop_laser_grid:
      del shapes['laser_grid']
    return shapes

  def TransformDTypes(self, dtypes):
    p = self.params
    dtypes.pillar_count = tf.int32
    dtypes.pillar_locations = tf.int32
    dtypes.pillar_points = tf.float32
    dtypes.pillar_centers = tf.float32
    if p.drop_laser_grid:
      del dtypes['laser_grid']
    return dtypes


class GridAnchorCenters(Preprocessor):
  """Create anchor centers on a grid.

  Anchors are placed in the middle of each grid cell. For example, on a 2D grid
  range (0 -> 10, 0 -> 10) with a 10 x 5 grid size, the anchors will be placed
  at [(0.5, 1), (0.5, 3), ... , (9.5, 7), (9.5, 9)].

  Adds the following features:
    anchor_centers: [num_locations, 3] - Floating point output containing the
      center (x, y, z) locations for tiling anchor boxes.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'grid_size', (20, 20, 1), 'Grid size along x,y,z axis. This will '
        'be used to generate the anchor center locations. Note that this '
        'would likely be different from the grid_* parameters in '
        'LaserGridExtractor: the grid extractor may choose to extract '
        'points more densely. Instead, this should correspond to the '
        'model\'s prediction layer: the predicted anchor box residuals '
        'should match this grid.')
    p.Define('grid_range_x', (-25, 25), 'The x-axis range covered by the grid.')
    p.Define('grid_range_y', (-25, 25), 'The y-axis range covered by the grid.')
    p.Define('grid_range_z', (0, 0), 'The z-axis range covered by the grid.')
    return p

  def TransformFeatures(self, features):
    p = self.params
    utils_3d = detection_3d_lib.Utils3D()
    grid_size_x, grid_size_y, grid_size_z = p.grid_size
    grid_shape = list(p.grid_size) + [3]
    anchor_centers = utils_3d.CreateDenseCoordinates([
        list(p.grid_range_x) + [grid_size_x],
        list(p.grid_range_y) + [grid_size_y],
        list(p.grid_range_z) + [grid_size_z],
    ],
                                                     center_in_cell=True)
    features.anchor_centers = tf.reshape(anchor_centers, grid_shape)

    return features

  def TransformShapes(self, shapes):
    p = self.params
    shapes.anchor_centers = tf.TensorShape(list(p.grid_size) + [3])
    return shapes

  def TransformDTypes(self, dtypes):
    dtypes.anchor_centers = tf.float32
    return dtypes


class SparseCenterSelector(Preprocessor):
  """Select centers for anchors and cells.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]

  and optionally points_padding of shape [P] corresponding to the padding.
  if points_padding is None, then all points are considered valid.

  If lasers.num_seeded_points of shape [] is provided, it indicates that the
  first num_seeded_points of lasers.points_xyz should be used as seeds for
  farthest point sampling (e.g., always chosen).  Currently the concept
  of seeding is not implemented for anything but farthest point sampling.

  Adds the following features:
    anchor_centers: [num_cell_centers, 3] - Floating point output containing the
      center (x, y, z) locations for tiling anchor boxes.
    cell_center_xyz: [num_cell_centers, 3] - Floating point output containing
      the center (x, y, z) locations for each cell to featurize.
  """

  _SAMPLING_METHODS = ['farthest_point', 'random_uniform']

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_cell_centers', 256, 'Number of centers.')
    p.Define(
        'features_preparation_layers', [],
        'A list of Params for layers to run on the features before '
        'performing farthest point sampling. For example, one may wish to '
        'drop points out of frustum for KITTI before selecting centers. '
        'Note that these layers will not mutate the original features, '
        'instead, a copy will be made.')
    p.Define(
        'sampling_method', 'farthest_point',
        'Which sampling method to use. One of {}'.format(cls._SAMPLING_METHODS))
    p.Define(
        'fix_z_to_zero', True, 'Whether to fix z to 0 when retrieving the '
        'center xyz coordinates.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    if p.sampling_method not in self._SAMPLING_METHODS:
      raise ValueError('Param `sampling_method` must be one of {}.'.format(
          self._SAMPLING_METHODS))
    if p.features_preparation_layers is not None:
      self.CreateChildren('features_preparation_layers',
                          p.features_preparation_layers)

  def _FarthestPointSampleCenters(self, points_xyz, num_seeded_points):
    """Samples centers with Farthest Point Sampling.

    Args:
      points_xyz: An unpadded tf.float32 Tensor of shape [P, 3] with per point
        (x, y, z) locations. We expect any padded points to be removed before
        this function is called.
      num_seeded_points: integer indicating how many of the first
        num_seeded_points points in points_xyz should be considered
        as seeds for FPS (always chosen).

    Returns:
      A tf.float32 Tensor of shape [p.num_cell_centers, 3] with selected centers
      to use as anchors.
    """
    p = self.params
    num_points = tf.shape(points_xyz)[0]
    points_padding = tf.zeros((num_points,), dtype=tf.float32)
    padded_num_points = tf.maximum(num_points, p.num_cell_centers)

    # Pad both the points and padding if for some reason the input pointcloud
    # has less points than p.num_cell_centers.
    points_xy = py_utils.PadOrTrimTo(points_xyz[:, :2], [padded_num_points, 2])
    points_padding = py_utils.PadOrTrimTo(
        points_padding, [padded_num_points], pad_val=1.0)

    sampled_idx, _ = car_lib.FarthestPointSampler(
        points_xy[tf.newaxis, ...],
        points_padding[tf.newaxis, ...],
        p.num_cell_centers,
        num_seeded_points=num_seeded_points,
        random_seed=p.random_seed)
    sampled_idx = sampled_idx[0, :]

    # Gather centers.
    if p.fix_z_to_zero:
      centers = tf.concat([
          tf.gather(points_xy, sampled_idx),
          tf.zeros((p.num_cell_centers, 1)),
      ], axis=-1)  # pyformat: disable
    else:
      centers = tf.gather(points_xyz, sampled_idx)

    return centers

  def _RandomUniformSampleCenters(self, points_xyz):
    """Samples centers with Random Uniform Sampling.

    Args:
      points_xyz: An unpadded tf.float32 Tensor of shape [P, 3] with per point
        (x, y, z) locations. We expect any padded points to be removed before
        this function is called.

    Returns:
      A tf.float32 Tensor of shape [p.num_cell_centers, 3] with selected centers
      to use as anchors.
    """
    p = self.params
    # We want the center Z value to be 0 so just exclude it
    centers_xy = tf.random.shuffle(points_xyz[:, :2], seed=p.random_seed)
    selected_centers_xy = py_utils.PadOrTrimTo(centers_xy,
                                               [p.num_cell_centers, 2])
    return tf.concat([selected_centers_xy,
                      tf.zeros((p.num_cell_centers, 1))],
                     axis=-1)

  def _SampleCenters(self, points_xyz, num_seeded_points):
    p = self.params
    if p.sampling_method == 'farthest_point':
      return self._FarthestPointSampleCenters(points_xyz, num_seeded_points)
    elif p.sampling_method == 'random_uniform':
      if num_seeded_points > 0:
        raise NotImplementedError(
            'Random sampling with seeded points not yet implemented.')
      return self._RandomUniformSampleCenters(points_xyz)
    else:
      raise ValueError('Param `sampling_method` must be one of {}.'.format(
          self._SAMPLING_METHODS))

  def TransformFeatures(self, features):
    p = self.params

    prepared_features = features.DeepCopy()
    for prep_layer in self.features_preparation_layers:
      prepared_features = prep_layer.FPropDefaultTheta(prepared_features)

    num_seeded_points = prepared_features.lasers.get('num_seeded_points', 0)
    points_data = prepared_features.lasers

    points_xyz = points_data.points_xyz
    if 'points_padding' in points_data:
      points_padding = points_data.points_padding
      points_mask = 1 - points_padding
      points_xyz = tf.boolean_mask(points_xyz, points_mask)

    centers = self._SampleCenters(points_xyz, num_seeded_points)
    centers = py_utils.HasShape(centers, [p.num_cell_centers, 3])

    features.anchor_centers = centers
    features.cell_center_xyz = centers

    return features

  def TransformShapes(self, shapes):
    p = self.params
    shapes.anchor_centers = tf.TensorShape([p.num_cell_centers, 3])
    shapes.cell_center_xyz = tf.TensorShape([p.num_cell_centers, 3])
    return shapes

  def TransformDTypes(self, dtypes):
    dtypes.anchor_centers = tf.float32
    dtypes.cell_center_xyz = tf.float32
    return dtypes


class SparseCellGatherFeatures(Preprocessor):
  """Select local features for each cell.

  This preprocessor expects features to contain:
  - lasers.points_xyz of shape [P, 3]
  - lasers.points_feature of shape [P, F]
  - cell_center_xyz of shape [C, 3]

  and optionally points_padding of shape [P] corresponding to the padding.
  if points_padding is None, then all points are considered valid.

  Adds the following features:
    cell_points_xyz: [num_centers, num_points_per_cell, 3] - Floating point
      output containing the (x, y, z) locations for each point for a given
      center.
    cell_feature: [num_centers, num_points_per_cell, F] - Floating point output
      containing the features for each point for a given center.
    cell_points_padding: [num_centers, num_points_per_cell] - 0/1 padding
      for the points in each cell.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_points_per_cell', 128, 'The number of points per cell.')
    p.Define('max_distance', 3.0, 'Max distance of point to cell center.')
    p.Define(
        'sample_neighbors_uniformly', False,
        'Whether to sample the neighbor points for every cell center '
        'uniformly at random. If False, this will default to selecting by '
        'distance.')
    return p

  def TransformFeatures(self, features):
    p = self.params

    num_centers = py_utils.GetShape(features.cell_center_xyz, 1)[0]
    num_features = py_utils.GetShape(features.lasers.points_feature)[-1]

    points_xyz = features.lasers.points_xyz
    points_feature = features.lasers.points_feature
    if 'points_padding' in features.lasers:
      points_mask = 1 - features.lasers.points_padding
      points_xyz = tf.boolean_mask(points_xyz, points_mask)
      points_feature = tf.boolean_mask(points_feature, points_mask)

    # Note: points_xyz and points_feature must be unpadded as we pass
    # padding=None to neighborhood indices. Ensuring that it is unpadded
    # helps improve performance.

    # Get nearby points using kNN.
    sample_indices, sample_indices_padding = car_lib.NeighborhoodIndices(
        tf.expand_dims(points_xyz, 0),
        tf.expand_dims(features.cell_center_xyz, 0),
        p.num_points_per_cell,
        points_padding=None,
        max_distance=p.max_distance,
        sample_neighbors_uniformly=p.sample_neighbors_uniformly)

    # Take first example since NeighboorhoodIndices expects batch dimension.
    sample_indices = sample_indices[0, :, :]
    sample_indices_padding = sample_indices_padding[0, :, :]

    sample_indices = py_utils.HasShape(sample_indices,
                                       [num_centers, p.num_points_per_cell])

    cell_points_xyz = tf.gather(points_xyz, sample_indices)
    cell_points_xyz = py_utils.HasShape(cell_points_xyz,
                                        [num_centers, p.num_points_per_cell, 3])

    cell_feature = tf.gather(points_feature, sample_indices)
    cell_feature = py_utils.HasShape(
        cell_feature, [num_centers, p.num_points_per_cell, num_features])

    cell_points_padding = py_utils.HasShape(
        sample_indices_padding, [num_centers, p.num_points_per_cell])

    features.update({
        'cell_points_xyz': cell_points_xyz,
        'cell_feature': cell_feature,
        'cell_points_padding': cell_points_padding,
    })
    return features

  def TransformShapes(self, shapes):
    p = self.params
    num_centers = shapes.cell_center_xyz[0]
    base_shape = [num_centers, p.num_points_per_cell]
    num_features = shapes.lasers.points_feature[-1]
    shapes.cell_points_xyz = tf.TensorShape(base_shape + [3])
    shapes.cell_feature = tf.TensorShape(base_shape + [num_features])
    shapes.cell_points_padding = tf.TensorShape(base_shape)
    return shapes

  def TransformDTypes(self, dtypes):
    dtypes.cell_points_xyz = tf.float32
    dtypes.cell_feature = tf.float32
    dtypes.cell_points_padding = tf.float32
    return dtypes


class SparseCellCentersTopK(Preprocessor):
  """Given selected centers and gathered points/features, apply a filter.

  This preprocessor expects features to contain `cell_center_xyz` and all
  entries in params.features_to_modify, and that the leading dimension should
  all be the same (num_cell_centers from SparseCenterSelector).

  We then modify all values in features that are specified in
  params.features_to_modify by sorting them with the specified sort function
  (specified by params.sort_by) operating on features.cell_center_xyz, and then
  taking the top K (specified by params.num_cell_centers) along the first
  dimension.
  """

  _REGISTERED_SORT_FUNCTIONS = ['distance']

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_cell_centers', 512, 'The number of centers after filtering.')
    p.Define(
        'sort_by', 'distance', 'A string specifying which sort function '
        'to use. Currently we just support `distance`.')
    p.Define('features_to_modify', [
        'cell_center_xyz', 'anchor_centers', 'cell_points_xyz', 'cell_feature',
        'cell_points_padding'
    ], 'A list of keys from the features dict to modify.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.sort_by not in self._REGISTERED_SORT_FUNCTIONS:
      raise ValueError('{} not supported. We only support {}.'.format(
          p.sort_by, self._REGISTERED_SORT_FUNCTIONS))
    if len(p.features_to_modify) < 1:
      raise ValueError('Need to modify at least one feature.')

  def _SortByDistance(self, features):
    dist = tf.linalg.norm(features.cell_center_xyz, axis=-1)
    return tf.argsort(dist, axis=-1, direction='ASCENDING')

  def _Sort(self, features):
    p = self.params
    if p.sort_by == 'distance':
      return self._SortByDistance(features)
    else:
      raise ValueError('Unsupported sort function: {}.'.format(p.sort_by))

  def TransformFeatures(self, features):
    p = self.params
    sort_indices = self._Sort(features)
    sort_indices_top_k = sort_indices[:p.num_cell_centers, ...]

    # Gather each of the relevant items
    for key in p.features_to_modify:
      shape = py_utils.GetShape(features[key])
      output_shape = [p.num_cell_centers] + shape[1:]
      features[key] = py_utils.PadOrTrimTo(
          tf.gather(features[key], sort_indices_top_k), output_shape)
    return features

  def TransformShapes(self, shapes):
    p = self.params
    for key in p.features_to_modify:
      shapes[key] = tf.TensorShape([p.num_cell_centers] + shapes[key][1:])
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class TileAnchorBBoxes(Preprocessor):
  """Creates anchor_bboxes given anchor_centers.

  This preprocessor expects features to contain the following keys:
  - anchor_centers of shape [...base shape..., 3]

  Adds the following features:
    anchor_bboxes: base_shape + [7] - Floating point anchor box
      output containing the anchor boxes and the 7 floating point
      values for each box that define the box (x, y, z, dx, dy, dz, phi).
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('anchor_box_dimensions', [],
             'List of anchor box sizes per center.')
    p.Define('anchor_box_offsets', [], 'List of anchor box offsets per center.')
    p.Define('anchor_box_rotations', [],
             'List of anchor box rotations per center.')
    return p

  def TransformFeatures(self, features):
    p = self.params
    utils_3d = detection_3d_lib.Utils3D()

    assert p.anchor_box_dimensions
    assert p.anchor_box_offsets
    assert p.anchor_box_rotations

    base_shape = py_utils.GetShape(features.anchor_centers)[:-1]
    num_box_per_center = len(p.anchor_box_dimensions)

    anchor_centers = tf.reshape(features.anchor_centers, [-1, 3])
    anchor_bboxes = utils_3d.MakeAnchorBoxes(
        anchor_centers, tf.identity(p.anchor_box_dimensions),
        tf.identity(p.anchor_box_offsets), tf.identity(p.anchor_box_rotations))
    features.anchor_bboxes = tf.reshape(anchor_bboxes,
                                        base_shape + [num_box_per_center, 7])

    return features

  def TransformShapes(self, shapes):
    p = self.params
    base_shape = shapes.anchor_centers[:-1]
    num_box_per_center = len(p.anchor_box_dimensions)
    shapes.anchor_bboxes = base_shape.concatenate([num_box_per_center, 7])
    return shapes

  def TransformDTypes(self, dtypes):
    dtypes.anchor_bboxes = tf.float32
    return dtypes


class _AnchorBoxSettings:
  """Helper class to parameterize and update anchor box settings."""
  # Implementations should fill out the following class members.
  DIMENSION_PRIORS = []
  ROTATIONS = []
  CENTER_X_OFFSETS = []
  CENTER_Y_OFFSETS = []
  CENTER_Z_OFFSETS = []

  @classmethod
  def NumAnchors(cls):
    return np.prod([
        len(cls.DIMENSION_PRIORS),
        len(cls.ROTATIONS),
        len(cls.CENTER_X_OFFSETS),
        len(cls.CENTER_Y_OFFSETS),
        len(cls.CENTER_Z_OFFSETS)
    ])

  @classmethod
  def GenerateAnchorSettings(cls):
    """Generate anchor settings.

    Returns:
      A `NestedMap` containing three lists of the same length:
        - anchor_box_dimensions
        - anchor_box_rotations
        - anchor_box_offsets

      These can be used with the TileAnchorBBoxes preprocessor.
    """
    anchor_box_dimensions = []
    anchor_box_rotations = []
    anchor_box_offsets = []

    # The following is equivalent to a formulation of itertools.product, but
    # is explicitly listed for readability.

    # *Please note*: The ordering is important for ModelV2, which makes
    # assumptions that the offset dimensions come first.
    for cx in cls.CENTER_X_OFFSETS:
      for cy in cls.CENTER_Y_OFFSETS:
        for cz in cls.CENTER_Z_OFFSETS:
          for rot in cls.ROTATIONS:
            for dims in cls.DIMENSION_PRIORS:
              anchor_box_dimensions += [dims]
              anchor_box_rotations += [rot]
              anchor_box_offsets += [(cx, cy, cz)]

    # Check one of the lists has entries.
    assert anchor_box_dimensions

    return py_utils.NestedMap(
        anchor_box_dimensions=anchor_box_dimensions,
        anchor_box_rotations=anchor_box_rotations,
        anchor_box_offsets=anchor_box_offsets)

  @classmethod
  def Update(cls, params):
    """Updates anchor box settings from input configuration lists.

    Given dimensions priors, rotations, and offsets, computes the cartesian
    product of the settings.

    Args:
      params: The KITTIAnchorExtractorBase.Params() object to update.

    Returns:
      Params updated with the anchor settings.

      In total there are N combinations, where each (anchor_box_dimensions[i],
        anchor_box_rotations[i], anchor_box_offsets[i]) for i in range(N) is an
        option.
    """
    p = params
    settings = cls.GenerateAnchorSettings()
    p.anchor_box_dimensions = settings.anchor_box_dimensions
    p.anchor_box_rotations = settings.anchor_box_rotations
    p.anchor_box_offsets = settings.anchor_box_offsets
    return p


def MakeAnchorBoxSettings(dimension_priors, rotations, center_x_offsets,
                          center_y_offsets, center_z_offsets):
  """Returns a configured class for setting anchor box settings."""

  class CustomAnchorBoxSettings(_AnchorBoxSettings):
    DIMENSION_PRIORS = dimension_priors
    ROTATIONS = rotations
    CENTER_X_OFFSETS = center_x_offsets
    CENTER_Y_OFFSETS = center_y_offsets
    CENTER_Z_OFFSETS = center_z_offsets

  return CustomAnchorBoxSettings


class SparseCarV1AnchorBoxSettings(_AnchorBoxSettings):
  """Anchor box settings for training on Cars for Sparse models."""
  # Borrowed from PointPillar dimension prior for cars.
  DIMENSION_PRIORS = [(1.6, 3.9, 1.56)]

  # 4 Rotations with axis aligned and both diagonals.
  ROTATIONS = [0, np.pi / 2, np.pi / 4, 3 * np.pi / 4]

  # 25 offsets per anchor box with fixed z offset at -1.
  CENTER_X_OFFSETS = np.linspace(-1.5, 1.5, 5)
  CENTER_Y_OFFSETS = np.linspace(-1.5, 1.5, 5)
  CENTER_Z_OFFSETS = [-1.]


class PointPillarAnchorBoxSettingsCar(_AnchorBoxSettings):
  DIMENSION_PRIORS = [(1.6, 3.9, 1.56)]
  ROTATIONS = [0, np.pi / 2]
  # Fixed offset for every anchor box, based on a reading of the paper / code
  # 0 offsets for x and y, and -1 for z.
  CENTER_X_OFFSETS = [0.]
  CENTER_Y_OFFSETS = [0.]
  CENTER_Z_OFFSETS = [-1.]


class PointPillarAnchorBoxSettingsPed(PointPillarAnchorBoxSettingsCar):
  DIMENSION_PRIORS = [(0.6, 0.8, 1.73)]
  CENTER_Z_OFFSETS = [-0.6]


class PointPillarAnchorBoxSettingsCyc(PointPillarAnchorBoxSettingsCar):
  DIMENSION_PRIORS = [(0.6, 1.76, 1.73)]
  CENTER_Z_OFFSETS = [-0.6]


class PointPillarAnchorBoxSettingsPedCyc(PointPillarAnchorBoxSettingsCar):
  DIMENSION_PRIORS = [(0.6, 0.8, 1.7), (0.6, 1.76, 1.73)]
  CENTER_Z_OFFSETS = [-0.6]


class AnchorAssignment(Preprocessor):
  """Perform anchor assignment on the features.

  This preprocessor expects features to contain the following keys:
  - anchor_bboxes of shape [...base shape..., 7]
  - labels.bboxes_3d
  - labels.labels
  - labels.bboxes_3d_mask

  Adds the following features:

    anchor_localization_residuals: base_shape + [7] floating point tensor of
      residuals. The model is expected to regress against these residuals as
      targets. The residuals can be converted back into bboxes using
      detection_3d_lib.Utils3D.ResidualsToBBoxes.
    assigned_gt_idx: base_shape - The corresponding index of the ground
      truth bounding box for each anchor box in anchor_bboxes, anchors not
      assigned will have idx be set to -1.
    assigned_gt_bbox: base_shape + [7] - The corresponding ground
      truth bounding box for each anchor box in anchor_bboxes.
    assigned_gt_labels: base_shape - The assigned groundtruth label
      for each anchor box.
    assigned_gt_similarity_score: base_shape - The similarity score
      for each assigned anchor box.
    assigned_cls_mask: base_shape mask for classification loss per anchor.
      This should be 1.0 if the anchor has a foreground or background
      assignment; otherwise, it will be assigned to 0.0.
    assigned_reg_mask: base_shape mask for regression loss per anchor.
      This should be 1.0 if the anchor has a foreground assignment;
      otherwise, it will be assigned to 0.0.
      Note: background anchors do not have regression targets.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'foreground_assignment_threshold', 0.5,
        'Score (usually IOU) threshold for assigning a box as foreground.')
    p.Define(
        'background_assignment_threshold', 0.35,
        'Score (usually IOU) threshold for assigning a box as background.')
    return p

  def TransformFeatures(self, features):
    p = self.params
    utils_3d = detection_3d_lib.Utils3D()

    # anchor_bboxes will be returned with shape [#centers, #boxes_per_center, 7]
    # flatten boxes here for matching.
    base_shape = py_utils.GetShape(features.anchor_bboxes)[:-1]
    anchor_bboxes = tf.reshape(features.anchor_bboxes, [-1, 7])

    assigned_anchors = utils_3d.AssignAnchors(
        anchor_bboxes,
        features.labels.bboxes_3d,
        features.labels.labels,
        features.labels.bboxes_3d_mask,
        foreground_assignment_threshold=p.foreground_assignment_threshold,
        background_assignment_threshold=p.background_assignment_threshold)

    # Add new features.
    features.assigned_gt_idx = tf.reshape(assigned_anchors.assigned_gt_idx,
                                          base_shape)
    features.assigned_gt_bbox = tf.reshape(assigned_anchors.assigned_gt_bbox,
                                           base_shape + [7])
    features.assigned_gt_labels = tf.reshape(
        assigned_anchors.assigned_gt_labels, base_shape)
    features.assigned_gt_similarity_score = tf.reshape(
        assigned_anchors.assigned_gt_similarity_score, base_shape)
    features.assigned_cls_mask = tf.reshape(assigned_anchors.assigned_cls_mask,
                                            base_shape)
    features.assigned_reg_mask = tf.reshape(assigned_anchors.assigned_reg_mask,
                                            base_shape)

    # Compute residuals.
    features.anchor_localization_residuals = utils_3d.LocalizationResiduals(
        features.anchor_bboxes, features.assigned_gt_bbox)

    return features

  def TransformShapes(self, shapes):
    base_shape = shapes.anchor_bboxes[:-1]
    box_shape = base_shape.concatenate([7])

    shapes.anchor_localization_residuals = box_shape
    shapes.assigned_gt_idx = base_shape
    shapes.assigned_gt_bbox = box_shape
    shapes.assigned_gt_labels = base_shape
    shapes.assigned_gt_similarity_score = base_shape
    shapes.assigned_cls_mask = base_shape
    shapes.assigned_reg_mask = base_shape
    return shapes

  def TransformDTypes(self, dtypes):
    dtypes.anchor_localization_residuals = tf.float32
    dtypes.assigned_gt_idx = tf.int32
    dtypes.assigned_gt_bbox = tf.float32
    dtypes.assigned_gt_labels = tf.int32
    dtypes.assigned_gt_similarity_score = tf.float32
    dtypes.assigned_cls_mask = tf.float32
    dtypes.assigned_reg_mask = tf.float32
    return dtypes


class DropLaserPointsOutOfRange(Preprocessor):
  """Drops laser points that are out of pre-defined x/y/z ranges.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]
  - lasers.points_feature of shape [P, F]

  and optionally points_padding of shape [P] corresponding to the padding.
  if points_padding is None, then all points are considered valid.

  Modifies the following features:
    Removes or sets padding to 1 for all points outside a given range. Modifies
    all items in the lasers subdictionary like lasers.points_xyz,
    lasers.points_feature, lasers.points_padding, and optionally
    lasers.points_label, lasers.points_bbox_id.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('keep_x_range', (-np.inf, np.inf),
             'Only points that have x coordinates within this range are kept.')
    p.Define('keep_y_range', (-np.inf, np.inf),
             'Only points that have y coordinates within this range are kept.')
    p.Define(
        'keep_z_range', (-np.inf, np.inf),
        'Only points that have z coordinates within this range are kept. '
        'Approximate ground-removal can be performed by specifying a '
        'lower-bound on the z-range.')
    return p

  def TransformFeatures(self, features):
    p = self.params

    points_xyz = features.lasers.points_xyz
    if 'points_padding' in features.lasers:
      points_mask = tf.cast(1 - features.lasers.points_padding, tf.bool)
    else:
      # All points are real, we keep points unpadded by applying boolean_mask
      # on points_mask later.
      points_mask = tf.ones_like(points_xyz[:, 0], dtype=tf.bool)

    min_x, max_x = p.keep_x_range
    min_y, max_y = p.keep_y_range
    min_z, max_z = p.keep_z_range

    # Short-circuit if all ranges are set to -inf, inf.
    if (np.all(np.isneginf([min_x, min_y, min_z])) and
        np.all(np.isposinf([max_x, max_y, max_z]))):
      return features

    if min_x != -np.inf:
      points_mask &= points_xyz[:, 0] >= min_x
    if min_y != -np.inf:
      points_mask &= points_xyz[:, 1] >= min_y
    if min_z != -np.inf:
      points_mask &= points_xyz[:, 2] >= min_z

    if max_x != np.inf:
      points_mask &= points_xyz[:, 0] <= max_x
    if max_y != np.inf:
      points_mask &= points_xyz[:, 1] <= max_y
    if max_z != np.inf:
      points_mask &= points_xyz[:, 2] <= max_z

    if 'points_padding' in features.lasers:
      # Suffices to just update the padding.
      features.lasers.points_padding = 1. - tf.cast(points_mask, tf.float32)
    else:
      features.lasers = features.lasers.Transform(
          _GetApplyPointMaskFn(points_mask))

    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class KITTIDropPointsOutOfFrustum(Preprocessor):
  """Drops laser points that are outside of the camera frustum.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]
  - lasers.points_feature of shape [P, F]
  - images.velo_to_image_plane of shape [3, 4]
  - images.width of shape [1]
  - images.height of shape [1]

  and optionally points_padding of shape [P] corresponding to the padding.
  if points_padding is None, then all points are considered valid.

  Modifies the following features:
    lasers.points_xyz, lasers.points_feature, lasers.points_padding, and
    optionally lasers.points_label, lasers.points_bbox_id so that
    points outside the frustum have padding set to 1 or are removed.
  """

  def TransformFeatures(self, features):
    # Drop points behind the car (behind x-axis = 0).
    images = features.images
    front_indices = features.lasers.points_xyz[:, 0] >= 0

    if 'points_padding' not in features.lasers:
      # Keep tensors unpadded and small using boolean_mask.
      features.lasers.points_xyz = tf.boolean_mask(features.lasers.points_xyz,
                                                   front_indices)
      features.lasers.points_feature = tf.boolean_mask(
          features.lasers.points_feature, front_indices)

    # Drop those points outside the image plane.
    points_image = geometry.PointsToImagePlane(features.lasers.points_xyz,
                                               images.velo_to_image_plane)
    in_image_plane = (
        (points_image[:, 0] >= 0) &
        (points_image[:, 0] <= tf.cast(images.width, tf.float32)) &
        (points_image[:, 1] >= 0) &
        (points_image[:, 1] <= tf.cast(images.height, tf.float32)))

    if 'points_padding' in features.lasers:
      # Update padding to only include front indices and in image plane.
      points_mask = tf.cast(1 - features.lasers.points_padding, tf.bool)
      points_mask &= front_indices
      points_mask &= in_image_plane
      features.lasers.points_padding = 1. - tf.cast(points_mask, tf.float32)
    else:
      features.lasers = features.lasers.Transform(
          _GetApplyPointMaskFn(in_image_plane))
    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class RandomWorldRotationAboutZAxis(Preprocessor):
  """Rotates the world randomly as a form of data augmentation.

  Rotations are performed around the *z-axis*. This assumes that the car is
  always level. In general, we'd like to instead rotate the car on the spot,
  this would then make sense for cases where the car is on a slope.

  When there are leading dimensions, this will rotate the boxes with the same
  transformation across all the frames. This is useful when the input is a
  sequence of frames from the same run segment.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [..., 3]
  - labels.bboxes_3d of shape [..., 7]

  Modifies the following features:
    lasers.points_xyz, labels.bboxes_3d with the same rotation applied to both.

  Adds the following features:
    world_rot_z which contains the rotation applied to the example.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'max_rotation', None,
        'The rotation amount will be randomly picked from '
        '[-max_rotation, max_rotation).')
    p.Define(
        'include_world_rot_z', True,
        'Whether to include the applied rotation as an additional tensor. '
        'It can be helpful to disable this when using the preprocessor in a '
        'way that expects the structure of the features to be the same '
        '(e.g., as a branch in tf.cond).')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.max_rotation is None:
      raise ValueError('max_rotation needs to be specified, instead of None.')

  def TransformFeatures(self, features):
    p = self.params
    rot = tf.random.uniform((),
                            minval=-p.max_rotation,
                            maxval=p.max_rotation,
                            seed=p.random_seed)

    # Rotating about the z-axis is equal to experiencing yaw.
    pose = [0., 0., 0., rot, 0., 0.]

    # Rotate points.
    features.lasers.points_xyz = geometry.CoordinateTransform(
        features.lasers.points_xyz, pose)

    # Rotate bboxes, note that heading has a special case.
    bboxes_xyz = features.labels.bboxes_3d[..., :3]
    bboxes_dims = features.labels.bboxes_3d[..., 3:6]
    bboxes_rot = features.labels.bboxes_3d[..., 6:]

    bboxes_xyz = geometry.CoordinateTransform(bboxes_xyz, pose)

    # The heading correction should subtract rot from the bboxes rotations.
    bboxes_rot = geometry.WrapAngleRad(bboxes_rot - rot)

    features.labels.bboxes_3d = tf.concat([bboxes_xyz, bboxes_dims, bboxes_rot],
                                          axis=-1)
    if p.include_world_rot_z:
      features.world_rot_z = rot
    return features

  def TransformShapes(self, shapes):
    if self.params.include_world_rot_z:
      shapes.world_rot_z = tf.TensorShape([])
    return shapes

  def TransformDTypes(self, dtypes):
    if self.params.include_world_rot_z:
      dtypes.world_rot_z = tf.float32
    return dtypes


class DropPointsOutOfFrustum(Preprocessor):
  """Drops points outside of pre-defined theta / phi ranges.

  Note that the ranges for keep_phi_range can be negative, this is because the
  phi values wrap around 2*pi. Thus, a valid range that filters the 90 deg
  frontal field of view of the car can be specified as [-pi/4, pi/4].

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]
  - lasers.points_feature of shape [P, F]

  Modifies the following features:
  - lasers.points_xyz removing any points out of frustum.
  - lasers.points_feature removing any points out of frustum.

  Note: We expect a downstream processor that filters out boxes with few points
  to drop the corresponding bboxes.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('keep_theta_range', (0., np.pi),
             'Only points that have theta coordinates within this range.')
    p.Define('keep_phi_range', (0., 2. * np.pi),
             'Only points that have phi coordinates within this range.')
    return p

  def TransformFeatures(self, features):
    p = self.params

    if 'points_padding' in features.lasers:
      raise ValueError('DropPointsOutOfFrustum preprocessor does not support '
                       'padded lasers.')

    points_xyz = features.lasers.points_xyz
    points_feature = features.lasers.points_feature

    min_theta, max_theta = p.keep_theta_range
    if (min_theta < 0. or min_theta > np.pi or max_theta < 0. or
        max_theta > np.pi):
      raise ValueError('Valid values for theta are between 0 and pi, '
                       'keep_theta_range={}'.format(p.keep_theta_range))

    if min_theta > max_theta:
      raise ValueError('min_theta must be <= max_theta, '
                       'keep_theta_range={}'.format(p.keep_theta_range))

    min_phi, max_phi = p.keep_phi_range
    if (min_phi < -2. * np.pi or min_phi > 2. * np.pi or
        max_phi < -2. * np.pi or max_phi > 2. * np.pi):
      raise ValueError('Valid values for phi are between -2*pi and 2*pi,'
                       'keep_phi_range={}'.format(p.keep_phi_range))

    if min_phi > max_phi:
      raise ValueError('min_phi must be <= max_phi, '
                       'keep_phi_range={}'.format(p.keep_phi_range))

    _, theta, phi = tf.unstack(
        geometry.SphericalCoordinatesTransform(points_xyz), axis=-1)

    # phi is returned in range [-pi, pi], we shift the values which are between
    # [-pi, 0] to be [pi, 2pi] instead to make the logic below easier to follow.
    # Hence, all phi values after this will be [0, 2pi].
    phi = tf.where(phi >= 0., phi, 2. * np.pi + phi)

    # Theta does not have circular boundary conditions, a simple check suffices.
    points_mask = (theta >= min_theta) & (theta <= max_theta)

    if min_phi < 0. and max_phi < 0.:
      # Both are less than zero, we just just add 2pi and will use the regular
      # check.
      min_phi += 2. * np.pi
      max_phi += 2. * np.pi

    if min_phi < 0.:
      # The minimum threshold is below 0, so we split into checking between
      # (0 to min_phi) and (0 to max_phi). Note that min_phi is negative, but
      # phi is always positive, so we take 2*pi + min_phi to get the range of
      # appropriate values.
      points_mask &= (phi >= (2. * np.pi + min_phi)) | (phi <= max_phi)
    else:
      # Both must be greater than 0 if we get to this condition.
      assert min_phi >= 0.
      assert max_phi >= 0.
      points_mask &= (phi >= min_phi) & (phi <= max_phi)

    features.lasers.points_xyz = tf.boolean_mask(points_xyz, points_mask)
    features.lasers.points_feature = tf.boolean_mask(points_feature,
                                                     points_mask)
    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class DropBoxesOutOfRange(Preprocessor):
  """Drops boxes outside of pre-defined x/y/z ranges (boundaries inclusive).

  This preprocessor expects features to contain the following keys:
  - labels.bboxes_3d of shape [N, 7]
  - labels.bboxes_3d_mask of shape [N]

  Modifies the following features:
  - labels.bboxes_3d_mask to mask out any additional boxes.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('keep_x_range', (-np.inf, np.inf),
             'Only boxes that have x coordinates within this range are kept.')
    p.Define('keep_y_range', (-np.inf, np.inf),
             'Only boxes that have y coordinates within this range are kept.')
    p.Define('keep_z_range', (-np.inf, np.inf),
             'Only boxes that have z coordinates within this range are kept.')
    return p

  def TransformFeatures(self, features):
    p = self.params

    min_x, max_x = p.keep_x_range
    min_y, max_y = p.keep_y_range
    min_z, max_z = p.keep_z_range

    # Short-circuit if all ranges are set to -inf, inf.
    if (np.all(np.isneginf([min_x, min_y, min_z])) and
        np.all(np.isposinf([max_x, max_y, max_z]))):
      return features

    # For each bounding box, compute whether any of its extrema
    # fall outside of the range.
    bboxes_3d_corners = geometry.BBoxCorners(
        features.labels.bboxes_3d[tf.newaxis, ...])[0]
    bboxes_3d_corners = py_utils.HasShape(bboxes_3d_corners, [-1, 8, 3])

    min_bbox_x = tf.reduce_min(bboxes_3d_corners[:, :, 0], axis=-1)
    max_bbox_x = tf.reduce_max(bboxes_3d_corners[:, :, 0], axis=-1)

    min_bbox_y = tf.reduce_min(bboxes_3d_corners[:, :, 1], axis=-1)
    max_bbox_y = tf.reduce_max(bboxes_3d_corners[:, :, 1], axis=-1)

    min_bbox_z = tf.reduce_min(bboxes_3d_corners[:, :, 2], axis=-1)
    max_bbox_z = tf.reduce_max(bboxes_3d_corners[:, :, 2], axis=-1)

    mask = (
        tf.math.logical_and(min_bbox_x >= min_x, max_bbox_x <= max_x)
        & tf.math.logical_and(min_bbox_y >= min_y, max_bbox_y <= max_y)
        & tf.math.logical_and(min_bbox_z >= min_z, max_bbox_z <= max_z))

    max_num_boxes = py_utils.GetShape(features.labels.bboxes_3d_mask)
    mask = py_utils.HasShape(mask, max_num_boxes)

    features.labels.bboxes_3d_mask *= tf.cast(mask, tf.float32)
    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class PadLaserFeatures(Preprocessor):
  """Pads laser features so that the dimensions are fixed.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]
  - lasers.points_feature of shape [P, F]

  and optionally points_padding of shape [P] corresponding to the padding.
  if points_padding is None, then all points are considered valid.

  Modifies the following features:
    lasers.points_xyz and lasers.points_feature to add padding.
    Optionally also modifies lasers.points_label and lasers.points_bbox_id
    if they exist to add padding.
  Modifies/adds the following features:
    labels.points_padding of shape [P] representing the padding.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('max_num_points', 128500,
             'Max number of points to pad the points to.')
    return p

  def TransformFeatures(self, features):
    p = self.params

    if 'points_padding' in features.lasers:
      points_mask = 1 - features.lasers.points_padding
      points_mask = tf.cast(points_mask, tf.bool)
      features.lasers = features.lasers.Transform(
          _GetApplyPointMaskFn(points_mask))

    npoints = tf.shape(features.lasers.points_xyz)[0]
    features.lasers.points_padding = tf.ones([npoints])

    shuffled_idx = tf.range(npoints)
    shuffled_idx = tf.random.shuffle(shuffled_idx, seed=p.random_seed)

    def _PadOrTrimFn(points_tensor):
      # Shuffle before trimming so we have a random sampling
      points_tensor = tf.gather(points_tensor, shuffled_idx)
      return py_utils.PadOrTrimTo(points_tensor, [p.max_num_points] +
                                  points_tensor.shape[1:].as_list())

    features.lasers = features.lasers.Transform(_PadOrTrimFn)
    features.lasers.points_padding = 1.0 - features.lasers.points_padding
    return features

  def TransformShapes(self, shapes):
    p = self.params

    def _TransformShape(points_shape):
      return tf.TensorShape([p.max_num_points] + points_shape[1:].as_list())

    shapes.lasers = shapes.lasers.Transform(_TransformShape)
    shapes.lasers.points_padding = tf.TensorShape([p.max_num_points])
    return shapes

  def TransformDTypes(self, dtypes):
    dtypes.lasers.points_padding = tf.float32
    return dtypes


class WorldScaling(Preprocessor):
  """Scale the world randomly as a form of data augmentation.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]
  - labels.bboxes_3d of shape [L, 7]

  Modifies the following features:
    lasers.points_xyz, labels.bboxes_3d with the same scaling applied to both.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('scaling', None, 'The scaling range.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.scaling is None:
      raise ValueError('scaling needs to be specified, instead of None.')
    if len(p.scaling) != 2:
      raise ValueError('scaling needs to be a list of two elements.')

  def TransformFeatures(self, features):
    p = self.params
    scaling = tf.random.uniform((),
                                minval=p.scaling[0],
                                maxval=p.scaling[1],
                                seed=p.random_seed,
                                dtype=features.lasers.points_xyz.dtype)

    # Scale points [num_points, 3].
    features.lasers.points_xyz *= scaling

    # Scaling bboxes (location and dimensions).
    bboxes_xyz = features.labels.bboxes_3d[..., :3] * scaling
    bboxes_dims = features.labels.bboxes_3d[..., 3:6] * scaling
    bboxes_rot = features.labels.bboxes_3d[..., 6:]

    features.labels.bboxes_3d = tf.concat([bboxes_xyz, bboxes_dims, bboxes_rot],
                                          axis=-1)
    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class RandomDropLaserPoints(Preprocessor):
  """Randomly dropout laser points and the corresponding features.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]
  - lasers.points_feature of shape [P, F]


  Modifies the following features:
    lasers.points_xyz, lasers.points_feature.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('keep_prob', 0.95, 'Probability for keeping points.')
    return p

  def TransformFeatures(self, features):
    p = self.params
    num_points, _ = py_utils.GetShape(features.lasers.points_xyz)

    pts_keep_sample_prob = tf.random.uniform([num_points],
                                             minval=0,
                                             maxval=1,
                                             seed=p.random_seed)
    pts_keep_mask = pts_keep_sample_prob < p.keep_prob

    if 'points_padding' in features.lasers:
      # Update points_padding so that where pts_keep_mask is True,
      # points_padding remains 0.
      points_mask = 1 - features.lasers.points_padding
      points_mask *= tf.cast(pts_keep_mask, tf.float32)
      features.lasers.points_padding = 1 - points_mask
    else:
      features.lasers.points_xyz = tf.boolean_mask(features.lasers.points_xyz,
                                                   pts_keep_mask)
      features.lasers.points_feature = tf.boolean_mask(
          features.lasers.points_feature, pts_keep_mask)
    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class RandomFlipY(Preprocessor):
  """Flip the world along axis Y as a form of data augmentation.

  When there are leading dimensions, this will flip the boxes with the same
  transformation across all the frames. This is useful when the input is a
  sequence of frames from the same run segment.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [..., 3]
  - labels.bboxes_3d of shape [..., 7]

  Modifies the following features:
    lasers.points_xyz, labels.bboxes_3d with the same flipping applied to both.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('flip_probability', 0.5, 'Probability of flipping.')
    return p

  def TransformFeatures(self, features):
    p = self.params
    threshold = 1. - p.flip_probability
    choice = tf.random.uniform(
        (), minval=0.0, maxval=1.0, seed=p.random_seed) >= threshold

    # Flip points
    points_xyz = features.lasers.points_xyz
    points_y = tf.where(choice, -points_xyz[..., 1:2], points_xyz[..., 1:2])
    features.lasers.points_xyz = tf.concat(
        [points_xyz[..., 0:1], points_y, points_xyz[..., 2:3]], axis=-1)

    # Flip boxes
    bboxes_xyz = features.labels.bboxes_3d[..., :3]
    bboxes_y = tf.where(choice, -bboxes_xyz[..., 1:2], bboxes_xyz[..., 1:2])
    bboxes_xyz = tf.concat(
        [bboxes_xyz[..., 0:1], bboxes_y, bboxes_xyz[..., 2:3]], axis=-1)
    # Compensate rotation.
    bboxes_dims = features.labels.bboxes_3d[..., 3:6]
    bboxes_rot = features.labels.bboxes_3d[..., 6:]
    bboxes_rot = tf.where(choice, geometry.WrapAngleRad(-bboxes_rot),
                          bboxes_rot)
    features.labels.bboxes_3d = tf.concat([bboxes_xyz, bboxes_dims, bboxes_rot],
                                          axis=-1)
    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class GlobalTranslateNoise(Preprocessor):
  """Add global translation noise of xyz coordinates to points and boxes.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]
  - labels.bboxes_3d of shape [L, 7]

  Modifies the following features:
    lasers.points_xyz, labels.bboxes_3d with the same
      random translation noise applied to both.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('noise_std', [0.2, 0.2, 0.2],
             'Standard deviation of translation noise per axis.')
    return p

  def TransformFeatures(self, features):
    p = self.params
    # Use three different seeds but the same base seed so
    # that the values are different.
    base_seed = p.random_seed
    x_seed = base_seed
    y_seed = None if base_seed is None else base_seed + 1
    z_seed = None if base_seed is None else base_seed + 2
    random_translate_x = tf.random.normal((),
                                          mean=0.0,
                                          stddev=p.noise_std[0],
                                          seed=x_seed)
    random_translate_y = tf.random.normal((),
                                          mean=0.0,
                                          stddev=p.noise_std[1],
                                          seed=y_seed)
    random_translate_z = tf.random.normal((),
                                          mean=0.0,
                                          stddev=p.noise_std[2],
                                          seed=z_seed)

    pose = tf.stack([
        random_translate_x, random_translate_y, random_translate_z, 0.0, 0.0,
        0.0
    ],
                    axis=0)

    # Translate points.
    points_xyz = features.lasers.points_xyz
    features.lasers.points_xyz = geometry.CoordinateTransform(points_xyz, pose)

    # Translate boxes
    bboxes_xyz = features.labels.bboxes_3d[..., :3]
    bboxes_xyz = geometry.CoordinateTransform(bboxes_xyz, pose)
    features.labels.bboxes_3d = tf.concat(
        [bboxes_xyz, features.labels.bboxes_3d[..., 3:]], axis=-1)
    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class RandomBBoxTransform(Preprocessor):
  """Randomly transform bounding boxes and the points inside them.

  This preprocessor expects features to contain the following keys:
    - lasers.points_xyz of shape [P, 3]
    - lasers.points_feature of shape [P, F]
    - lasers.points_padding of shape [P]
    - labels.bboxes_3d of shape [L, 7]
    - labels.bboxes_3d_mask of shape [L]

  Modifies the following features:
    lasers.points_{xyz,feature,padding}, labels.bboxes_3d with the
      transformed bounding boxes and points.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'max_rotation', None,
        'The rotation amount will be randomly picked from '
        '[-max_rotation, max_rotation).')
    # At the moment we don't use this because it can cause boxes to collide with
    # each other.  We need to compute box intersections when deciding whether to
    # apply the translation jitter.  Theoretically we should also do this for
    # rotation.
    p.Define('noise_std', [0.0, 0.0, 0.0],
             'Standard deviation of translation noise per axis.')
    p.Define(
        'max_scaling', None,
        'An optional float list of length 3. When max_scaling is not none, '
        'delta parameters s_x, s_y, s_z are drawn from '
        '[-max_scaling[i], max_scaling[i]] where i is in [0, 2].')
    p.Define(
        'max_shearing', None,
        'An optional float list of length 6. When max_shearing is not none, '
        'shearing parameters sh_x^y, sh_x^z, sh_y^x, sh_y^z, sh_z^x, sh_z^y are'
        'drawn from [-max_shearing[i], max_shearing[i]], where i is in [0, 5].')
    p.Define(
        'max_num_points_per_bbox', 16384,
        'The maximum number of points that fall within a bounding box. '
        'Bounding boxes with more points than this value will '
        'have some points droppped.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.max_rotation is None:
      raise ValueError('max_rotation needs to be specified, instead of None.')
    if p.max_scaling is not None:
      if len(p.max_scaling) != 3:
        raise ValueError('max_scaling needs to be specified as either None or '
                         'list of 3 floating point numbers, instead of {}.'
                         ''.format(p.max_scaling))
    if p.max_shearing is not None:
      if len(p.max_shearing) != 6:
        raise ValueError('max_shearing needs to be specified as either None or '
                         'list of 6 floating point numbers, instead of {}.'
                         ''.format(p.max_shearing))

  def _Foreground(self, features, points_xyz, points_feature, real_bboxes_3d,
                  points_in_bbox_mask, rotation, translate_pose, transform_fn):
    """Extract and transform foreground points and features."""
    out_bbox_xyz, out_bbox_feature, out_bbox_mask = self._ForLoopBuffers(
        features)

    # Only iterate over the actual number of boxes in the scene.
    actual_num_bboxes = tf.reduce_sum(
        tf.cast(features.labels.bboxes_3d_mask, tf.int32))

    ret = py_utils.ForLoop(
        body=transform_fn,
        start=0,
        limit=actual_num_bboxes,
        delta=1,
        loop_state=py_utils.NestedMap(
            points_xyz=points_xyz,
            points_feature=points_feature,
            bboxes_3d=real_bboxes_3d,
            points_in_bbox_mask=points_in_bbox_mask,
            rotation=rotation,
            translate_pose=translate_pose,
            out_bbox_points=out_bbox_xyz,
            out_bbox_feature=out_bbox_feature,
            out_bbox_mask=out_bbox_mask))

    # Gather all of the transformed points and features
    out_bbox_xyz = tf.reshape(ret.out_bbox_points, [-1, 3])
    num_features = features.lasers.points_feature.shape[-1]
    out_bbox_feature = tf.reshape(ret.out_bbox_feature, [-1, num_features])
    out_bbox_mask = tf.cast(tf.reshape(ret.out_bbox_mask, [-1]), tf.bool)
    fg_xyz = tf.boolean_mask(out_bbox_xyz, out_bbox_mask)
    fg_feature = tf.boolean_mask(out_bbox_feature, out_bbox_mask)
    return fg_xyz, fg_feature

  def _Background(self, points_xyz, points_feature, points_in_bbox_mask):
    # If a point is in any bounding box, it is a foreground point.
    foreground_points_mask = tf.reduce_any(points_in_bbox_mask, axis=-1)
    # All others are background.  We rotate all of the foreground points to
    # final_points_* and keep the background points unchanged
    background_points_mask = tf.math.logical_not(foreground_points_mask)
    background_points_xyz = tf.boolean_mask(points_xyz, background_points_mask)
    background_points_feature = tf.boolean_mask(points_feature,
                                                background_points_mask)
    return background_points_xyz, background_points_feature

  def _ForLoopBuffers(self, features):
    """Create and return the buffers for the for loop."""
    p = self.params
    bboxes_3d = features.labels.bboxes_3d

    # Compute the shapes and create the buffers for the For loop.
    max_num_bboxes = tf.shape(bboxes_3d)[0]
    per_box_shape = [max_num_bboxes, p.max_num_points_per_bbox, 3]
    out_bbox_points = inplace_ops.empty(
        per_box_shape, dtype=tf.float32, init=True)

    num_features = features.lasers.points_feature.shape[-1]
    bbox_feature_shape = [
        max_num_bboxes, p.max_num_points_per_bbox, num_features
    ]
    out_bbox_feature = inplace_ops.empty(
        bbox_feature_shape, dtype=tf.float32, init=True)

    per_box_mask_shape = [max_num_bboxes, p.max_num_points_per_bbox]
    out_bbox_mask = inplace_ops.empty(
        per_box_mask_shape, dtype=tf.float32, init=True)

    return out_bbox_points, out_bbox_feature, out_bbox_mask

  def TransformFeatures(self, features):
    p = self.params

    num_features = features.lasers.points_feature.shape[-1]

    def Transform(i, state):
      """Transform the points in bounding box `i`."""
      state.points_xyz = tf.reshape(state.points_xyz, [-1, 3])
      bbox_mask = tf.reshape(state.points_in_bbox_mask[:, i], [-1])

      # Fetch only the points in the bounding box.
      points_xyz_masked = tf.boolean_mask(state.points_xyz, bbox_mask)
      points_feature_masked = tf.boolean_mask(state.points_feature, bbox_mask)

      num_points = tf.shape(points_xyz_masked)[0]

      # TODO(vrv): Fold the following into a single transformation
      # matrix.
      #
      # Translate the box to the origin, then rotate the desired
      # rotation angle.
      translation_vec = state.bboxes_3d[i, 0:3]
      rotation_vec = [state.rotation[i], 0., 0.]
      pose = tf.concat([-translation_vec, rotation_vec], axis=0)
      points_xyz_adj = geometry.CoordinateTransform(points_xyz_masked, pose)
      if p.max_scaling is not None or p.max_shearing is not None:
        # Translate the points in the bounding box by moving dz/2 so that the
        # bottom of the bounding box is at Z = 0 when any of the two
        # (max_scaling or max_shearing) is not None
        translation_scale_or_shear = tf.stack(
            [0., 0., state.bboxes_3d[i, 5] / 2], axis=0)
        pose1 = tf.concat([translation_scale_or_shear, [0., 0., 0.]], axis=0)
        points_xyz_adj = geometry.CoordinateTransform(points_xyz_adj, pose1)
      else:
        translation_scale_or_shear = tf.stack([0., 0., 0.], axis=0)

      if p.max_scaling is not None:
        # Perform scaling to the point cloud
        # Scaling matrix
        # [[s_x+1    0      0]
        #  [ 0      s_y+1   0]
        #  [ 0       0     s_z+1]]
        sx = tf.random.uniform([],
                               minval=-p.max_scaling[0],
                               maxval=p.max_scaling[0],
                               seed=p.random_seed)
        sy = tf.random.uniform([],
                               minval=-p.max_scaling[1],
                               maxval=p.max_scaling[1],
                               seed=p.random_seed)
        sz = tf.random.uniform([],
                               minval=-p.max_scaling[2],
                               maxval=p.max_scaling[2],
                               seed=p.random_seed)
        scaling_matrix = tf.stack(
            [[sx + 1., 0., 0.], [0., sy + 1., 0.], [0., 0., sz + 1.]], axis=0)

        points_xyz_adj = tf.einsum('ij,kj->ki', scaling_matrix, points_xyz_adj)

      if p.max_shearing is not None:
        # Perform shearing to the point cloud
        # Shearing matrix
        # [[1       sh_x^y  sh_x^z]
        #  [sh_y^x     1    sh_y^z]
        #  [sh_z^x  sh_z^y     1  ]]
        sxy = tf.random.uniform([],
                                minval=-p.max_shearing[0],
                                maxval=p.max_shearing[0],
                                seed=p.random_seed)
        sxz = tf.random.uniform([],
                                minval=-p.max_shearing[1],
                                maxval=p.max_shearing[1],
                                seed=p.random_seed)
        syx = tf.random.uniform([],
                                minval=-p.max_shearing[2],
                                maxval=p.max_shearing[2],
                                seed=p.random_seed)
        syz = tf.random.uniform([],
                                minval=-p.max_shearing[3],
                                maxval=p.max_shearing[3],
                                seed=p.random_seed)
        szx = tf.random.uniform([],
                                minval=-p.max_shearing[4],
                                maxval=p.max_shearing[4],
                                seed=p.random_seed)
        szy = tf.random.uniform([],
                                minval=-p.max_shearing[5],
                                maxval=p.max_shearing[5],
                                seed=p.random_seed)
        shearing_matrix = tf.stack(
            [[1., sxy, sxz], [syx, 1., syz], [szx, szy, 1.]], axis=0)
        points_xyz_adj = tf.einsum('ij,kj->ki', shearing_matrix, points_xyz_adj)

      # Translate the points back, adding noise if needed.
      translation_with_noise = (
          translation_vec - translation_scale_or_shear +
          state.translate_pose[i])
      pose2 = tf.concat([translation_with_noise, [0., 0., 0.]], axis=0)
      final_points_xyz = geometry.CoordinateTransform(points_xyz_adj, pose2)

      # final_points_xyz is an [M, 3] Tensor where M is the number of points in
      # the box.
      points_mask = tf.ones([num_points], dtype=tf.float32)

      final_points_xyz = py_utils.PadOrTrimTo(final_points_xyz,
                                              [p.max_num_points_per_bbox, 3])
      final_points_feature = py_utils.PadOrTrimTo(
          points_feature_masked, [p.max_num_points_per_bbox, num_features])
      points_mask = py_utils.PadOrTrimTo(points_mask,
                                         [p.max_num_points_per_bbox])
      state.out_bbox_points = inplace_ops.alias_inplace_update(
          state.out_bbox_points, [i], tf.expand_dims(final_points_xyz, 0))
      state.out_bbox_feature = inplace_ops.alias_inplace_update(
          state.out_bbox_feature, [i], tf.expand_dims(final_points_feature, 0))
      state.out_bbox_mask = inplace_ops.alias_inplace_update(
          state.out_bbox_mask, [i], tf.expand_dims(points_mask, 0))

      return state

    # Get the points and features that reside in boxes.
    if 'points_padding' in features.lasers:
      points_mask = 1 - features.lasers.points_padding
      points_xyz = tf.boolean_mask(features.lasers.points_xyz, points_mask)
      points_feature = tf.boolean_mask(features.lasers.points_feature,
                                       points_mask)
    else:
      points_xyz = features.lasers.points_xyz
      points_feature = features.lasers.points_feature

    # Fetch real bounding boxes and compute point mask.
    real_bboxes_3d = tf.boolean_mask(features.labels.bboxes_3d,
                                     features.labels.bboxes_3d_mask)
    points_in_bbox_mask = geometry.IsWithinBBox3D(points_xyz, real_bboxes_3d)

    # Choose a random rotation for every real box.
    num_boxes = tf.shape(real_bboxes_3d)[0]
    rotation = tf.random.uniform([num_boxes],
                                 minval=-p.max_rotation,
                                 maxval=p.max_rotation,
                                 seed=p.random_seed)

    base_seed = p.random_seed
    x_seed = base_seed
    y_seed = None if base_seed is None else base_seed + 1
    z_seed = None if base_seed is None else base_seed + 2
    random_translate_x = tf.random.normal([num_boxes],
                                          mean=0.0,
                                          stddev=p.noise_std[0],
                                          seed=x_seed)
    random_translate_y = tf.random.normal([num_boxes],
                                          mean=0.0,
                                          stddev=p.noise_std[1],
                                          seed=y_seed)
    random_translate_z = tf.random.normal([num_boxes],
                                          mean=0.0,
                                          stddev=p.noise_std[2],
                                          seed=z_seed)

    translate_pose = tf.stack(
        [random_translate_x, random_translate_y, random_translate_z], axis=1)

    fg_xyz, fg_feature = self._Foreground(features, points_xyz, points_feature,
                                          real_bboxes_3d, points_in_bbox_mask,
                                          rotation, translate_pose, Transform)

    # Concatenate them with the background points and features.
    bg_xyz, bg_feature = self._Background(points_xyz, points_feature,
                                          points_in_bbox_mask)
    all_points = tf.concat([bg_xyz, fg_xyz], axis=0)
    all_features = tf.concat([bg_feature, fg_feature], axis=0)

    # Shuffle the points/features randomly.
    all_points, all_features = _ConsistentShuffle((all_points, all_features),
                                                  p.random_seed)

    # Padding should technically be unnecessary: the number of points before and
    # after should be the same, but in practice we sometimes seem to drop a few
    # points, and so we pad to make the shape fixed.
    #
    # TODO(vrv): Identify the source of this problem and then assert a shape
    # matching check.
    if 'points_padding' in features.lasers:
      features.lasers.points_xyz = py_utils.PadOrTrimTo(
          all_points, tf.shape(features.lasers.points_xyz))
      features.lasers.points_feature = py_utils.PadOrTrimTo(
          all_features, tf.shape(features.lasers.points_feature))
      total_points = tf.shape(all_points)[0]
      features.lasers.points_padding = 1.0 - py_utils.PadOrTrimTo(
          tf.ones([total_points]), tf.shape(features.lasers.points_padding))
    else:
      features.lasers.points_xyz = all_points
      features.lasers.points_feature = all_features

    # Translate noise.
    bboxes_xyz = real_bboxes_3d[..., :3]
    bboxes_xyz += translate_pose[..., :3]

    bboxes_dim = real_bboxes_3d[..., 3:6]
    # Rotate bboxes by their corresponding rotation.
    bboxes_rot = real_bboxes_3d[..., 6:]
    bboxes_rot -= rotation[:, tf.newaxis]
    features.labels.bboxes_3d = py_utils.PadOrTrimTo(
        tf.concat([bboxes_xyz, bboxes_dim, bboxes_rot], axis=-1),
        tf.shape(features.labels.bboxes_3d))
    features.labels.bboxes_3d_mask = py_utils.PadOrTrimTo(
        tf.ones(tf.shape(real_bboxes_3d)[0]),
        tf.shape(features.labels.bboxes_3d_mask))
    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class GroundTruthAugmentor(Preprocessor):
  """Augment bounding box labels and points from a database.

  This preprocessor expects features to contain the following keys:
    lasers.points_xyz of shape [P, 3]

    lasers.points_feature of shape [P, F]

    lasers.points_padding of shape [P]

    labels.bboxes_3d of shape [L, 7]

    labels.bboxes_3d_mask of shape [L]

    labels.labels of shape [L]

  Modifies the above features so that additional objects from
  a groundtruth database are added.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'groundtruth_database', None,
        'If not None, loads groundtruths from this database and adds '
        'them to the current scene. Groundtruth database is expected '
        'to be a TFRecord of KITTI or Waymo crops.')
    p.Define(
        'num_db_objects', None,
        'Number of objects in the database. Because we use TFRecord '
        'we cannot easily query the number of objects efficiencly.')
    p.Define('max_num_points_per_bbox', 2048,
             'Maximum number of points in each bbox to augment with.')
    p.Define(
        'filter_min_points', 0,
        'Minimum number of points each database object must have '
        'to be included in an example.')
    p.Define(
        'filter_max_points', None,
        'Maximum number of points each database object must have '
        'to be included in an example.')
    p.Define(
        'difficulty_sampling_probability', None,
        'Probability for sampling ground truth example whose difficulty '
        'equals {0, 1, 2, 3, ...}. Example: [1.0, 1.0, 1.0, 1.0] for '
        'uniform sampling 4 different difficulties. Default value is '
        'None = uniform sampling for all difficulties.')
    p.Define(
        'class_sampling_probability', None,
        'Probability for sampling ground truth example based on its class index'
        ' Example: For KITTI classes are [Background, Car, Van, Truck, '
        'Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare], using '
        'probability vector [0., 1.0, 1.0, 0., 0., 0., 0.,0., 0., 0.], we '
        'uniformly sampling Car and Van. Default value is None: Uses '
        'label_filter flag and does not sample based on class.')
    p.Define('filter_min_difficulty', 0,
             'Filter ground truth boxes whose difficulty is < this value.')
    p.Define('max_augmented_bboxes', 15,
             'Maximum number of augmented bounding boxes per scene.')
    p.Define(
        'label_filter', [],
        'A list where if specified, only examples of these label integers will '
        'be included in an example.')
    p.Define(
        'batch_mode', False, 'Bool value to control whether the whole'
        'groundtruth database is loaded or partially loaded to save memory'
        'usage. Setting to False loads the whole ground truth database into '
        'memory. Otherwise, only a fraction of the data will be loaded into '
        'the memory.')
    return p

  def _ReadDB(self, file_patterns):
    """Read the groundtruth database and return as a NestedMap of Tensors."""
    p = self.params

    def Process(record):
      """Process a groundtruth record."""
      feature_map = {
          'num_points': tf.io.FixedLenFeature((), tf.int64, 0),
          'points': tf.io.VarLenFeature(dtype=tf.float32),
          'points_feature': tf.io.VarLenFeature(dtype=tf.float32),
          'bbox_3d': tf.io.VarLenFeature(dtype=tf.float32),
          'label': tf.io.FixedLenFeature((), tf.int64, 0),
          'difficulty': tf.io.FixedLenFeature((), tf.int64, 0),
          'text': tf.io.VarLenFeature(dtype=tf.string),
      }

      example_data = tf.io.parse_single_example(record, feature_map)
      num_points = example_data['num_points']

      points = tf.reshape(_Dense(example_data['points']), [num_points, 3])
      features = tf.reshape(
          _Dense(example_data['points_feature']), [num_points, 1])
      points_mask = tf.ones(num_points, dtype=tf.bool)

      # TODO(vrv): Use random selection instead of first N points.
      points = py_utils.PadOrTrimTo(points, [p.max_num_points_per_bbox, 3])
      features = py_utils.PadOrTrimTo(features, [p.max_num_points_per_bbox, 1])
      points_mask = py_utils.PadOrTrimTo(points_mask,
                                         [p.max_num_points_per_bbox])

      bboxes_3d = tf.reshape(_Dense(example_data['bbox_3d']), [7])
      label = tf.cast(example_data['label'], tf.int32)
      difficulty = tf.cast(example_data['difficulty'], tf.int32)
      return (points, features, points_mask, bboxes_3d, label, difficulty)

    if p.batch_mode:
      # Prepare dataset for ground truth bounding boxes. Randomly shuffle the
      # file patterns.
      file_count = len(tf.io.gfile.glob(file_patterns))
      dataset = tf.stateless_list_files(file_patterns)
      dataset = dataset.apply(tf.stateless_cache_dataset())
      dataset = dataset.apply(
          tf.stateless_shuffle_dataset(
              buffer_size=file_count, reshuffle_each_iteration=True))
      dataset = dataset.interleave(
          tf.data.TFRecordDataset, cycle_length=10, num_parallel_calls=10)
      dataset = dataset.repeat()
      # Only prefetch a few objects from the database to reduce memory
      # consumption.
      dataset = dataset.map(Process, num_parallel_calls=10)
      # We need more bboxes than max_augmented_bboxes in a batch, because some
      # of the boxes are filtered out.
      dataset = dataset.batch(p.max_augmented_bboxes * 10)
      dataset = dataset.apply(tf.stateless_cache_dataset()).prefetch(
          p.max_augmented_bboxes * 30)
    else:
      # Prepare dataset for ground truth bounding boxes.
      dataset = tf.stateless_list_files(file_patterns)
      dataset = dataset.interleave(
          tf.data.TFRecordDataset, cycle_length=10, num_parallel_calls=10)
      # Read the entire dataset into memory.
      dataset = dataset.take(p.num_db_objects)
      dataset = dataset.map(Process, num_parallel_calls=10)
      # We batch the output of the dataset into a very large Tensor, then cache
      # it in memory.
      dataset = dataset.batch(p.num_db_objects)
      dataset = dataset.apply(tf.stateless_cache_dataset()).repeat()

    iterator = dataset.make_one_shot_iterator()
    input_batch = iterator.get_next()

    (db_points_xyz, db_points_feature, db_points_mask, db_bboxes, db_labels,
     db_difficulties) = input_batch
    return py_utils.NestedMap(
        points_xyz=db_points_xyz,
        points_feature=db_points_feature,
        points_mask=db_points_mask,
        bboxes_3d=db_bboxes,
        labels=db_labels,
        difficulties=db_difficulties)

  def _CreateExampleFilter(self, db):
    """Construct db example filter.

    Args:
      db: NestedMap of the following Tensors: points_mask - [N, P] - The points
        mask for every object in the database, where N is the number of objects
        and P is the maximum number of points per object.  labels - [N] - int32
        Label for each object in the database.  difficulties - [N] - int32
        Difficulty for each label in the database.

    Returns:
      A [N] boolean Tensor for each object in the database, True if
      that corresponding object passes the filter.
    """
    p = self.params
    db_points_mask = db.points_mask
    db_label = db.labels
    db_difficulty = db.difficulties

    num_objects_in_database = tf.shape(db_points_mask)[0]

    # Filter number of objects.
    points_per_object = tf.reduce_sum(tf.cast(db_points_mask, tf.int32), axis=1)
    example_filter = points_per_object >= p.filter_min_points
    if p.filter_max_points:
      example_filter = tf.math.logical_and(
          example_filter, points_per_object <= p.filter_max_points)

    if p.difficulty_sampling_probability is not None:
      # Sample db based on difficulity of each example.
      sampling_prob = p.difficulty_sampling_probability
      db_difficulty_probability = tf.zeros_like(db_difficulty, dtype=tf.float32)
      for difficulty_idx, difficulty_prob in enumerate(sampling_prob):
        db_difficulty_probability += (
            tf.cast(tf.equal(db_difficulty, difficulty_idx), tf.float32) *
            difficulty_prob)

      sampled_filter = tf.random.uniform(
          tf.shape(example_filter),
          minval=0,
          maxval=1,
          dtype=tf.float32,
          seed=p.random_seed)
      sampled_filter = sampled_filter < db_difficulty_probability
      example_filter &= sampled_filter
    else:
      # Filter out db examples below min difficulty
      example_filter = tf.math.logical_and(
          example_filter, db_difficulty >= p.filter_min_difficulty)

    example_filter = tf.reshape(example_filter, [num_objects_in_database])
    db_label = tf.reshape(db_label, [num_objects_in_database])
    if p.class_sampling_probability is not None:
      # Sample example based on its class probability.
      sampling_prob = p.class_sampling_probability
      db_class_probability = tf.zeros_like(db_label, dtype=tf.float32)

      for class_idx, class_prob in enumerate(sampling_prob):
        db_class_probability += (
            tf.cast(tf.equal(db_label, class_idx), tf.float32) * class_prob)

      sampled_filter = tf.random.uniform(
          tf.shape(example_filter),
          minval=0,
          maxval=1,
          dtype=tf.float32,
          seed=p.random_seed)
      sampled_filter = sampled_filter < db_class_probability
      example_filter &= sampled_filter
    elif p.label_filter:
      # Filter based on labels.
      # Create a label filter where all is false
      valid_labels = tf.constant(p.label_filter)
      label_mask = tf.reduce_any(
          tf.equal(db_label[..., tf.newaxis], valid_labels), axis=1)
      example_filter = tf.math.logical_and(example_filter, label_mask)
    return example_filter

  # TODO(vrv): Create an overlap filter that also ensures that boxes don't
  # overlap with groundtruth points, so that the scenes are more plausible.
  def _FilterIndices(self, gt_bboxes_3d, db_bboxes, db_idx):
    """Identify database boxes that don't overlap with other boxes."""
    # We accomplish overlap filtering by first computing the pairwise 3D IoU of
    # all boxes (concatenated) as a way of computing pairwise box overlaps.
    num_gt_bboxes = tf.shape(gt_bboxes_3d)[0]
    filtered_bboxes = tf.gather(db_bboxes, db_idx)
    all_bboxes = tf.concat([gt_bboxes_3d, filtered_bboxes], axis=0)
    pairwise_overlap = ops.pairwise_iou3d(all_bboxes, all_bboxes)

    # We now have an M x M matrix with 1s on the diagonal and non-zero entries
    # whenever a box collides with another.
    #
    # To increase the number of boxes selected, we filter the upper triangular
    # entries so that the boxes are chosen greedily: boxes with smaller indices
    # will be selected before later boxes, because earlier boxes will not appear
    # to collide with later boxes, but later boxes may collide with earlier
    # ones.
    pairwise_overlap = tf.linalg.band_part(pairwise_overlap, -1, 0)

    # We compute the sum of the IoU overlaps for all database boxes.
    db_overlap_sums = tf.reduce_sum(pairwise_overlap[num_gt_bboxes:], axis=1)

    # Those boxes that don't overlap with any other boxes will only have
    # a 1.0 IoU with itself.
    non_overlapping_boxes = tf.reshape(db_overlap_sums <= 1., [-1])

    # Filter to select only those object ids that pass this filter.
    db_idx = tf.boolean_mask(db_idx, non_overlapping_boxes)
    return db_idx

  def TransformFeatures(self, features):
    p = self.params

    tf.logging.info('Loading groundtruth database at %s' %
                    (p.groundtruth_database))
    db = self._ReadDB(p.groundtruth_database)

    original_features_shape = tf.shape(features.lasers.points_feature)

    # Compute the number of bboxes to augment.
    num_bboxes_in_scene = tf.reduce_sum(
        tf.cast(features.labels.bboxes_3d_mask, tf.int32))
    max_bboxes = tf.shape(features.labels.bboxes_3d_mask)[0]
    num_augmented_bboxes = tf.minimum(max_bboxes - num_bboxes_in_scene,
                                      p.max_augmented_bboxes)

    # Compute an object index over all objects in the database.
    num_objects_in_database = tf.shape(db.points_xyz)[0]
    db_idx = tf.range(num_objects_in_database)

    # Find those indices whose examples pass the filters, and select only those
    # indices.
    example_filter = self._CreateExampleFilter(db)
    db_idx = tf.boolean_mask(db_idx, example_filter)

    # At this point, we might still have a large number of object candidates,
    # from which we only need a sample.
    # To reduce the amount of computation, we randomly subsample to slightly
    # more than we want to augment.
    db_idx = tf.random.shuffle(
        db_idx, seed=p.random_seed)[0:num_augmented_bboxes * 5]

    # After filtering, further filter out the db boxes that would occlude with
    # other boxes (including other database boxes).
    #
    # Gather the filtered ground truth bounding boxes according to the mask, so
    # we can compute overlaps below.
    gt_bboxes_3d_mask = tf.cast(features.labels.bboxes_3d_mask, tf.bool)
    gt_bboxes_3d = tf.boolean_mask(features.labels.bboxes_3d, gt_bboxes_3d_mask)
    gt_bboxes_3d = py_utils.HasShape(gt_bboxes_3d, [num_bboxes_in_scene, 7])
    db_idx = self._FilterIndices(gt_bboxes_3d, db.bboxes_3d, db_idx)

    # From the filtered object ids, select only as many boxes as we need.
    shuffled_idx = db_idx[0:num_augmented_bboxes]
    num_augmented_bboxes = tf.shape(shuffled_idx)[0]

    # Gather based off the indices.
    sampled_points_xyz = tf.gather(db.points_xyz, shuffled_idx)
    sampled_points_feature = tf.gather(db.points_feature, shuffled_idx)
    sampled_mask = tf.reshape(
        tf.gather(db.points_mask, shuffled_idx),
        [num_augmented_bboxes, p.max_num_points_per_bbox])
    sampled_bboxes = tf.gather(db.bboxes_3d, shuffled_idx)
    sampled_labels = tf.gather(db.labels, shuffled_idx)

    # Mask points/features.
    sampled_points_xyz = tf.boolean_mask(sampled_points_xyz, sampled_mask)
    sampled_points_feature = tf.boolean_mask(sampled_points_feature,
                                             sampled_mask)

    # Flatten before concatenation with ground truths.
    sampled_points_xyz = tf.reshape(sampled_points_xyz, [-1, 3])
    sampled_points_feature = tf.reshape(sampled_points_feature,
                                        [-1, original_features_shape[-1]])
    sampled_bboxes = tf.reshape(sampled_bboxes, [-1, 7])

    # Concatenate the samples with the ground truths.
    if 'points_padding' in features.lasers:
      points_mask = tf.cast(1. - features.lasers.points_padding, tf.bool)
      # Densify the original points.
      dense_points_xyz = tf.boolean_mask(features.lasers.points_xyz,
                                         points_mask)
      dense_points_feature = tf.boolean_mask(features.lasers.points_feature,
                                             points_mask)

      # Concatenate the dense original points with our new sampled oints.
      points_xyz = tf.concat([dense_points_xyz, sampled_points_xyz], axis=0)
      points_feature = tf.concat([dense_points_feature, sampled_points_feature],
                                 axis=0)
      original_points_shape = tf.shape(features.lasers.points_xyz)
      features.lasers.points_xyz = py_utils.PadOrTrimTo(points_xyz,
                                                        original_points_shape)
      features.lasers.points_feature = py_utils.PadOrTrimTo(
          points_feature, original_features_shape)
      # Compute the modified mask / padding.
      final_points_mask = py_utils.PadOrTrimTo(
          tf.ones(tf.shape(points_xyz)[0]),
          tf.shape(features.lasers.points_padding))
      features.lasers.points_padding = 1. - final_points_mask
    else:
      points_xyz = tf.concat([features.lasers.points_xyz, sampled_points_xyz],
                             axis=0)
      points_feature = tf.concat(
          [features.lasers.points_feature, sampled_points_feature], axis=0)
      features.lasers.points_xyz = points_xyz
      features.lasers.points_feature = points_feature

    # Reconstruct a new, dense, bboxes_3d vector that includes the filtered
    # groundtruth bounding boxes followed by the database augmented boxes.
    bboxes_3d = tf.concat([gt_bboxes_3d, sampled_bboxes], axis=0)
    bboxes_3d = py_utils.PadOrTrimTo(bboxes_3d, [max_bboxes, 7])
    features.labels.bboxes_3d = bboxes_3d
    bboxes_3d_mask = tf.ones(
        num_bboxes_in_scene + num_augmented_bboxes, dtype=tf.float32)
    features.labels.bboxes_3d_mask = py_utils.PadOrTrimTo(
        bboxes_3d_mask, [max_bboxes])

    gt_labels = tf.boolean_mask(features.labels.labels, gt_bboxes_3d_mask)
    gt_labels = py_utils.HasShape(gt_labels, [num_bboxes_in_scene])

    labels = tf.concat([gt_labels, sampled_labels], axis=0)
    features.labels.labels = py_utils.PadOrTrimTo(labels, [max_bboxes])

    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class FrustumDropout(Preprocessor):
  """Randomly drops out points in a frustum.

  All points are first converted to spherical coordinates, and then a point
  is randomly selected. All points in the frustum around that point within
  a given phi, theta angle width and distance to the original greater than
  a given value are dropped with probability = 1 - keep_prob.

  Here, we can specify whether the dropped frustum is the union or intersection
  of the phi and theta angle filters.


  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]
  - lasers.points_feature of shape [P, F]

  Optionally points_padding of shape [P] corresponding to the padding.
  if points_padding is None, then all points are considered valid.

  Modifies the following features:
    lasers.points_xyz, lasers.points_feature, lasers.points_padding with points
    randomly dropped out.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('theta_width', 0.03, 'Theta angle width for dropping points.')
    p.Define('phi_width', 0.0, 'Phi angle width for dropping points.')
    p.Define(
        'distance', 0.0, 'Drop points that have larger distance to the'
        'origin than the value given here.')
    p.Define(
        'keep_prob', 0.0, 'keep_prob: 1. = drop no points in the Frustum,'
        '0 = drop all points, between 0 and 1 = down sample the points.')
    p.Define(
        'drop_type', 'union', 'Drop either the union or intersection of '
        'phi width and theta width.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.phi_width < 0:
      raise ValueError('phi_width must be >= 0, phi_width={}'.format(
          p.phi_width))
    if p.theta_width < 0:
      raise ValueError('theta_width must be >= 0, theta_width={}'.format(
          p.theta_width))
    if p.distance < 0:
      raise ValueError('distance must be >= 0, distance={}'.format(p.distance))
    if p.keep_prob < 0 or p.keep_prob > 1:
      raise ValueError('keep_prob must be >= 0 and <=1, keep_prob={}'.format(
          p.keep_prob))
    if p.drop_type not in ['union', 'intersection']:
      raise ValueError('drop_type must be union or intersection ,'
                       'drop_type={}'.format(p.drop_type))

  def TransformFeatures(self, features):
    p = self.params
    points_xyz = features.lasers.points_xyz
    points_feature = features.lasers.points_feature
    if 'points_padding' in features.lasers:
      points_padding = features.lasers.points_padding
    else:
      points_padding = None

    if points_padding is not None:
      points_mask = tf.cast(1 - points_padding, tf.bool)
      num_total_points = py_utils.GetShape(points_mask)[0]
      real_points_idx = tf.boolean_mask(
          tf.range(0, num_total_points, dtype=tf.int32), points_mask)
      num_points = py_utils.GetShape(real_points_idx)[0]
    else:
      points_mask = tf.ones_like(points_xyz[:, 0], dtype=tf.bool)
      num_total_points = py_utils.GetShape(points_mask)[0]
      num_points = py_utils.GetShape(points_xyz)[0]

    r, theta, phi = tf.unstack(
        geometry.SphericalCoordinatesTransform(points_xyz), axis=-1)

    def _PickRandomPoint():
      point_idx = tf.random.uniform((),
                                    minval=0,
                                    maxval=num_points,
                                    dtype=tf.int32)
      if points_padding is not None:
        point_idx = real_points_idx[point_idx]
      return point_idx

    # Pick a point at random and drop all points that are near that point in the
    # frustum for distance larger than r; repeat this for both theta and phi.
    if p.theta_width > 0:
      theta_half_width = p.theta_width / 2.
      point_idx = _PickRandomPoint()
      # Points within theta width and further than distance will be dropped.
      theta_drop_filter = ((theta < (theta[point_idx] + theta_half_width)) &
                           (theta > (theta[point_idx] - theta_half_width)) &
                           (r > p.distance))
    else:
      theta_drop_filter = tf.zeros_like(points_mask, dtype=tf.bool)

    if p.phi_width > 0:
      phi_half_width = p.phi_width / 2.
      point_idx = _PickRandomPoint()
      # Points within phi width and further than distance will be dropped.
      phi_drop_filter = ((phi < (phi[point_idx] + phi_half_width)) &
                         (phi >
                          (phi[point_idx] - phi_half_width)) & (r > p.distance))
    else:
      phi_drop_filter = tf.zeros_like(points_mask, dtype=tf.bool)

    #  Create drop_filter by combining filters. This contains a filter for the
    #  points to be removed. One can use the intersection method to limit the
    #  dropped points be within both phi and theta ranges.
    if p.drop_type == 'union':
      drop_filter = theta_drop_filter | phi_drop_filter
    elif p.drop_type == 'intersection':
      drop_filter = theta_drop_filter & phi_drop_filter

    if p.keep_prob == 0:
      # Drop all points in drop_filter.
      down_sampling_filter = drop_filter
    else:
      # Randomly drop points in drop_filter based on keep_prob.
      sampling_drop_filter = tf.random.uniform([num_total_points],
                                               minval=0,
                                               maxval=1,
                                               dtype=tf.float32)
      # Points greater than the threshold (keep_prob) will be dropped.
      sampling_drop_filter = sampling_drop_filter > p.keep_prob

      # Instead of dropping all points in the frustum, we drop out points
      # that are in the selected frustum (drop_filter).
      down_sampling_filter = drop_filter & sampling_drop_filter

    points_mask &= ~down_sampling_filter

    if points_padding is not None:
      features.lasers.points_padding = 1 - tf.cast(points_mask, tf.float32)
    else:
      features.lasers.points_xyz = tf.boolean_mask(points_xyz, points_mask)
      features.lasers.points_feature = tf.boolean_mask(points_feature,
                                                       points_mask)

    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class RepeatPreprocessor(Preprocessor):
  """Repeat a preprocessor multiple times.

  This preprocessor takes a preprocessor as a subprocessor and apply the
  subprocessor to features multiple times (repeat_count).

  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('repeat_count', 1, 'Number of times the subprocessor is applied to'
             ' features.')
    p.Define('subprocessor', None, 'One of the input preprocessors.')

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.subprocessor is None:
      raise ValueError('No subprocessor was specified for RepeatPreprocessor.')
    if p.repeat_count < 0 or not isinstance(p.repeat_count, int):
      raise ValueError(
          'repeat_count must be >= 0 and int, repeat_count={}'.format(
              p.repeat_count))

    self.CreateChild('subprocessor', p.subprocessor)

  def TransformFeatures(self, features):
    p = self.params
    for _ in range(p.repeat_count):
      features = self.subprocessor.FPropDefaultTheta(features)

    return features

  def TransformShapes(self, shapes):
    p = self.params
    for _ in range(p.repeat_count):
      shapes = self.subprocessor.TransformShapes(shapes)

    return shapes

  def TransformDTypes(self, dtypes):
    p = self.params
    for _ in range(p.repeat_count):
      dtypes = self.subprocessor.TransformDTypes(dtypes)

    return dtypes


class RandomApplyPreprocessor(Preprocessor):
  """Randomly apply a preprocessor with certain probability.

  This preprocessor takes a preprocessor as a subprocessor and apply the
  subprocessor to features with certain probability.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('prob', 1.0, 'The probability the subprocessor being executed.')
    p.Define('subprocessor', None, 'Params for an input preprocessor.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.subprocessor is None:
      raise ValueError('No subprocessor was specified for RepeatPreprocessor.')
    if p.prob < 0 or p.prob > 1 or not isinstance(p.prob, float):
      raise ValueError(
          'prob must be >= 0 and <=1 and float type, prob={}'.format(p.prob))

    self.CreateChild('subprocessor', p.subprocessor)

  def TransformFeatures(self, features):
    p = self.params
    choice = tf.random.uniform(
        (), minval=0.0, maxval=1.0, seed=p.random_seed) <= p.prob
    # Features is passed downstream and may be modified, we make deep copies
    # here to use with tf.cond to avoid having tf.cond access updated
    # versions. Note that we need one copy for each branch in case the branches
    # further modify features.
    features_0, features_1 = features.DeepCopy(), features.DeepCopy()
    features = tf.cond(choice,
                       lambda: self.subprocessor.TransformFeatures(features_0),
                       lambda: features_1)
    return features

  def TransformShapes(self, shapes):
    shapes_transformed = self.subprocessor.TransformShapes(shapes)

    if not shapes.IsCompatible(shapes_transformed):
      raise ValueError(
          'NestedMap structures are different between shapes and transformed'
          'shapes. Original shapes: {}. Transformed shapes: {}'.format(
              shapes, shapes_transformed))

    def IsCompatibleWith(a, b):
      return a.is_compatible_with(b)

    if not all(
        py_utils.Flatten(
            py_utils.Transform(IsCompatibleWith, shapes, shapes_transformed))):
      raise ValueError(
          'Shapes after transformation - {} are different from original '
          'shapes - {}.'.format(shapes_transformed, shapes))

    return shapes

  def TransformDTypes(self, dtypes):
    transformed_dtypes = self.subprocessor.TransformDTypes(dtypes)
    if transformed_dtypes != dtypes:
      raise ValueError(
          'DTypes after transformation of preprocessor - {} should be '
          'the same as {}, but get {}.'.format(self.params.subprocessor, dtypes,
                                               transformed_dtypes))
    return dtypes


class ConstantPreprocessor(Preprocessor):
  """Preprocessor that produces specified constant values in a nested output."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'constants', py_utils.NestedMap(),
        'Map of key names to numpy arrays of constant values to use. '
        'Must be a NestedMap or dict convertible to NestedMap.')
    return p

  def TransformFeatures(self, features):
    constants = py_utils.NestedMap(self.params.constants)
    features.update(constants.Transform(tf.constant))
    return features

  def TransformShapes(self, shapes):
    constants = py_utils.NestedMap(self.params.constants)
    shapes.update(
        constants.Transform(lambda x: tf.TensorShape(np.array(x).shape)))
    return shapes

  def TransformDTypes(self, dtypes):
    constants = py_utils.NestedMap(self.params.constants)
    dtypes.update(constants.Transform(lambda x: tf.as_dtype(np.array(x).dtype)))
    return dtypes


class IdentityPreprocessor(Preprocessor):
  """Preprocessor that passes all inputs through.

  This may be useful for situations where one wants a 'no-op' preprocessor, such
  as being able to randomly choose to do nothing among a set of preprocessor
  choices.
  """

  def TransformFeatures(self, features):
    return features

  def TransformShapes(self, shapes):
    return shapes

  def TransformDTypes(self, dtypes):
    return dtypes


class RandomChoicePreprocessor(Preprocessor):
  """Randomly applies a preprocessor with specified weights.

  The input at features[p.weight_tensor_key] must be a floating point vector
  Tensor whose length matches the number of subprocessors to select among. The
  values in that Tensor are interpreted as relative weights.

  For example, if p.subprocessors = [preprocessor1, preprocessor2] and the
  weights are [1., 2.], then preprocessor1 will be applied with probability 1/3,
  and preprocessor2 will be applied with probability 2/3.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'subprocessors', [],
        'Params for preprocessors. Each value should be a tuple of '
        '(Preprocessor.Params(), BaseSchedule.Params()), where the schedule '
        'defines the weights to use over time.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if not p.subprocessors:
      raise ValueError('No subprocessors were specified.')

    subprocessors, schedules = zip(*p.subprocessors)

    def _FilterNonSchedules(v):
      return not issubclass(getattr(v, 'cls', False), schedule.BaseSchedule)

    invalid_values = [_FilterNonSchedules(s) for s in schedules]
    if any(invalid_values):
      raise TypeError('Not all schedule values were schedules: '
                      f'{invalid_values}')

    self.CreateChildren('subprocessors', list(subprocessors))
    self.CreateChildren('schedules', list(schedules))

  def TransformFeatures(self, features):
    p = self.params

    choice_list = []
    weight_list = []

    # Pass a unique copy of the input to each branch, in case the
    # subprocessor destructively modifies the features in unexpected ways.
    for subp, sched in zip(self.subprocessors, self.schedules):
      choice_list.append(
          lambda subp=subp: subp.TransformFeatures(features.DeepCopy()))
      weight_list.append(sched.Value())

    weight_tensor = tf.stack(weight_list)
    chosen_bin = tf.random.categorical(
        tf.math.log(weight_tensor[tf.newaxis]),
        1,
        seed=p.random_seed,
        dtype=tf.int32)[0, 0]
    features = tf.switch_case(chosen_bin, branch_fns=choice_list)
    return features

  def TransformShapes(self, shapes):
    transformed_shapes = [
        subp.TransformShapes(shapes.DeepCopy()) for subp in self.subprocessors
    ]
    if not all(transformed_shapes[0] == curr for curr in transformed_shapes):
      raise ValueError('Shapes after transformations were not identical: '
                       f'{transformed_shapes}')
    return transformed_shapes[0]

  def TransformDTypes(self, dtypes):
    transformed_dtypes = [
        subp.TransformDTypes(dtypes.DeepCopy()) for subp in self.subprocessors
    ]
    if not all(transformed_dtypes[0] == curr for curr in transformed_dtypes):
      raise ValueError('DTypes after transformations were not identical: '
                       f'{transformed_dtypes}')
    return transformed_dtypes[0]


class Sequence(Preprocessor):
  """Packages a sequence of preprocessors as one preprocessor."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'preprocessors', [], 'A list of preprocessors. '
        'Each should be of type Preprocessor.Params().')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChildren('preprocessors', p.preprocessors)

  def TransformFeatures(self, features):
    for preprocessor in self.preprocessors:
      features = preprocessor.TransformFeatures(features)
    return features

  def TransformShapes(self, shapes):
    for preprocessor in self.preprocessors:
      shapes = preprocessor.TransformShapes(shapes)
    return shapes

  def TransformDTypes(self, dtypes):
    for preprocessor in self.preprocessors:
      dtypes = preprocessor.TransformDTypes(dtypes)
    return dtypes


class SparseSampler(Preprocessor):
  """Fused SparseCenterSelector and SparseCellGatherFeatures.

  This preprocessor expects features to contain the following keys:
  - lasers.points_xyz of shape [P, 3]
  - lasers.points_feature of shape [P, F]

  Adds the following features:
    anchor_centers - [num_centers, 3] - Floating point output containing the
    center (x, y, z) locations for tiling anchor boxes.

    cell_center_xyz - [num_centers, 3] - Floating point output containing
    the center (x, y, z) locations for each cell to featurize.

    cell_center_padding - [num_centers] - 0/1 padding for each center.

    cell_points_xyz - [num_centers, num_neighbors, 3] - Floating point
    output containing the (x, y, z) locations for each point for a given
    center.

    cell_feature - [num_centers, num_neighbors, F] - Floating point output
    containing the features for each point for a given center.

    cell_points_padding - [num_centers, num_neighbors] - 0/1 padding
    for the points in each cell.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('center_selector', 'farthest', 'Method to sample centers. '
             'Valid options - uniform, farthest.')
    p.Define('neighbor_sampler', 'uniform', 'Method to select neighbors. '
             'Valid options - uniform, closest.')
    p.Define('num_centers', 16, 'The number of centers to sample.')
    p.Define(
        'features_preparation_layers', [],
        'A list of Params for layers to run on the features before '
        'performing farthest point sampling. For example, one may wish to '
        'drop points out of frustum for KITTI before selecting centers. '
        'Note that these layers will not mutate the original features, '
        'instead, a copy will be made.')
    p.Define(
        'keep_z_range', (-np.inf, np.inf),
        'Only points that have z coordinates within this range are kept. '
        'Approximate ground-removal can be performed by specifying a '
        'lower-bound on the z-range.')
    p.Define('num_neighbors', 64, 'Sample these many points within the '
             'neighorhood.')
    p.Define(
        'max_distance', 1.0, 'Points with L2 distances from a center '
        'larger than this threshold are not considered to be in the '
        'neighborhood.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.features_preparation_layers:
      self.CreateChildren('features_preparation_layers',
                          p.features_preparation_layers)

  def TransformFeatures(self, features):
    p = self.params
    n, m = p.num_centers, p.num_neighbors

    prepared_features = features.DeepCopy()
    if p.features_preparation_layers:
      for prep_layer in self.features_preparation_layers:
        prepared_features = prep_layer.FPropDefaultTheta(prepared_features)

    points_data = prepared_features.lasers
    points = py_utils.HasShape(points_data.points_xyz, [-1, 3])

    if 'points_padding' in points_data:
      points_mask = 1 - points_data.points_padding
      points = tf.boolean_mask(points, points_mask)

    # If num_points < num_centers, pad points to have at least num_centers
    # points.
    num_points = tf.shape(points)[0]
    required_num_points = tf.maximum(num_points, p.num_centers)
    zeros = tf.zeros([required_num_points - num_points, 3])
    points = tf.concat([points, zeros], axis=0)

    num_seeded_points = points_data.get('num_seeded_points', 0)

    neighbor_algorithm = 'auto'
    # Based on benchmarks, the hash solution works better when the number of
    # centers is >= 16 and there are at least 10k points per point cloud.
    if p.num_centers >= 16:
      neighbor_algorithm = 'hash'

    centers, center_paddings, indices, indices_paddings = ops.sample_points(
        points=tf.expand_dims(points, 0),
        points_padding=tf.zeros([1, required_num_points], tf.float32),
        num_seeded_points=num_seeded_points,
        center_selector=p.center_selector,
        neighbor_sampler=p.neighbor_sampler,
        neighbor_algorithm=neighbor_algorithm,
        num_centers=p.num_centers,
        center_z_min=p.keep_z_range[0],
        center_z_max=p.keep_z_range[1],
        num_neighbors=p.num_neighbors,
        max_distance=p.max_distance,
        random_seed=p.random_seed if p.random_seed else -1)
    centers = py_utils.HasShape(centers, [1, n])[0, :]
    center_paddings = py_utils.HasShape(center_paddings, [1, n])[0, :]
    indices = py_utils.HasShape(indices, [1, n, m])[0, :]
    indices_paddings = py_utils.HasShape(indices_paddings, [1, n, m])[0, :]
    features.cell_center_padding = center_paddings
    features.cell_center_xyz = py_utils.HasShape(
        tf.gather(points, centers), [n, 3])
    features.anchor_centers = features.cell_center_xyz
    features.cell_points_xyz = py_utils.HasShape(
        tf.gather(points, indices), [n, m, 3])
    features.cell_feature = tf.gather(points_data.points_feature, indices)
    features.cell_points_padding = indices_paddings
    return features

  def TransformShapes(self, shapes):
    p = self.params
    n, m, f = p.num_centers, p.num_neighbors, shapes.lasers.points_feature[-1]
    shapes.anchor_centers = tf.TensorShape([n, 3])
    shapes.cell_center_padding = tf.TensorShape([n])
    shapes.cell_center_xyz = tf.TensorShape([n, 3])
    shapes.cell_points_xyz = tf.TensorShape([n, m, 3])
    shapes.cell_feature = tf.TensorShape([n, m, f])
    shapes.cell_points_padding = tf.TensorShape([n, m])
    return shapes

  def TransformDTypes(self, dtypes):
    dtypes.anchor_centers = tf.float32
    dtypes.cell_center_padding = tf.float32
    dtypes.cell_center_xyz = tf.float32
    dtypes.cell_points_xyz = tf.float32
    dtypes.cell_feature = tf.float32
    dtypes.cell_points_padding = tf.float32
    return dtypes
