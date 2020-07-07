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
"""Input generator for KITTI data."""

from lingvo import compat as tf
from lingvo.core import datasource
from lingvo.core import hyperparams
from lingvo.core import ops
from lingvo.core import py_utils
from lingvo.tasks.car import geometry
from lingvo.tasks.car import input_extractor
from lingvo.tasks.car import input_preprocessors
from lingvo.tasks.car import kitti_metadata


def _Dense(sparse, default_value=0):
  return tf.sparse_to_dense(
      sparse_indices=sparse.indices,
      output_shape=sparse.dense_shape,
      sparse_values=sparse.values,
      default_value=default_value)


def _NestedMapToParams(nmap):
  p = hyperparams.Params()
  for k, v in nmap.FlattenItems():
    p.Define(k, v, '')
  return p


def ComputeKITTIDifficulties(box_image_height, occlusion, truncation):
  """Compute difficulties from box height, occlusion, and truncation."""
  # Easy: No occlusion, max truncation 15%
  easy_level = tf.cast((box_image_height >= 40.) & (occlusion <= 0.)
                       & (truncation <= 0.15), tf.int32) * 3
  # Moderate: max occlusion: partly occluded, max truncation 30%
  moderate_level = tf.cast((occlusion <= 1.) & (truncation <= 0.3)
                           & (box_image_height >= 25.), tf.int32) * 2
  # Hard: Difficult to see, max truncation 50%
  hard_level = tf.cast((occlusion <= 2.) & (truncation <= 0.5)
                       & (box_image_height >= 25.), tf.int32) * 1

  # Occlusion = 3 and higher truncation is "super hard", and
  # will map to 0 (ignored).
  difficulties = tf.maximum(tf.maximum(hard_level, moderate_level), easy_level)

  return difficulties


class KITTILaserExtractor(input_extractor.LaserExtractor):
  """Base extractor for the laser points from a KITTI tf.Example."""

  @classmethod
  def Params(cls):
    p = super().Params().Set(max_num_points=None, num_features=1)
    return p

  def FeatureMap(self):
    feature_map = {
        'pointcloud/xyz': tf.io.VarLenFeature(dtype=tf.float32),
        'pointcloud/reflectance': tf.io.VarLenFeature(dtype=tf.float32),
    }
    return feature_map

  def _Extract(self, features):
    p = self.params
    points_xyz = tf.reshape(_Dense(features['pointcloud/xyz']), [-1, 3])
    points_feature = tf.reshape(
        _Dense(features['pointcloud/reflectance']), [-1, p.num_features])

    if p.max_num_points is not None:
      npoints = tf.shape(points_xyz)[0]
      points_xyz = py_utils.PadOrTrimTo(points_xyz, [p.max_num_points, 3])
      points_feature = py_utils.PadOrTrimTo(points_feature,
                                            [p.max_num_points, p.num_features])
      points_padding = 1.0 - py_utils.PadOrTrimTo(
          tf.ones([npoints]), [p.max_num_points])

    ret = py_utils.NestedMap(
        points_xyz=points_xyz, points_feature=points_feature)
    if p.max_num_points is not None:
      ret.points_padding = points_padding
    return ret


class KITTIImageExtractor(input_extractor.FieldsExtractor):
  """Extracts the image information (left camera) from a KITTI tf.Example.

  Produces:
    image: [512, 1382, 3] - Floating point Tensor containing image data. Note
    that image may not be produced if decode_image is set to False. During
    training, we may not want to decode the images.

    width: [1] - integer scalar width of the original image.

    height: [1] - integer scalar width of the original image.

    velo_to_image_plane: [3, 4] - transformation matrix from velo xyz to image
    plane xy. After multiplication, you need to divide by last coordinate to
    recover 2D pixel locations.

    velo_to_camera: [4, 4] - transformation matrix from velo xyz to camera xyz.

    camera_to_velo: [4, 4] - transformation matrix from camera xyz to velo xyz.
  """
  _KITTI_MAX_HEIGHT = 512
  _KITTI_MAX_WIDTH = 1382

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('decode_image', True, 'Whether to decode and produce image.')
    return p

  def FeatureMap(self):
    p = self.params
    feature_map = {
        'image/format':
            tf.io.FixedLenFeature((), tf.string, default_value='png'),
        'image/height':
            tf.io.FixedLenFeature((), tf.int64, default_value=1),
        'image/width':
            tf.io.FixedLenFeature((), tf.int64, default_value=1),
        'image/source_id':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        # The camera calibration matrices can be used later with width/height
        # to perform out of camera frustum point dropping.
        'transform/velo_to_image_plane':
            tf.io.FixedLenFeature(shape=(3, 4), dtype=tf.float32),
        'transform/velo_to_camera':
            tf.io.FixedLenFeature(shape=(4, 4), dtype=tf.float32),
        'transform/camera_to_velo':
            tf.io.FixedLenFeature(shape=(4, 4), dtype=tf.float32),
    }
    if p.decode_image:
      feature_map['image/encoded'] = tf.io.FixedLenFeature((),
                                                           tf.string,
                                                           default_value='')
    return feature_map

  def _Extract(self, features):
    p = self.params

    if p.decode_image:
      raw = features['image/encoded']
      image = tf.image.decode_png(raw, channels=3)
      image = tf.image.convert_image_dtype(image, tf.float32)
      # Padding instead of rescaling to preserve the pixel coordinates.
      image = py_utils.PadOrTrimTo(
          image, [self._KITTI_MAX_HEIGHT, self._KITTI_MAX_WIDTH, 3])

    width = tf.reshape(features['image/width'], [1])
    height = tf.reshape(features['image/height'], [1])

    velo_to_image_plane = features['transform/velo_to_image_plane']
    velo_to_camera = features['transform/velo_to_camera']
    camera_to_velo = features['transform/camera_to_velo']

    extracted_features = py_utils.NestedMap(
        width=width,
        height=height,
        velo_to_image_plane=velo_to_image_plane,
        velo_to_camera=velo_to_camera,
        camera_to_velo=camera_to_velo)

    if p.decode_image:
      extracted_features.image = image

    return extracted_features

  def Shape(self):
    p = self.params
    shape = py_utils.NestedMap(
        width=tf.TensorShape([1]),
        height=tf.TensorShape([1]),
        velo_to_image_plane=tf.TensorShape([3, 4]),
        velo_to_camera=tf.TensorShape([4, 4]),
        camera_to_velo=tf.TensorShape([4, 4]))
    if p.decode_image:
      shape.image = tf.TensorShape(
          [self._KITTI_MAX_HEIGHT, self._KITTI_MAX_WIDTH, 3])
    return shape

  def DType(self):
    p = self.params
    dtype = py_utils.NestedMap(
        width=tf.int64,
        height=tf.int64,
        velo_to_image_plane=tf.float32,
        velo_to_camera=tf.float32,
        camera_to_velo=tf.float32)
    if p.decode_image:
      dtype.image = tf.float32
    return dtype


# Various coordinate systems in the outputs:
# - bboxes: 2D "image" coordinate.
# - bboxes_3d[3:6] (locations): 3D "world" coordinate.
# - bboxes_3d[:3] (dimension)
# - points_xyz: 3D "world" coordinate.
#
# To convert from:
# "camera" to "world": use extrinsics/R and extrinsics/t.
# "camera" to "image": use intrinsics/K


class KITTILabelExtractor(input_extractor.FieldsExtractor):
  """Extracts the object labels from a KITTI tf.Example.

  Emits:
    bboxes_count: Scalar number of 2D bounding boxes in the example.

    bboxes: [p.max_num_objects, 4] - 2D bounding box data in [ymin, xmin, ymax,
    xmax] format.

    bboxes_padding: [p.max_num_objects] - Padding for bboxes.

    bboxes_3d: [p.max_num_objects, 7] - 3D bounding box data in [x, y, z, dx,
    dy, dz, phi] format.  x, y, z are the object center; dx, dy, dz are the
    dimensions of the box, and phi is the rotation angle around the z-axis.
    3D bboxes are defined in the velodyne coordinate frame.

    bboxes_3d_mask: [p.max_num_objects] - Mask for bboxes (mask is the inversion
    of padding).

    bboxes3d_proj_to_image_plane: [p.max_num_objects, 8, 2] - For each
    bounding box, the 8 corners of the bounding box in projected image
    coordinates (x, y).

    bboxes_td: [p.max_num_objects, 4] - The 3D bounding box data in top down
    projected coordinates (ymin, xmin, ymax, xmax).  This currently ignores
    rotation.

    bboxes_td_mask: [p.max_num_objects]: Mask for bboxes_td.

    bboxes_3d_num_points: [p.max_num_objects]: Number of points in each box.

    labels: [p.max_num_objects] - Integer label for each bounding box object
    corresponding to the index in KITTI_CLASS_NAMES.

    texts: [p.max_num_objects] - The class name for each label in labels.

    source_id: Scalar string. The unique identifier for each example.

  See ComputeKITTIDifficulties for more info of the following::

    box_image_height: [p.max_num_objects] - The height of the box in pixels
    of each box in the projected image plane.

    occlusion: [p.max_num_objects] - The occlusion level of each bounding box.

    truncation: [p.max_num_objects] - The truncation level of each bounding box.

    difficulties: [p.max_num_objects] - The computed difficulty based on the
    above three factors.
  """
  KITTI_CLASS_NAMES = kitti_metadata.KITTIMetadata().ClassNames()

  # Sub-classes for filtering labels when training class specific models.
  SUBCLASS_DICT = {
      'human': [4, 5],
      'cyclist': [6],
      'motor': [1, 2, 3, 7],
      'pedestrian': [4],
  }

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('max_num_objects', 50, 'The number of objects per example.')
    p.Define('filter_labels', None, 'If not None, specifies a list of label '
             'indices to keep.')
    return p

  def FeatureMap(self):
    return {
        'image/source_id':
            tf.io.FixedLenFeature((), tf.string, ''),
        'object/image/bbox/xmin':
            tf.io.VarLenFeature(tf.float32),
        'object/image/bbox/xmax':
            tf.io.VarLenFeature(tf.float32),
        'object/image/bbox/ymin':
            tf.io.VarLenFeature(tf.float32),
        'object/image/bbox/ymax':
            tf.io.VarLenFeature(tf.float32),
        'object/label':
            tf.io.VarLenFeature(tf.string),
        'object/has_3d_info':
            tf.io.VarLenFeature(dtype=tf.int64),
        'object/occlusion':
            tf.io.VarLenFeature(dtype=tf.int64),
        'object/truncation':
            tf.io.VarLenFeature(dtype=tf.float32),
        'object/velo/bbox/xyz':
            tf.io.VarLenFeature(dtype=tf.float32),
        'object/velo/bbox/dim_xyz':
            tf.io.VarLenFeature(dtype=tf.float32),
        'object/velo/bbox/phi':
            tf.io.VarLenFeature(dtype=tf.float32),
        'transform/velo_to_image_plane':
            tf.io.FixedLenFeature(shape=(3, 4), dtype=tf.float32),
    }

  def _Extract(self, features):
    p = self.params

    source_id = py_utils.HasShape(features['image/source_id'], [])
    xmin = _Dense(features['object/image/bbox/xmin'])
    xmax = _Dense(features['object/image/bbox/xmax'])
    ymin = _Dense(features['object/image/bbox/ymin'])
    ymax = _Dense(features['object/image/bbox/ymax'])

    # 2d bounding box in image coordinates.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
    bboxes_count = tf.shape(bboxes)[0]
    bboxes = py_utils.PadOrTrimTo(bboxes, [p.max_num_objects, 4])

    bboxes_padding = 1.0 - py_utils.PadOrTrimTo(
        tf.ones([bboxes_count]), [p.max_num_objects])

    dim_xyz = tf.reshape(_Dense(features['object/velo/bbox/dim_xyz']), [-1, 3])
    loc_xyz = tf.reshape(_Dense(features['object/velo/bbox/xyz']), [-1, 3])
    phi = tf.reshape(_Dense(features['object/velo/bbox/phi']), [-1, 1])
    # bboxes_3d is in [x, y, z, dx, dy, dz, phi].
    bboxes_3d = tf.concat([loc_xyz, dim_xyz, phi], axis=1)

    cx, cy, _, dx, dy, _, _ = tf.unstack(bboxes_3d, num=7, axis=-1)
    bboxes_td = tf.stack([
        cy - dy / 2,
        cx - dx / 2,
        cy + dy / 2,
        cx + dx / 2,
    ], axis=-1)  # pyformat: disable
    bboxes_td = py_utils.PadOrTrimTo(bboxes_td, [p.max_num_objects, 4])

    has_3d_info = tf.cast(_Dense(features['object/has_3d_info']), tf.float32)
    bboxes_3d_mask = py_utils.PadOrTrimTo(has_3d_info, [p.max_num_objects])
    bboxes_td_mask = bboxes_3d_mask

    # Fill in difficulties from bounding box height, truncation and occlusion.
    bb_height = ymax - ymin
    box_image_height = py_utils.PadOrTrimTo(bb_height, [p.max_num_objects])
    box_image_height *= bboxes_3d_mask

    # 0 to 3 indicating occlusion level. 0 means fully visible, 1 means partly,
    occlusion = tf.reshape(_Dense(features['object/occlusion']), [-1])
    occlusion = tf.cast(occlusion, tf.float32)
    occlusion = py_utils.PadOrTrimTo(occlusion, [p.max_num_objects])
    occlusion *= bboxes_3d_mask

    # Truncation: 0 -> not truncated, 1.0 -> truncated
    truncation = tf.reshape(_Dense(features['object/truncation']), [-1])
    truncation = py_utils.PadOrTrimTo(truncation, [p.max_num_objects])
    truncation *= bboxes_3d_mask

    difficulties = ComputeKITTIDifficulties(box_image_height, occlusion,
                                            truncation)
    difficulties = py_utils.PadOrTrimTo(difficulties, [p.max_num_objects])

    # Make a batch axis to call BBoxCorners, and take the first result back.
    bbox3d_corners = geometry.BBoxCorners(bboxes_3d[tf.newaxis, ...])[0]

    # Project the 3D bbox to the image plane.
    velo_to_image_plane = features['transform/velo_to_image_plane']
    bboxes3d_proj_to_image_plane = geometry.PointsToImagePlane(
        tf.reshape(bbox3d_corners, [-1, 3]), velo_to_image_plane)

    # Output is [num_objects, 8 corners per object, (x, y)].
    bboxes3d_proj_to_image_plane = tf.reshape(bboxes3d_proj_to_image_plane,
                                              [-1, 8, 2])
    bboxes3d_proj_to_image_plane = py_utils.PadOrTrimTo(
        bboxes3d_proj_to_image_plane, [p.max_num_objects, 8, 2])

    texts = features['object/label'].values
    labels = ops.static_map_string_int(x=texts, keys=self.KITTI_CLASS_NAMES)

    labels = py_utils.PadOrTrimTo(labels, [p.max_num_objects])
    texts = py_utils.PadOrTrimTo(texts, [p.max_num_objects])

    # Filter labels by setting bboxes_padding, bboxes_3d_mask, and
    # bboxes_td_mask appropriately.
    if p.filter_labels is not None:
      valid_labels = tf.constant([p.filter_labels])
      bbox_mask = tf.reduce_any(
          tf.equal(tf.expand_dims(labels, 1), valid_labels), axis=1)
      bbox_mask = tf.cast(bbox_mask, tf.float32)
      bboxes_padding = 1 - bbox_mask * (1 - bboxes_padding)
      filtered_bboxes_3d_mask = bboxes_3d_mask * bbox_mask
      bboxes_td_mask *= bbox_mask
    else:
      filtered_bboxes_3d_mask = bboxes_3d_mask

    # Placeholder for counting the number of laser points that reside within
    # each 3-d bounding box. This must be filled in outside of this function
    # based on the loaded 3-d laser points.
    bboxes_3d_num_points = tf.zeros([p.max_num_objects], dtype=tf.int32)
    bboxes_3d_num_points = py_utils.PadOrTrimTo(bboxes_3d_num_points,
                                                [p.max_num_objects])

    # Pad bboxes_3d.
    bboxes_3d = py_utils.PadOrTrimTo(bboxes_3d, [p.max_num_objects, 7])

    return py_utils.NestedMap(
        source_id=source_id,
        bboxes_count=bboxes_count,
        bboxes=bboxes,
        bboxes_padding=bboxes_padding,
        bboxes_3d=bboxes_3d,
        bboxes_3d_mask=filtered_bboxes_3d_mask,
        unfiltered_bboxes_3d_mask=bboxes_3d_mask,
        bboxes3d_proj_to_image_plane=bboxes3d_proj_to_image_plane,
        bboxes_td=bboxes_td,
        bboxes_td_mask=bboxes_td_mask,
        bboxes_3d_num_points=bboxes_3d_num_points,
        labels=labels,
        texts=texts,
        box_image_height=box_image_height,
        occlusion=occlusion,
        truncation=truncation,
        difficulties=difficulties)

  def Shape(self):
    p = self.params
    return py_utils.NestedMap(
        source_id=tf.TensorShape([]),
        bboxes_count=tf.TensorShape([]),
        bboxes=tf.TensorShape([p.max_num_objects, 4]),
        bboxes_padding=tf.TensorShape([p.max_num_objects]),
        bboxes_3d=tf.TensorShape([p.max_num_objects, 7]),
        bboxes_3d_mask=tf.TensorShape([p.max_num_objects]),
        unfiltered_bboxes_3d_mask=tf.TensorShape([p.max_num_objects]),
        bboxes3d_proj_to_image_plane=tf.TensorShape([p.max_num_objects, 8, 2]),
        bboxes_td=tf.TensorShape([p.max_num_objects, 4]),
        bboxes_td_mask=tf.TensorShape([p.max_num_objects]),
        bboxes_3d_num_points=tf.TensorShape([p.max_num_objects]),
        labels=tf.TensorShape([p.max_num_objects]),
        texts=tf.TensorShape([p.max_num_objects]),
        box_image_height=tf.TensorShape([p.max_num_objects]),
        occlusion=tf.TensorShape([p.max_num_objects]),
        truncation=tf.TensorShape([p.max_num_objects]),
        difficulties=tf.TensorShape([p.max_num_objects]))

  def DType(self):
    return py_utils.NestedMap(
        source_id=tf.string,
        bboxes_count=tf.int32,
        bboxes=tf.float32,
        bboxes_padding=tf.float32,
        bboxes_3d=tf.float32,
        bboxes_3d_mask=tf.float32,
        unfiltered_bboxes_3d_mask=tf.float32,
        bboxes3d_proj_to_image_plane=tf.float32,
        bboxes_td=tf.float32,
        bboxes_td_mask=tf.float32,
        bboxes_3d_num_points=tf.int32,
        labels=tf.int32,
        texts=tf.string,
        box_image_height=tf.float32,
        occlusion=tf.float32,
        truncation=tf.float32,
        difficulties=tf.int32)


class KITTIBase(input_extractor.BaseExtractor):
  """KITTI dataset base parameters."""

  @classmethod
  def Params(cls, *args, **kwargs):
    p = super().Params(*args, **kwargs)

    # Subclasses should set the following in file_datasource:
    # - file_pattern_prefix: path to data directory (may be overridden at
    #   runtime)
    # - file_pattern: file pattern of records relative to the
    #   data directory
    p.file_datasource = datasource.PrefixedDataSource.Params()
    p.file_datasource.file_type = 'tfrecord'

    return p

  @property
  def class_names(self):
    return KITTILabelExtractor.KITTI_CLASS_NAMES


class KITTILaser(KITTIBase):
  """KITTI object detection dataset.

  This class emits KITTI images, labels, and the raw laser
  representation of the data.  See KITTIGrid and KITTISparse
  for alternative laser representations.

  Input batch contains outputs from:
    - KITTIImageExtractor
    - KITTILabelExtractor
    - KITTILaserExtractor
  """

  @classmethod
  def Params(cls):
    """Defaults params."""
    extractors = hyperparams.Params()
    extractors.Define('labels', KITTILabelExtractor.Params(), '')
    extractors.Define('lasers', KITTILaserExtractor.Params(), '')
    extractors.Define('images', KITTIImageExtractor.Params(), '')
    preprocessors = py_utils.NestedMap(
        count_points=input_preprocessors.CountNumberOfPointsInBoxes3D.Params(),
        viz_copy=input_preprocessors.CreateDecoderCopy.Params(),
        pad_lasers=input_preprocessors.PadLaserFeatures.Params().Set(
            max_num_points=128500))

    p = super().Params(extractors).Set(
        preprocessors=_NestedMapToParams(preprocessors),
        preprocessors_order=['viz_copy', 'count_points', 'pad_lasers'])
    return p

  @property
  def class_names(self):
    return KITTILabelExtractor.KITTI_CLASS_NAMES


class KITTISparseLaser(KITTIBase):
  """KITTI object detection dataset for sparse detection models.

  This class emits KITTI images, labels, and the sparse laser
  representation of the data.  See KITTIGrid and KITTISparse
  for alternative laser representations.

  Input batch contains outputs from:
    - KITTILabelExtractor
    - KITTILaserExtractor

  Transformed with:
   - Metadata annotation:
     - CountNumberOfPointsInBoxes3D
   - Visualization:
     - CreateDecoderCopy
   - Sparse gather of points for featurization:
     - SparseCenterSelector
     - SparseCellGatherFeatures
   - Anchor creation for classification regression targets:
     - TileAnchorBBoxes
     - AnchorAssignment
  """

  @classmethod
  def Params(cls):
    """Defaults params."""
    extractors = hyperparams.Params()
    extractors.Define('labels', KITTILabelExtractor.Params(), '')
    extractors.Define('lasers', KITTILaserExtractor.Params(), '')
    extractors.Define('images', KITTIImageExtractor.Params(), '')
    preprocessors = py_utils.NestedMap(
        count_points=input_preprocessors.CountNumberOfPointsInBoxes3D.Params(),
        viz_copy=input_preprocessors.CreateDecoderCopy.Params(),
        keep_xyz_range=input_preprocessors.DropLaserPointsOutOfRange.Params(),
        select_centers=input_preprocessors.SparseCenterSelector.Params(),
        gather_features=input_preprocessors.SparseCellGatherFeatures.Params(),
        tile_anchors=input_preprocessors.TileAnchorBBoxes.Params(),
        assign_anchors=input_preprocessors.AnchorAssignment.Params(),
        pad_lasers=input_preprocessors.PadLaserFeatures.Params().Set(),
    )

    p = super().Params(extractors).Set(
        preprocessors=_NestedMapToParams(preprocessors),
        preprocessors_order=[
            'viz_copy',
            'keep_xyz_range',
            'count_points',
            'select_centers',
            'gather_features',
            'tile_anchors',
            'assign_anchors',
            'pad_lasers',
        ],
    )
    return p


class KITTIGrid(KITTIBase):
  """KITTI object detection dataset.

  This class emits KITTI images, labels, and the fixed grid laser
  representation of the data.

  Input batch contains outputs from:
    - KITTILabelExtractor
    - KITTILaserExtractor

  Transformed with:
   - Metadata annotation:
     - CountNumberOfPointsInBoxes3D
   - Visualization:
     - CreateDecoderCopy
   - Points to Pillars
     - PointsToGrid
     - GridToPillars
   - Anchor creation for classification regression targets:
     - GridAnchorCenters
     - TileAnchorBBoxes
     - AnchorAssignment
  """

  @classmethod
  def Params(cls):
    """Defaults params."""
    extractors = hyperparams.Params()
    extractors.Define('labels', KITTILabelExtractor.Params(), '')
    extractors.Define('lasers', KITTILaserExtractor.Params(), '')
    preprocessors = py_utils.NestedMap(
        count_points=input_preprocessors.CountNumberOfPointsInBoxes3D.Params(),
        viz_copy=input_preprocessors.CreateDecoderCopy.Params(),
        points_to_grid=input_preprocessors.PointsToGrid.Params(),
        grid_to_pillars=input_preprocessors.GridToPillars.Params(),
        grid_anchor_centers=input_preprocessors.GridAnchorCenters.Params(),
        tile_anchors=input_preprocessors.TileAnchorBBoxes.Params(),
        assign_anchors=input_preprocessors.AnchorAssignment.Params(),
        pad_lasers=input_preprocessors.PadLaserFeatures.Params().Set(
            max_num_points=128500),
    )
    p = super().Params(extractors).Set(
        preprocessors=_NestedMapToParams(preprocessors),
        preprocessors_order=[
            'viz_copy',
            'count_points',
            'points_to_grid',
            'grid_to_pillars',
            'grid_anchor_centers',
            'tile_anchors',
            'assign_anchors',
            'pad_lasers',
        ],
    )
    return p
