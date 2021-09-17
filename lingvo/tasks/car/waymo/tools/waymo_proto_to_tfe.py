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
r"""Library to convert Waymo Open Dataset to tf.Examples.

Generates a tf.Example proto for every dataset_pb2.Frame containing
the following keys and their values:

* Frame-level metadata

run_segment: string - The identifier of the driving sequence in the dataset.

run_start_offset: int64 - The start offset within the run_segment sequence.

time_of_day: string - Categorical description of time of day, e.g., "Day".

location: string - Categorical description of geographical location, e.g.,
"location_sf".

weather: string - Categorical description of weather of scene, e.g., "sunny".

pose: float: 4x4 transformation matrix for converting from "world" coordinates
to SDC coordinates.

* Lasers

There are 5 LIDAR sensors: "TOP", "SIDE_LEFT", "SIDE_RIGHT", "FRONT", "REAR".
Each LIDAR currently provides two returns, "ri1" and "ri2" for the first and
second returns of each shot.

For every $LASER and $RI, we embed the raw range image:

$LASER_$RI: float - flattened range image data of shape [H, W, C] from the
original proto.

$LASER_$RI_shape: int64 - shape of the range image.

For every lidar $LASER, we extract the calibrations:

$LASER_beam_inclinations: float - List of beam angle inclinations for TOP
LIDAR (non-uniform).

$LASER_beam_inclination_min: float - Minimum beam inclination for uniform
LIDARs.

$LASER_beam_inclination_max: float - Maximum beam inclination for uniform
LIDARs.

$LASER_extrinsics: float - 4x4 transformation matrix for converting from
SDC coordinates to LIDAR coordinates.

The TOP LIDAR currently has a per-pixel range image pose to accommodate for
rolling shutter effects when projecting to 3D cartesian coordinates.  We
embed this range image pose as TOP_pose.

To allow for easier use, we also project all $LASERs to a stacked 3D cartesian
coordinate point cloud as:

laser_$LASER_$RI - float: An [N, 6] matrix where there are N total points,
the first three dimensions are the x, y, z caresian coordinates, and the last
three dimensions are the intensity, elongation, and "is_in_no_label_zone"
bit for each point.

* Camera images

There are 5 cameras in the dataset: "FRONT", "FRONT_LEFT", "FRONT_RIGHT",
"SIDE_LEFT", and "SIDE_RIGHT".

For each $CAM, we store:

image_$CAM: string - Scalar Png format camera image.

image_$CAM_shape: int64 - [3] - Vector containing the shape of the camera
image as [height, width, channels].

image_$CAM_pose: float - [4, 4] Matrix transformation for converting from
world coordinates to camera center.

image_$CAM_pose: float - Scalar timestamp offset of when image was taken.

image_$CAM_shutter: float - Scalar shutter value.

image_$CAM_velocity: float - [6] Vector describing velocity of camera for
rolling shutter adjustment.  See original proto for details.

image_%CAM_camera_trigger_time: Scalar float for when camera was triggered.

image_$CAM_camera_readout_done_time: Scalar float for when camera image finished
reading out data.

camera_$CAM_extrinsics: float - 4x4 pose transformation for converting from
camera center coordinates to 2d projected view.

camera_$CAM_intrinsics: float - [9] intrinsics transformation for converting
from camera center coordinates to 2d projected view.

camera_$CAM_width: int64 - Scalar width of image.

camera_$CAM_height: int64 - Scalar height of image.

camera_$CAM_rolling_shutter_direction: int64 - Scalar value indicating the
direction of the rolling shutter adjustment.

* Labels

For each frame, we store the following label information for the M bounding
boxes in the frame.

labels: int64 - [M] - The integer label class for every 3D bounding box
corresponding to the enumeration defined in the proto.

label_ids: string - [M] - The unique label string identifying each labeled
object. This can be used for associating the same object across frames of the
same run segment.

bboxes_3d: float - A flattened [M, 7] matrix where there are M boxes in the
frame, and each box is defined by a 7-DOF format - [center_x ,center_y,
center_z, length, width, height, heading].

label_metadata: floating point - A flattened [M, 4] matrix where there are
M boxes in the frame, and each md entry is the [speed_x, speed_y, accel_x,
accel_y] of the object.

bboxes_3d_num_points: int64 - [M] - The number of points that fall into each
3D bounding box: can be used for computing the difficulty of each bounding
box.

detection_difficulties: int64 - DO NOT USE FOR EVALUATION. Indicates whether the
labelers have determined that the object is of LEVEL_2 difficulty.
Should be used jointly with num_points above to set the difficulty level,
which we save in `single_frame_detection_difficulties`. Because it does not
include information about the number of points in its calculation,
it is an incomplete definition of difficulty and will not correspond to the
leaderboard if used to calculate metrics.

single_frame_detection_difficulties: int64 - Indicates the difficulty level as
either LEVEL_1 (1), or LEVEL_2 (2), or IGNORE (999). We first ignore all 3D
labels without any LiDAR points. Next, we assign LEVEL_2 to examples where
either the labeler annotates as hard or if the example has <= 5 LiDAR points.
Finally, the rest of the examples are assigned to LEVEL_1.

tracking_difficulties: int64 - Indicates whether the labelers have determined
that the tracked object is of LEVEL_2 difficulty.

nlz_proto_strs: string - Vector of NoLabelZone polygon protos.  Currently
unused.
"""

import zlib

import apache_beam as beam
from lingvo import compat as tf
from lingvo.core import py_utils
import numpy as np
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils


class FrameToTFE(object):
  """Converter utility from car.open_dataset.Frame to tf.Examples."""

  def __init__(self, use_range_image_index_as_lidar_feature=None):
    """FrameToTFE.

    Args:
      use_range_image_index_as_lidar_feature: If True, the lidar point feature
        is filled with the (row index, col index, NLZ) instead of (intensity,
        elongation, NLZ). This is used _only_ for data processing code to
        construct another range image that's parallel to the main range image
        (e.g., the sceneflow range image).
    """
    self._use_range_image_index_as_lidar_feature = (
        use_range_image_index_as_lidar_feature)

  def process(self, item):
    """Convert 'item' into tf.Example format."""
    assert isinstance(item, dataset_pb2.Frame)
    output = tf.train.Example()
    feature = output.features.feature

    # Convert run segment
    run_segment = item.context.name
    run_start_offset = item.timestamp_micros
    key = run_segment + '_' + str(run_start_offset)
    feature['run_segment'].bytes_list.value[:] = [
        tf.compat.as_bytes(run_segment)
    ]
    feature['run_start_offset'].int64_list.value[:] = [run_start_offset]

    # Extract metadata about frame.
    feature['time_of_day'].bytes_list.value[:] = [
        tf.compat.as_bytes(item.context.stats.time_of_day)
    ]
    feature['location'].bytes_list.value[:] = [
        tf.compat.as_bytes(item.context.stats.location)
    ]
    feature['weather'].bytes_list.value[:] = [
        tf.compat.as_bytes(item.context.stats.weather)
    ]

    # Convert pose: a 4x4 transformation matrix.
    feature['pose'].float_list.value[:] = list(item.pose.transform)
    self.frame_pose = tf.convert_to_tensor(
        np.reshape(np.array(item.pose.transform), [4, 4]), dtype=tf.float32)

    # Extract laser names.
    laser_names = []
    for laser in item.lasers:
      laser_name = laser.name
      real_name = dataset_pb2.LaserName.Name.Name(laser_name)
      laser_names += [real_name]

    # Extract laser data (range images) and the calibrations.
    self.extract_lasers(feature, item.lasers)

    self.extract_laser_calibrations(feature, item.context.laser_calibrations)

    range_image_pose = self._get_range_image_pose(item.lasers)
    feature['TOP_pose'].float_list.value[:] = range_image_pose.numpy().reshape(
        [-1])

    # From the range images, also turn them into 3D point clouds.
    self.add_point_cloud(feature, laser_names, range_image_pose)

    self.add_labels(feature, item.laser_labels)
    self.add_no_label_zones(feature, item.no_label_zones)

    camera_calibrations_dict = ({
        camera_calibration.name: camera_calibration
        for camera_calibration in item.context.camera_calibrations
    })
    # Extract camera image data and the calibrations.
    self.extract_camera_images(feature, item.images, camera_calibrations_dict)
    self.extract_camera_calibrations(feature, camera_calibrations_dict.values())

    return key, output

  def _get_range_image_pose(self, lasers):
    """Fetches the per-pixel pose information for the range image."""
    range_image_top_pose = None
    for laser in lasers:
      if laser.name != dataset_pb2.LaserName.TOP:
        continue
      pose_str = zlib.decompress(laser.ri_return1.range_image_pose_compressed)
      # Deserialize from MatrixFloat serialization.
      range_image_top_pose = dataset_pb2.MatrixFloat()
      range_image_top_pose.ParseFromString(pose_str)

    assert range_image_top_pose is not None
    shape = list(range_image_top_pose.shape.dims)
    range_image_top_pose_tensor = np.array(
        range_image_top_pose.data).reshape(shape)
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1], range_image_top_pose_tensor[...,
                                                                         2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[...,
                                                                          3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    assert range_image_top_pose_tensor.shape == (64, 2650, 4, 4)
    return range_image_top_pose_tensor

  def _parse_range_image(self, range_image):
    """Parse range_image proto and convert to MatrixFloat form."""
    if range_image.range_image_compressed:
      ri_str = zlib.decompress(range_image.range_image_compressed)
      # Deserialize from MatrixFloat serialization.
      ri = dataset_pb2.MatrixFloat()
      ri.ParseFromString(ri_str)
    else:
      ri = range_image.range_image
    if range_image.range_image_flow_compressed:
      ri_str = zlib.decompress(range_image.range_image_flow_compressed)
      # Deserialize from MatrixFloat serialization.
      ri_flow = dataset_pb2.MatrixFloat()
      ri_flow.ParseFromString(ri_str)
    else:
      ri_flow = None
    return ri, ri_flow

  def extract_camera_images(self,
                            feature,
                            camera_images,
                            camera_calibrations_dict,
                            include_image_bytes=True):
    """Extract the images into the tf.Example feature map.

    Args:
      feature: A tf.Example feature map.
      camera_images: A repeated car.open_dataset.CameraImage proto.
      camera_calibrations_dict: A dictionary maps camera name to
        car.open_dataset.CameraCalibration proto.
      include_image_bytes: Whether to add the image bytes to the output.
    """
    for camera_image in camera_images:
      camera_name = camera_image.name
      camera_calibration = camera_calibrations_dict[camera_name]
      real_name = dataset_pb2.CameraName.Name.Name(camera_name)
      if include_image_bytes:
        feature['image_%s' % real_name].bytes_list.value[:] = [
            tf.compat.as_bytes(camera_image.image)
        ]
      feature['image_%s_shape' % real_name].int64_list.value[:] = ([
          camera_calibration.height, camera_calibration.width, 3
      ])
      feature['image_%s_pose' % real_name].float_list.value[:] = (
          list(camera_image.pose.transform))
      velocity = camera_image.velocity
      feature['image_%s_velocity' % real_name].float_list.value[:] = ([
          velocity.v_x, velocity.v_y, velocity.v_z, velocity.w_x, velocity.w_y,
          velocity.w_z
      ])
      feature['image_%s_pose_timestamp' %
              real_name].float_list.value[:] = ([camera_image.pose_timestamp])
      feature['image_%s_shutter' %
              real_name].float_list.value[:] = ([camera_image.shutter])
      feature['image_%s_camera_trigger_time' %
              real_name].float_list.value[:] = ([
                  camera_image.camera_trigger_time
              ])
      feature['image_%s_camera_readout_done_time' %
              real_name].float_list.value[:] = ([
                  camera_image.camera_readout_done_time
              ])

  def extract_camera_calibrations(self, feature, camera_calibrations):
    """Extract the camera calibrations into the tf.Example feature map.

    Args:
      feature: A tf.Example feature map.
      camera_calibrations: A CameraCalibration proto from the Waymo Dataset.
    """
    for camera_calibration in camera_calibrations:
      camera_name = camera_calibration.name
      real_name = dataset_pb2.CameraName.Name.Name(camera_name)

      feature['camera_%s_extrinsics' % real_name].float_list.value[:] = list(
          camera_calibration.extrinsic.transform)
      feature['camera_%s_intrinsics' % real_name].float_list.value[:] = list(
          camera_calibration.intrinsic)
      feature['camera_%s_width' %
              real_name].int64_list.value[:] = [camera_calibration.width]
      feature['camera_%s_height' %
              real_name].int64_list.value[:] = [camera_calibration.height]
      feature['camera_%s_rolling_shutter_direction' %
              real_name].int64_list.value[:] = [
                  camera_calibration.rolling_shutter_direction
              ]

  def extract_lasers(self, feature, lasers):
    """Extract the lasers from range_images into the tf.Example feature map.

    Args:
      feature: A tf.Example feature map.
      lasers: A repeated car.open_dataset.Laser proto.
    """
    for laser in lasers:
      ri1, ri1_flow = self._parse_range_image(laser.ri_return1)
      ri2, ri2_flow = self._parse_range_image(laser.ri_return2)

      # Add the range image data (flattened) and their original shape
      # to the output feature map.
      laser_name = laser.name
      real_name = dataset_pb2.LaserName.Name.Name(laser_name)
      feature['%s_ri1' % real_name].float_list.value[:] = ri1.data
      feature['%s_ri1_shape' % real_name].int64_list.value[:] = ri1.shape.dims
      feature['%s_ri2' % real_name].float_list.value[:] = ri2.data
      feature['%s_ri2_shape' % real_name].int64_list.value[:] = ri2.shape.dims
      if ri1_flow:
        feature['%s_ri1_flow' % real_name].float_list.value[:] = ri1_flow.data
        feature['%s_ri1_flow_shape' %
                real_name].int64_list.value[:] = ri1_flow.shape.dims
      if ri2_flow:
        feature['%s_ri2_flow' % real_name].float_list.value[:] = ri2_flow.data
        feature['%s_ri2_flow_shape' %
                real_name].int64_list.value[:] = ri2_flow.shape.dims

  def extract_laser_calibrations(self, feature, laser_calibrations):
    """Extract the laser calibrations into the tf.Example feature map.

    Args:
      feature: A tf.Example feature map.
      laser_calibrations: A LaserCalibrations proto from the Waymo Dataset.
    """
    for laser_calibration in laser_calibrations:
      laser_name = laser_calibration.name
      real_name = dataset_pb2.LaserName.Name.Name(laser_name)
      feature['%s_beam_inclinations' % real_name].float_list.value[:] = (
          laser_calibration.beam_inclinations)
      feature['%s_beam_inclination_min' % real_name].float_list.value[:] = ([
          laser_calibration.beam_inclination_min
      ])
      feature['%s_beam_inclination_max' % real_name].float_list.value[:] = ([
          laser_calibration.beam_inclination_max
      ])
      feature['%s_extrinsics' % real_name].float_list.value[:] = list(
          laser_calibration.extrinsic.transform)

  def add_point_cloud(self, feature, laser_names, range_image_pose):
    """Convert the range images in `feature` to 3D point clouds.

    Adds the point cloud data to the tf.Example feature map.

    Args:
      feature: A tf.Example feature map.
      laser_names: A list of laser names (e.g., 'TOP', 'REAR', 'SIDE_LEFT').
      range_image_pose: A range image pose Tensor for the top laser.
    """
    # Stash metadata for laser. These metadata can be useful
    # for reconstructing the range image.
    self.laser_info = {}

    for laser_name in laser_names:
      beam_inclinations = np.array(feature['%s_beam_inclinations' %
                                           laser_name].float_list.value[:])
      # beam_inclinations will be populated if there is a non-uniform
      # beam configuration (e.g., for the TOP lasers).  Others that have
      # uniform beam inclinations are only parameterized by the min and max.
      # We use these min and max if the beam_inclinations are not present,
      # and turn them into a uniform inclinations array.
      if beam_inclinations.size == 0:
        beam_inclination_min = feature['%s_beam_inclination_min' %
                                       laser_name].float_list.value[:]
        beam_inclination_max = feature['%s_beam_inclination_max' %
                                       laser_name].float_list.value[:]

        laser_ri_name = '%s_ri1' % laser_name
        range_image_shape = feature[laser_ri_name +
                                    '_shape'].int64_list.value[:]
        height = tf.cast(range_image_shape[0], tf.float32)

        beam_inclinations = tf.constant(
            [beam_inclination_min[0], beam_inclination_max[0]])
        beam_inclinations = range_image_utils.compute_inclination(
            beam_inclinations, height)

      beam_extrinsics = np.array(
          feature['%s_extrinsics' % laser_name].float_list.value[:]).reshape(
              4, 4)

      for ri_type in ['ri1', 'ri2']:
        laser_ri_name = '%s_%s' % (laser_name, ri_type)
        # For each of the 4 features of the lasers:
        range_image = np.array(feature[laser_ri_name].float_list.value[:])
        range_image_shape = feature[laser_ri_name +
                                    '_shape'].int64_list.value[:]
        range_image = range_image.reshape(range_image_shape)
        # Compute mask.  At the moment, invalid values in the range image
        # representation are indicated via a -1. entry.  Callers are expected
        # to create this mask when passing into the conversion function below.
        range_image_mask = range_image[..., 0] >= 0

        # Get the 'range' feature from the range images.
        range_image_range = range_image[..., 0]

        # Call utility to convert point cloud to cartesian coordinates.
        #
        # API expects a batch dimension for all inputs.
        batched_pixel_pose = None
        batched_frame_pose = None
        # At the moment, only the top has per-pixel pose.
        if laser_name == 'TOP':
          batched_pixel_pose = range_image_pose[tf.newaxis, ...]
          batched_frame_pose = self.frame_pose[tf.newaxis, ...]

        batched_range_image_range = tf.convert_to_tensor(
            range_image_range[np.newaxis, ...], dtype=tf.float32)
        batched_extrinsics = tf.convert_to_tensor(
            beam_extrinsics[np.newaxis, ...], dtype=tf.float32)
        batched_inclinations = tf.convert_to_tensor(
            beam_inclinations[np.newaxis, ...], dtype=tf.float32)

        batched_inclinations = tf.reverse(batched_inclinations, axis=[-1])

        range_image_cartesian = (
            range_image_utils.extract_point_cloud_from_range_image(
                batched_range_image_range,
                batched_extrinsics,
                batched_inclinations,
                pixel_pose=batched_pixel_pose,
                frame_pose=batched_frame_pose))

        info = py_utils.NestedMap()
        self.laser_info[laser_ri_name] = info
        info.range_image = range_image
        info.range_image_shape = range_image_shape

        ri_indices = tf.where(range_image_mask)
        points_xyz = tf.gather_nd(range_image_cartesian[0], ri_indices)
        info.num_points = tf.shape(points_xyz).numpy()[0]

        # Fetch the features corresponding to each xyz coordinate and
        # concatentate them together.
        points_features = tf.cast(
            tf.gather_nd(range_image[..., 1:], ri_indices), tf.float32)
        if self._use_range_image_index_as_lidar_feature:
          points_data = tf.concat([
              points_xyz,
              tf.cast(ri_indices, tf.float32), points_features[..., 2:]
          ],
                                  axis=-1)
        else:
          points_data = tf.concat([points_xyz, points_features], axis=-1)

        # Add laser feature to output.
        #
        # Skip embedding shape since we assume that all points have six features
        # and so we can reconstruct the number of points.
        points_list = list(points_data.numpy().reshape([-1]))
        feature['laser_%s' % laser_ri_name].float_list.value[:] = points_list

        laser_ri_flow_name = '%s_flow' % laser_ri_name
        if laser_ri_flow_name in feature:
          range_image_flow = np.array(
              feature[laser_ri_flow_name].float_list.value[:])
          range_image_flow_shape = feature[laser_ri_flow_name +
                                           '_shape'].int64_list.value[:]
          range_image_flow = range_image_flow.reshape(range_image_flow_shape)
          flow_data = tf.cast(
              tf.gather_nd(range_image_flow, ri_indices), tf.float32)
          flow_list = list(flow_data.numpy().reshape([-1]))
          feature['laser_%s' %
                  laser_ri_flow_name].float_list.value[:] = flow_list

  def _single_frame_detection_difficulty(self, human_difficulty, num_points):
    """Create the `single_frame_detection_difficulty` field.

    When labeling, humans have the option to label a particular frame's bbox
    as difficult, which overrides the normal number of points based
    definition. Additionally, boxes with 0 points are ignored by the metric
    code.

    Args:
      human_difficulty: What the human raters labeled the difficulty as. This is
        from the detection_difficulty_level field, and will be either 0 (default
        value, which is UKNOWN in the proto enum) or 2 (LEVEL_2 difficulty).
      num_points: The number of points in the bbox.

    Returns:
      single_frame_detection_difficulty: The single frame detection difficulty
        per the Waymo Open Dataset paper's definition.
    """
    if num_points <= 0:
      return 999

    if human_difficulty:
      return human_difficulty

    if num_points <= 5:
      return 2
    else:
      return 1

  def add_labels(self, feature, labels):
    """Add 3d bounding box labels into the output feature map.

    Args:
      feature: A tf.Example feature map.
      labels: A repeated car.open_dataset.Label proto.
    """
    label_classes = []
    label_ids = []
    detection_difficulty_levels = []
    tracking_difficulty_levels = []
    bboxes_3d_num_points = []
    single_frame_detection_difficulty_levels = []
    bboxes = []
    label_md = []

    for label in labels:
      box = label.box
      bbox_3d = [
          box.center_x, box.center_y, box.center_z, box.length, box.width,
          box.height, box.heading
      ]
      md = [
          label.metadata.speed_x, label.metadata.speed_y,
          label.metadata.accel_x, label.metadata.accel_y
      ]
      label_md += md
      bboxes += bbox_3d
      label_classes += [label.type]
      label_ids += [tf.compat.as_bytes(label.id)]
      detection_difficulty_levels += [label.detection_difficulty_level]
      tracking_difficulty_levels += [label.tracking_difficulty_level]
      bboxes_3d_num_points += [label.num_lidar_points_in_box]

      # Compute the single frame difficulty level per object.
      human_labeler_difficulty = label.detection_difficulty_level
      num_points = bboxes_3d_num_points[-1]
      single_frame_detection_difficulty = (
          self._single_frame_detection_difficulty(human_labeler_difficulty,
                                                  num_points))
      single_frame_detection_difficulty_levels += [
          single_frame_detection_difficulty
      ]

    bboxes = np.array(bboxes).reshape(-1)
    label_md = np.array(label_md).reshape(-1)
    feature['labels'].int64_list.value[:] = label_classes
    feature['label_ids'].bytes_list.value[:] = label_ids
    feature['detection_difficulties'].int64_list.value[:] = (
        detection_difficulty_levels)
    feature['single_frame_detection_difficulties'].int64_list.value[:] = (
        single_frame_detection_difficulty_levels)
    feature['tracking_difficulties'].int64_list.value[:] = (
        tracking_difficulty_levels)
    feature['bboxes_3d'].float_list.value[:] = list(bboxes)
    feature['label_metadata'].float_list.value[:] = list(label_md)
    feature['bboxes_3d_num_points'].int64_list.value[:] = (bboxes_3d_num_points)

  def add_no_label_zones(self, feature, no_label_zones):
    """Add no label zones into the output feature map.

    Args:
      feature: A tf.Example feature map.
      no_label_zones: A repeated car.open_dataset.Polygon2dProto proto.
    """
    nlz_proto_strs = []
    for nlz in no_label_zones:
      nlz_proto_strs += [tf.compat.as_bytes(nlz.SerializeToString())]
    feature['no_label_zones'].bytes_list.value[:] = nlz_proto_strs


class WaymoOpenDatasetConverter(beam.DoFn):
  """Converts WaymoOpenDataset into tf.Examples.  See file docstring."""

  def __init__(self, emitter_fn):
    self._emitter_fn = emitter_fn
    self._converter = FrameToTFE()

  def process(self, item):
    key, output = self._converter.process(item)
    return self._emitter_fn(key, output)
