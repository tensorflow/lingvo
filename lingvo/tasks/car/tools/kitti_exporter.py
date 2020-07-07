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
"""Create TFRecords files from KITTI raw data.

Parses KITTI raw data with different splits, indicated with split files.
A split file is a text file that specifies frame names to be included in the
split, with one name per line. Splits with 'test' in the filename use testing
data, while other splits use training data. This program expects KITTI raw data
in the following directory structure:

  kitti_object/
      training/  # Contains KITTI raw train data
          label2/
          velodyne/
          calib/
          image_2/
      testing/  # Contains KITTI raw test data
          velodyne/
          calib/
          image_2/
      splits/  # Contains split files identifying frame names in the split.
          split_name.txt

Outputs examples in TFRecords files correspond to KITTI frames with the
following format:

  # frame information
  image/source_id: unique frame name e.g '000000', '000010'

  # 2D image data
  image/encoded: PNG encoded string
  image/height: image height
  image/width: image width
  image/format: 'PNG'

  # 3D velodyne pointcloud data (variable P points per frame)
  pointcloud/xyz = point positions (P x 3 tensor).
  pointcloud/reflectance: point reflectances (P x 1 tensor).

  # Object level data (variable N objects per frame)
  object/image/bbox/xmin: min X pixel location in raw image (N x 1 tensor).

  object/image/bbox/xmax: max X pixel location in raw image (N x 1 tensor).

  object/image/bbox/ymin: min Y pixel location in raw image (N x 1 tensor).

  object/image/bbox/ymax: max Y pixel location in raw image (N x 1 tensor).

  object/label: one of {'Car', 'Pedestrian', 'Cyclist'} identifying object
  class (N x 1 tensor).

  object/has_3d_info: 1 if object has valid 3D info else 0 (N x 1 tensor).

  object/occlusion: int in {0, 1, 2, 3} of occlusion state (N x 1 tensor).

  object/truncation: float in 0 (non-truncated) to 1 (truncated) (N x 1 tensor).

  object/velo/bbox/xyz: 3D bbox locations in velo frame (N x 3 tensor).

  object/velo/bbox/dim_xyz: length (dx), width (dy), height (dz) indicating
  object dimensions (N x 3 tensor).

  object/velo/bbox/phi: bbox rotation in velo frame (N x 1 tensor).

  # Transformation matrices
  transform/velo_to_image_plane: 3x4 matrix from velo xyz to image plane xy.
  After multiplication, you need to divide by last coordinate to recover 2D
  pixel locations.

  transform/velo_to_camera: 4x4 matrix from velo xyz to camera xyz.

  transform/camera_to_velo 4x4 matrix from camera xyz to velo xyz.
"""

import contextlib
import io
import os

from absl import app
from absl import flags
from absl import logging

from lingvo import compat as tf
from lingvo.tasks.car.tools import kitti_data
import numpy as np
from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_string('kitti_object_dir', None,
                    'Path to a kitti object directory.')
flags.DEFINE_string('split', None,
                    'Name of the split file to be used for parsing.')
flags.DEFINE_string('tfrecord_path', None, 'Output tfrecord path.')
flags.DEFINE_integer(
    'num_shards', 1, 'Number of output shards (between 1 and 99999). Files'
    'named {tfrecord_path}-{shard_num}-of-{total_shards}.')


def _ReadObjectDataset(root_dir, frame_names):
  """Reads and parses KITTI dataset files into a list of TFExample protos."""
  examples = []

  total_frames = len(frame_names)
  for frame_index, frame_name in enumerate(frame_names):
    image_file_path = os.path.join(root_dir, 'image_2', frame_name + '.png')
    calib_file_path = os.path.join(root_dir, 'calib', frame_name + '.txt')
    velo_file_path = os.path.join(root_dir, 'velodyne', frame_name + '.bin')
    label_file_path = os.path.join(root_dir, 'label_2', frame_name + '.txt')

    example = tf.train.Example()
    feature = example.features.feature

    # frame information
    feature['image/source_id'].bytes_list.value[:] = [frame_name]

    # 2D image data
    encoded_image = tf.io.gfile.GFile(image_file_path).read()
    feature['image/encoded'].bytes_list.value[:] = [encoded_image]
    image = np.array(Image.open(io.BytesIO(encoded_image)))
    assert image.ndim == 3
    assert image.shape[2] == 3
    image_width = image.shape[1]
    image_height = image.shape[0]
    feature['image/width'].int64_list.value[:] = [image_width]
    feature['image/height'].int64_list.value[:] = [image_height]
    feature['image/format'].bytes_list.value[:] = ['PNG']

    # 3D velodyne point data
    velo_dict = kitti_data.LoadVeloBinFile(velo_file_path)
    point_list = velo_dict['xyz'].ravel().tolist()
    feature['pointcloud/xyz'].float_list.value[:] = point_list
    reflectance_list = velo_dict['reflectance'].ravel().tolist()
    feature['pointcloud/reflectance'].float_list.value[:] = reflectance_list

    # Object data
    calib_dict = kitti_data.LoadCalibrationFile(calib_file_path)
    if tf.io.gfile.exists(label_file_path):
      # Load object labels for training data
      object_dicts = kitti_data.LoadLabelFile(label_file_path)
      object_dicts = kitti_data.AnnotateKITTIObjectsWithBBox3D(
          object_dicts, calib_dict)
    else:
      # No object labels for test data
      object_dicts = {}

    num_objects = len(object_dicts)
    xmins = [None] * num_objects
    xmaxs = [None] * num_objects
    ymins = [None] * num_objects
    ymaxs = [None] * num_objects
    labels = [None] * num_objects
    has_3d_infos = [None] * num_objects

    # 3D info
    occlusions = [None] * num_objects
    truncations = [None] * num_objects
    xyzs = [None] * num_objects
    dim_xyzs = [None] * num_objects
    phis = [None] * num_objects

    for object_index, object_dict in enumerate(object_dicts):
      xmins[object_index] = object_dict['bbox'][0]
      xmaxs[object_index] = object_dict['bbox'][2]
      ymins[object_index] = object_dict['bbox'][1]
      ymaxs[object_index] = object_dict['bbox'][3]
      labels[object_index] = object_dict['type']
      has_3d_infos[object_index] = 1 if object_dict['has_3d_info'] else 0
      occlusions[object_index] = object_dict['occluded']
      truncations[object_index] = object_dict['truncated']
      xyzs[object_index] = object_dict['bbox3d'][:3]
      dim_xyzs[object_index] = object_dict['bbox3d'][3:6]
      phis[object_index] = object_dict['bbox3d'][6]

    feature['object/image/bbox/xmin'].float_list.value[:] = xmins
    feature['object/image/bbox/xmax'].float_list.value[:] = xmaxs
    feature['object/image/bbox/ymin'].float_list.value[:] = ymins
    feature['object/image/bbox/ymax'].float_list.value[:] = ymaxs
    feature['object/label'].bytes_list.value[:] = labels
    feature['object/has_3d_info'].int64_list.value[:] = has_3d_infos
    feature['object/occlusion'].int64_list.value[:] = occlusions
    feature['object/truncation'].float_list.value[:] = truncations
    xyzs = np.array(xyzs).ravel().tolist()
    feature['object/velo/bbox/xyz'].float_list.value[:] = xyzs
    dim_xyzs = np.array(dim_xyzs).ravel().tolist()
    feature['object/velo/bbox/dim_xyz'].float_list.value[:] = dim_xyzs
    feature['object/velo/bbox/phi'].float_list.value[:] = phis

    # Transformation matrices
    velo_to_image_plane = kitti_data.VeloToImagePlaneTransformation(calib_dict)
    feature['transform/velo_to_image_plane'].float_list.value[:] = (
        velo_to_image_plane.ravel().tolist())
    velo_to_camera = kitti_data.VeloToCameraTransformation(calib_dict)
    feature['transform/velo_to_camera'].float_list.value[:] = (
        velo_to_camera.ravel().tolist())
    cam_to_velo = kitti_data.CameraToVeloTransformation(calib_dict)
    feature['transform/camera_to_velo'].float_list.value[:] = (
        cam_to_velo.ravel().tolist())

    examples.append(example)
    if frame_index % 100 == 0:
      logging.info('Processed frame %d of %d.', frame_index, total_frames)

  return examples


def _ExportObjectDatasetToTFRecord(root_dir, split_file, tfrecord_path,
                                   num_shards):
  """Exports KITTI dataset files to TFRecord files."""
  if num_shards <= 0:
    raise ValueError('TFRecord dataset must have at least one shard.')

  logging.info('Reading frame names from split_file %s.', split_file)
  frame_names = [line.rstrip('\n') for line in tf.io.gfile.GFile(split_file)]
  logging.info('Reading object dataset with %d frames.', len(frame_names))
  dataset = _ReadObjectDataset(root_dir, frame_names)
  logging.info('Saving object dataset at %s with %d shards.', tfrecord_path,
               num_shards)

  tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(tfrecord_path, index, num_shards)
      for index in range(num_shards)
  ]

  with contextlib.ExitStack() as exit_stack:
    tf_record_writers = [
        exit_stack.enter_context(tf.io.TFRecordWriter(filename))
        for filename in tf_record_output_filenames
    ]
    total_examples = len(dataset)
    for example_index, example in enumerate(dataset):
      output_shard_index = example_index % num_shards
      serialized_example = example.SerializeToString()
      tf_record_writers[output_shard_index].write(serialized_example)
      if example_index % 100 == 0:
        logging.info('Wrote frame %d of %d.', example_index, total_examples)


def main(unused_argv):
  split_type = 'testing' if 'test' in FLAGS.split else 'training'
  root_dir = os.path.join(FLAGS.kitti_object_dir, split_type)
  split_file = os.path.join(FLAGS.kitti_object_dir, 'splits',
                            '{}.txt'.format(FLAGS.split))
  _ExportObjectDatasetToTFRecord(root_dir, split_file, FLAGS.tfrecord_path,
                                 FLAGS.num_shards)


if __name__ == '__main__':
  flags.mark_flag_as_required('kitti_object_dir')
  flags.mark_flag_as_required('split')
  flags.mark_flag_as_required('tfrecord_path')
  app.run(main)
