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
r"""Read saved Decoder's outputs and convert to KITTI text format.

First, obtain a KITTI camera calibration file.

To export all detections from a single model:

python export_kitti_detection.py \
--decoder_path=/path/to/decoder_out_000103000 \
--calib_file=/tmp/kitti_test_calibs.npz \
--output_dir=/tmp/my-kitti-export-directory \
--logtostderr

--- OR ---

Export combined detections selected from multiple models:

python export_kitti_detection.py \
--car_decoder_path=/path/to/car_decoder_out \
--ped_decoder_path=/path/to/ped_decoder_out \
--cyc_decoder_path=/path/to/cyc_decoder_out \
--calib_file=/tmp/kitti_test_calibs.npz \
--output_dir=/tmp/my-kitti-export-directory \
--logtostderr
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from lingvo import compat as tf
from lingvo.core.ops import record_pb2
from lingvo.tasks.car import kitti_metadata
from lingvo.tasks.car.tools import kitti_data
import numpy as np
from six.moves import zip


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "decoder_path", None, "Paths to decoder file containing output "
    "of decoder for everything. Either supply this argument or individual "
    "decoder paths for cars, pedestrians and cyclists.")
flags.DEFINE_string(
    "car_decoder_path", None,
    "Paths to decoder file containing output of decoder for cars."
    "Either supply plus cyclists and pedestrians or supply one "
    "decoder for all labels.")
flags.DEFINE_string(
    "ped_decoder_path", None,
    "Paths to decoder file containing output of decoder for "
    "pedestrians. Either supply plus cyclists and cars or "
    "supply one decoder for all labels.")
flags.DEFINE_string(
    "cyc_decoder_path", None,
    "Paths to decoder file containing output of decoder for cyclist. "
    "Either supply plus cars and pedestrians or supply one "
    "decoder for all labels.")
flags.DEFINE_string(
    "calib_file", None,
    "Path to a npz file that contains all calibration matrices.")
flags.DEFINE_string("output_dir", None, "Place to write detections.")
flags.DEFINE_float("score_threshold", 0, "Ignore detections with lower score.")


def LoadCalibData(fname):
  """Load and parse calibration data from NPZ file."""
  # If this throws an error, make sure the npz file was generated from
  # the same version of python as this binary.
  npz = np.load(fname)
  scene_to_calib = {}
  for idx, scene_id in enumerate(npz["scene_id"]):
    tf.logging.info("Processing %s", scene_id)
    raw_calib = {}
    raw_calib["P0"] = npz["P0"][idx]
    raw_calib["P1"] = npz["P1"][idx]
    raw_calib["P2"] = npz["P2"][idx]
    raw_calib["P3"] = npz["P3"][idx]
    raw_calib["R0_rect"] = npz["R0_rect"][idx]
    raw_calib["Tr_velo_to_cam"] = npz["Tr_velo_to_cam"][idx]
    raw_calib["Tr_imu_to_velo"] = npz["Tr_imu_to_velo"][idx]

    calib = kitti_data.ParseCalibrationDict(raw_calib)
    scene_to_calib[scene_id] = calib
  return scene_to_calib


def ExtractNpContent(np_dict, calib):
  """Parse saved np arrays and convert 3D bboxes to camera0 coordinates.

  Args:
    np_dict: a dict of numpy arrays.
    calib: a parsed calibration dictionary.

  Returns:
    A tuple of 6 ndarrays:

    - location_camera: [N, 3]. [x, y, z] in camera0 coordinate.
    - dimension_camera: [N, 3]. The [height, width, length] of objects.
    - phi_camera: [N]. Rotation around y-axis in camera0 coodinate.
    - bboxes_2d: [N, 4]. The corresponding 2D bboxes in the image coordinate.
    - scores: [N]. Confidence scores for each box for the assigned class.
    - class_ids: [N]. The class id assigned to each box.
  """
  bboxes = np_dict["bboxes"]
  scores = np_dict["scores"]
  class_ids = np_dict["class_ids"]
  bboxes_2d = np_dict["bboxes_2d"]

  # Transform from velodyne coordinates to camera coordinates.
  velo_to_cam_transform = kitti_data.VeloToCameraTransformation(calib)
  location_cam = np.zeros((len(bboxes), 3))
  dimension_cam = np.zeros((len(bboxes), 3))
  rotation_cam = np.zeros((len(bboxes), 1))
  for idx, bbox in enumerate(bboxes):
    location_cam[idx, :], dimension_cam[idx, :], rotation_cam[idx, :] = (
        kitti_data.BBox3DToKITTIObject(bbox, velo_to_cam_transform))

  return location_cam, dimension_cam, rotation_cam, bboxes_2d, scores, class_ids


_INCLUDED_KITTI_CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def ExportKITTIDetection(out_dir, source_id, location_cam, dimension_cam,
                         rotation_cam, bboxes_2d, scores, class_name, is_first):
  """Write detections to a text file in KITTI format."""
  tf.logging.info("Exporting %s for %s" % (class_name, source_id))
  fname = out_dir + "/" + source_id + ".txt"
  with tf.io.gfile.GFile(fname, "a") as fid:
    # Ensure we always create a file even when there's no detection.
    # TODO(shlens): Test whether this is actually necessary on the KITTI
    # eval server.
    if is_first:
      fid.write("")
    for location, dimension, ry, bbox_2d, score in zip(
        location_cam, dimension_cam, rotation_cam, bboxes_2d, scores):
      if score < FLAGS.score_threshold:
        continue
      # class_name, truncated(ignore), alpha(ignore), bbox2D x 4
      part1 = [class_name, -1, -1, -10] + list(bbox_2d)
      # dimesion x 3, location x 3, rotation_y x 1, score x 1
      fill = tuple(part1 + list(dimension) + list(location) + [ry] + [score])
      kitti_format_string = ("%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
                             "%lf %lf %lf %lf")
      kitti_line = kitti_format_string % fill
      fid.write(kitti_line + "\n")


def main(argv):
  if len(argv) > 1:
    raise tf.app.UsageError("Too many command-line arguments.")

  if FLAGS.decoder_path:
    assert not FLAGS.car_decoder_path and not FLAGS.ped_decoder_path \
        and not FLAGS.cyc_decoder_path, ("Either provide decoder_path or "
                                         "individual decoders but not both.")
  else:
    assert FLAGS.car_decoder_path and FLAGS.ped_decoder_path and \
        FLAGS.cyc_decoder_path, ("No decoder_path specified. Please supply all "
                                 "individual decoder_paths for labels.")
  is_single_decoder_file = FLAGS.decoder_path is not None

  if is_single_decoder_file:
    list_of_decoder_paths = [FLAGS.decoder_path]
  else:
    # Note the correspondence between _INCLUDED_KITTI_CLASS_NAMES ordering and
    # this list.
    list_of_decoder_paths = [
        FLAGS.car_decoder_path, FLAGS.ped_decoder_path, FLAGS.cyc_decoder_path
    ]

  # A list of dictionaries mapping img ids to a dictionary of numpy tensors.
  table_data = []

  img_ids = []
  for table_path in list_of_decoder_paths:
    img_id_dict = {}
    for serialized in tf.io.tf_record_iterator(table_path):
      record = record_pb2.Record()
      record.ParseFromString(serialized)
      img_id = str(tf.make_ndarray(record.fields["img_id"]))
      img_ids.append(img_id)
      np_dict = {k: tf.make_ndarray(v) for k, v in record.fields.items()}
      img_id_dict[img_id] = np_dict
    table_data.append(img_id_dict)
  img_ids = list(set(img_ids))

  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.mkdir(FLAGS.output_dir)

  all_kitti_class_names = kitti_metadata.KITTIMetadata().ClassNames()
  calib_data = LoadCalibData(tf.io.gfile.GFile(FLAGS.calib_file, "rb"))
  count = 0
  for img_id in img_ids:
    # Ignore padded samples where the img_ids are empty.
    if not img_id:
      continue
    for table_index, img_id_dict in enumerate(table_data):
      if img_id in img_id_dict:
        np_dict = img_id_dict[img_id]

        (location_cam, dimension_cam, rotation_cam, bboxes_2d, scores,
         class_ids) = ExtractNpContent(np_dict, calib_data[img_id + ".txt"])
        if is_single_decoder_file:
          valid_labels = _INCLUDED_KITTI_CLASS_NAMES
        else:
          valid_labels = [_INCLUDED_KITTI_CLASS_NAMES[table_index]]
        is_first = table_index == 0
        for class_name in valid_labels:
          class_mask = (class_ids == all_kitti_class_names.index(class_name))
          ExportKITTIDetection(FLAGS.output_dir, img_id,
                               location_cam[class_mask],
                               dimension_cam[class_mask],
                               rotation_cam[class_mask], bboxes_2d[class_mask],
                               scores[class_mask], class_name, is_first)
    count += 1
  tf.logging.info("Total example exported: %d", count)


if __name__ == "__main__":
  tf.app.run(main)
