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
r"""Read Waymo Decoder's outputs and convert to submission format.

Example:

python export_to_submission_format.py \
  --decoder_path=/tmp/decoder_out_000060600 \
  --output_dir=/tmp/test_decoder_output

preds.bin and gts.bin will be found in /tmp/test_decoder_output/ dir.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from lingvo import compat as tf
from lingvo.core.ops import record_pb2
from six.moves import range
from waymo_open_dataset.protos import metrics_pb2

FLAGS = flags.FLAGS
flags.DEFINE_string("decoder_path", None,
                    "Path to the decoder output tf.Record file.")
flags.DEFINE_string("output_dir", None, "Place to write detections.")
flags.DEFINE_float("score_threshold", 0, "Ignore detections with lower score.")


def convert_detections(table_path):
  """Convert detections in `table_path` to metric format.

  Args:
    table_path: Path to TFRecord file of decoder outputs.

  Returns:
    (preds, gts): metric_pb2.Objects() of predictions and groundtruths.
  """
  img_ids = []
  img_id_dict = {}
  for serialized in tf.io.tf_record_iterator(table_path):
    record = record_pb2.Record()
    record.ParseFromString(serialized)
    img_id = str(tf.make_ndarray(record.fields["frame_id"]))
    img_ids.append(img_id)
    np_dict = {k: tf.make_ndarray(v) for k, v in record.fields.items()}
    img_id_dict[img_id] = np_dict

  preds = metrics_pb2.Objects()
  gts = metrics_pb2.Objects()
  for img_id in img_ids:
    # Extract the underlying context string and timestamp
    # from the image id.
    #
    # TODO(vrv): Consider embedding these values into the decoder output
    # individually.
    context_name = img_id[2:img_id.rindex("_")]
    timestamp = int(img_id[img_id.rindex("_") + 1:-1])

    np_dict = img_id_dict[img_id]
    pred_bboxes = np_dict["bboxes"]  # [max boxes, 7]
    pred_scores = np_dict["scores"]  # [max_boxes]
    gt_bboxes = np_dict["gt_bboxes"]  # [num_gt_boxes, 7]
    gt_labels = np_dict["gt_labels"]  # [num_gt_boxes]
    class_ids = np_dict["class_ids"]  # [max_boxes]

    def _add_box(label, box_vec):
      label.box.center_x = box_vec[0]
      label.box.center_y = box_vec[1]
      label.box.center_z = box_vec[2]
      label.box.length = box_vec[3]
      label.box.width = box_vec[4]
      label.box.height = box_vec[5]
      label.box.heading = box_vec[6]

    num_gts = gt_bboxes.shape[0]
    for gt_idx in range(num_gts):
      gt_object = metrics_pb2.Object()
      gt_object.context_name = context_name
      gt_object.frame_timestamp_micros = timestamp
      label = gt_object.object
      _add_box(label, gt_bboxes[gt_idx])
      label.type = gt_labels[gt_idx]
      # We should fill in the difficulty level once we want to measure the
      # breakdown by LEVEL.
      label.detection_difficulty_level = 0
      gts.objects.append(gt_object)

    num_pds = pred_bboxes.shape[0]
    for pd_idx in range(num_pds):
      score = pred_scores[pd_idx]
      if score < FLAGS.score_threshold:
        continue
      pd_object = metrics_pb2.Object()
      pd_object.context_name = context_name
      pd_object.frame_timestamp_micros = timestamp
      pd_object.score = score
      label = pd_object.object
      _add_box(label, pred_bboxes[pd_idx])
      label.type = class_ids[pd_idx]
      preds.objects.append(pd_object)

  return preds, gts


def main(argv):
  if len(argv) > 1:
    raise tf.app.UsageError("Too many command-line arguments.")

  preds, gts = convert_detections(FLAGS.decoder_path)

  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.mkdir(FLAGS.output_dir)

  # Write the predictions and gts into individual files.
  #
  # The outputs can then be passed to the official metrics implementations and
  # server.
  with tf.io.gfile.GFile(os.path.join(FLAGS.output_dir, "preds.bin"), "w") as f:
    f.write(preds.SerializeToString())

  with tf.io.gfile.GFile(os.path.join(FLAGS.output_dir, "gts.bin"), "w") as f:
    f.write(gts.SerializeToString())


if __name__ == "__main__":
  tf.app.run(main)
