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
r"""Creates KITTI Classification dataset out of the KITTI Object detection data.

Produces TFRecords with the following format:
--------------------------------------------------------------------------------

Values:
  num_points: Int64 scalar with the number of points in that box.

  points: Float list of 3D locations (X, Y, Z) per point.

  points_feature: Float list of per-point feature (reflectance scalar).

  bbox_3d: Float list of the 7 box coordinates. bboxes_3d[0:3] is the absolute
  center location in meters. bboxes_3d[3:6] is the bbox dimensions in meters,
  and bboxes_3d[6] is the rotation in radians.

  label: An Int64 scalar with the class label id.

  text: A bytelist with the string name of the class.

  difficulty: An Int64 scalar with the difficulty level of the object.

  occlusion: An Int64 scalar with the occlusion level of the object.

  scene_id: An Int64 scalar with the original scene id the object is from.

  bbox_id: An Int64 scalar with which number bounding box the object is from.


Note: All scalars are wrapped in a list of len 1.
--------------------------------------------------------------------------------

IMPORTANT: Depending on your model, you may or may not want to enable
preprocessors to export for data augmentation. If unsure, do *not* run with
preprocessors as that might add additional data augmentation steps.

To run:

bazel run -c opt \
  //lingvo/tasks/car/tools:create_kitti_crop_dataset \
  --model_name=car.kitti.PillarsModelV1 \
  --norun_preprocessors \
  --input_file_pattern=/path/to/kitti/train_pattern \
  --output_file_pattern=/path/to/output/gt_objects@100
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import apache_beam as beam

from lingvo import compat as tf
from lingvo import model_registry
from lingvo.core import cluster_factory
from lingvo.core import py_utils
from lingvo.tasks.car import geometry
from lingvo.tasks.car import input_extractor
from lingvo.tasks.car.params import kitti  # pylint: disable=unused-import
from lingvo.tools import beam_utils

import numpy as np
from six.moves import range


flags.DEFINE_string('input_file_pattern', None, 'Where to get the data.')
flags.DEFINE_string('output_file_pattern', None,
                    'Output directory + prefix where to write files.')
flags.DEFINE_string(
    'model_name', 'car.kitti.PillarsModelV1',
    'Name of registered model whose input pipeline '
    'will be used to mine groundtruth objects from.')
flags.DEFINE_enum('split', 'Train', ['Train', 'Dev'],
                  'Split of data to generate GT objects from')
flags.DEFINE_bool(
    'run_preprocessors', False,
    'Whether to run the preprocessors when extracting data. '
    'Leave this as False if you are generating data for the '
    'purposes of analysis. If you are generating data for ground '
    'truth bbox augmentation, enable this.')

FLAGS = flags.FLAGS


def _GetFilteredBoundingBoxData(kitti_data):
  """Given a single batch element of data, process it for writing.

  Args:
    kitti_data: A NestedMap of KITTI input generator returned data with a batch
      size of 1.

  Returns:
    A NestedMap of all the output data we need to write per bounding box
    cropped pointclouds.
  """
  points = kitti_data.lasers.points_xyz
  points_feature = kitti_data.lasers.points_feature
  bboxes_3d = kitti_data.labels.bboxes_3d
  bboxes_3d_mask = kitti_data.labels.bboxes_3d_mask
  bboxes_3d = tf.boolean_mask(bboxes_3d, bboxes_3d_mask)

  if 'points_padding' in kitti_data.lasers:
    points_validity_mask = tf.cast(kitti_data.lasers.points_padding - 1,
                                   tf.bool)
    points = tf.boolean_mask(points, points_validity_mask)
    points_feature = tf.boolean_mask(points_feature, points_validity_mask)

  points_in_bboxes_mask = geometry.IsWithinBBox3D(points, bboxes_3d)

  output_map = py_utils.NestedMap()
  # Points and features contain the whole pointcloud, which we will use
  # per box boolean masks later in _ToTFExampleProto to subselect data per box.
  output_map.points = points
  output_map.points_feature = points_feature
  output_map.points_in_bboxes_mask = points_in_bboxes_mask

  output_map.source_id = kitti_data.labels.source_id

  # Add additional data
  output_keys = [
      'bboxes_3d',
      'labels',
      'texts',
      'occlusion',
      'difficulties',
      'truncation',
  ]
  for key in output_keys:
    output_map[key] = tf.boolean_mask(kitti_data.labels[key],
                                      kitti_data.labels.bboxes_3d_mask)
  return output_map


class _ProcessShard(beam.DoFn):
  """Process a given shard."""

  def __init__(self, model_name, split, run_preprocessors):
    self._model_name = model_name
    self._split = split
    self._run_preprocessors = run_preprocessors
    self._sess = None

    # Create a cluster configuration assuming evaluation; the input pipelines
    # need to know the cluster job type to set up the outputs correctly.
    cluster = cluster_factory.Current()
    cluster.params.job = 'evaler'
    cluster.params.mode = 'sync'
    cluster.params.task = 0
    cluster.params.evaler.replicas = 1
    self._cluster = cluster_factory.Cluster(cluster.params)

  def _create_graph(self):
    if self._sess is not None:
      return

    with self._cluster:
      cfg = model_registry.GetParams(self._model_name, self._split)
      cfg.input.batch_size = 1
      # Turn off label filtering so the database contains
      # all objects.
      cfg.input.extractors.labels.filter_labels = None

      # Disable preprocessors if they are not required.
      if not self._run_preprocessors:
        cfg.input.preprocessors_order = []

      graph = tf.Graph()
      with graph.as_default():
        inp = cfg.input.Instantiate()
        self._elem = tf.placeholder(tf.string)
        bucket, batch = inp.ExtractUsingExtractors(self._elem)
        self._filtered_data = _GetFilteredBoundingBoxData(batch)
        self._bucket = bucket
    self._sess = tf.Session(graph=graph)

  def _ToTFExampleProto(self, filtered_data, bbox_idx):
    num_boxes = filtered_data.bboxes_3d.shape[0]
    if bbox_idx >= num_boxes:
      raise ValueError('`bbox_id` should be < num_boxes')
    bbox_mask = filtered_data.points_in_bboxes_mask[:, bbox_idx]

    num_points, _ = py_utils.GetShape(filtered_data.points[bbox_mask], 2)

    example = tf.train.Example()
    feature = example.features.feature
    feature['num_points'].int64_list.value[:] = [num_points]
    feature['points'].float_list.value[:] = (
        filtered_data.points[bbox_mask].ravel().tolist())
    feature['points_feature'].float_list.value[:] = (
        filtered_data.points_feature[bbox_mask].ravel().tolist())
    # Note: bbox_3d, label, text, difficulty are singular, which is inconsistent
    # with the original KITTI data, but that is because each example
    # has only one bounding box, one label, one text, one difficulty.
    feature['bbox_3d'].float_list.value[:] = (
        filtered_data.bboxes_3d[bbox_idx, :].ravel().tolist())
    feature['label'].int64_list.value[:] = [
        filtered_data.labels[bbox_idx].astype(np.int64)
    ]
    feature['text'].bytes_list.value[:] = [filtered_data.texts[bbox_idx]]
    feature['difficulty'].int64_list.value[:] = [
        filtered_data.difficulties[bbox_idx].astype(np.int64)
    ]
    feature['occlusion'].int64_list.value[:] = [
        filtered_data.occlusion[bbox_idx].astype(np.int64)
    ]
    feature['scene_id'].int64_list.value[:] = [int(filtered_data.source_id)]
    feature['bbox_id'].int64_list.value[:] = [int(bbox_idx)]
    return example

  def process(self, value):
    self._create_graph()
    elem_str = value.SerializeToString()

    b, bucket = self._sess.run([self._filtered_data, self._bucket],
                               feed_dict={self._elem: elem_str})
    if bucket > input_extractor.BUCKET_UPPER_BOUND:
      return
    b = py_utils.NestedMap(b)

    # Flatten the batch.
    flatten = b.FlattenItems()
    if not flatten:
      return

    num_boxes = b.bboxes_3d.shape[0]

    # For each box, get the pointcloud and write it as an example.
    for bbox_id in range(num_boxes):
      tf_example = self._ToTFExampleProto(b, bbox_id)
      yield tf_example


def main(_):
  beam_utils.BeamInit()

  if not FLAGS.output_file_pattern:
    raise ValueError('Must provide an output_file_pattern')

  reader = beam.io.ReadFromTFRecord(
      FLAGS.input_file_pattern, coder=beam.coders.ProtoCoder(tf.train.Example))

  model_name = FLAGS.model_name
  split = FLAGS.split
  run_preprocessors = FLAGS.run_preprocessors

  with beam_utils.GetPipelineRoot() as root:
    _ = (
        root
        | 'Read' >> reader
        | 'ToTFExample' >> beam.ParDo(
            _ProcessShard(model_name, split, run_preprocessors))
        | 'Reshuffle' >> beam.Reshuffle()
        | 'Write' >> beam.io.WriteToTFRecord(
            FLAGS.output_file_pattern,
            coder=beam.coders.ProtoCoder(tf.train.Example)))


if __name__ == '__main__':
  app.run(main)
