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
r"""Tool to convert Waymo Open Dataset to tf.Examples.

An example of running on Cloud DataFlow: (not verified working yet).

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

path/to/generate_waymo_tf.py \
  --input_file_pattern=gs://path/to/waymo/inputs \
  --output_filebase=gs://$BUCKET/waymo/training/output@1000 -- \
  --project=$PROJECT \
  --temp_location=$TEMP_DIR \
  --runner=DataflowRunner
"""

from absl import app
from absl import flags

import apache_beam as beam
from lingvo import compat as tf
from lingvo.tasks.car.waymo.tools import waymo_proto_to_tfe
from lingvo.tools import beam_utils
from waymo_open_dataset import dataset_pb2

tf.enable_eager_execution()

flags.DEFINE_string('input_file_pattern', None, 'Path to read input')
flags.DEFINE_string('output_filebase', None, 'Path to write output')

FLAGS = flags.FLAGS


def main(argv):
  beam_utils.BeamInit()

  assert FLAGS.input_file_pattern
  assert FLAGS.output_filebase

  # Construct pipeline options from argv.
  options = beam.options.pipeline_options.PipelineOptions(argv[1:])

  reader = beam_utils.GetReader(
      'tfrecord',
      FLAGS.input_file_pattern,
      value_coder=beam.coders.ProtoCoder(dataset_pb2.Frame))

  writer = beam_utils.GetWriter(
      'tfrecord',
      file_pattern=FLAGS.output_filebase,
      value_coder=beam.coders.ProtoCoder(tf.train.Example))

  emitter_fn = beam_utils.GetEmitterFn('tfrecord')
  with beam_utils.GetPipelineRoot(options=options) as root:
    _ = (
        root
        | 'Read' >> reader
        | 'ConvertToTFExample' >> beam.ParDo(
            waymo_proto_to_tfe.WaymoOpenDatasetConverter(emitter_fn))
        | 'Write' >> writer)


if __name__ == '__main__':
  app.run(main)
