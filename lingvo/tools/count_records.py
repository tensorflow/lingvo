# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Tool to count number of records in a dataset.

Most other file formats have efficient ways to fetch the number of records in a
dataset.  However, some formats such as TFRecord requires you to essentially
scan the files to perform this count.

This is a short little beam script that can leverage many machines to read
all of the files in parallel potentially faster than a single machine script.
It is recommended that for other file formats, simply reading the metadata
available in their formats should work; this file should not really be
extended to any other format that already has efficient ways of counting
records.
"""

from absl import app
from absl import flags

import apache_beam as beam
from lingvo.tools import beam_utils

flags.DEFINE_string('input_file_pattern', None, 'Path to read input')
flags.DEFINE_string('output_count_file', None, 'File to write output to.')
flags.DEFINE_string('record_format', None,
                    'Record format of the input, e.g., tfrecord.')

FLAGS = flags.FLAGS


def main(argv):
  beam_utils.BeamInit()

  # Construct pipeline options from argv.
  options = beam.options.pipeline_options.PipelineOptions(argv[1:])

  reader = beam_utils.GetReader(
      FLAGS.record_format,
      FLAGS.input_file_pattern,
      value_coder=beam.coders.BytesCoder())

  with beam_utils.GetPipelineRoot(options=options) as root:
    _ = (
        root
        | 'Read' >> reader  # Read each record.
        | 'EmitOne' >> beam.Map(lambda _: 1)  # Emit a 1 for each record.
        | 'Count' >> beam.CombineGlobally(sum)  # Sum counts.
        | 'WriteToText' >> beam.io.WriteToText(FLAGS.output_count_file))


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['input_file_pattern', 'output_count_file', 'record_format'])
  app.run(main)
