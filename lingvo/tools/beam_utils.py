# Lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tools for car beam pipelines."""

import apache_beam as beam


def BeamInit():
  """Initialize the beam program.

  Typically first thing to run in main(). This call is needed before FLAGS
  are accessed, for example.
  """
  pass


def GetPipelineRoot(options=None):
  """Return the root of the beam pipeline.

  Typical usage looks like:

    with GetPipelineRoot() as root:
      _ = (root | beam.ParDo() | ...)

  In this example, the pipeline is automatically executed when the context is
  exited, though one can manually run the pipeline built from the root object as
  well.

  Args:
    options: A beam.options.pipeline_options.PipelineOptions object.

  Returns:
    A beam.Pipeline root object.
  """
  return beam.Pipeline(options=options)


def GetReader(record_format, file_pattern, value_coder, **kwargs):
  """Returns a beam Reader based on record_format and file_pattern.

  Args:
    record_format: String record format, e.g., 'tfrecord'.
    file_pattern: String path describing files to be read.
    value_coder: Coder to use for the values of each record.
    **kwargs: arguments to pass to the corresponding Reader object constructor.

  Returns:
    A beam reader object.

  Raises:
    ValueError: If an unsupported record_format is provided.
  """
  if record_format == "tfrecord":
    return beam.io.ReadFromTFRecord(file_pattern, coder=value_coder, **kwargs)

  raise ValueError("Unsupported record format: {}".format(record_format))


def GetWriter(record_format, file_pattern, value_coder, **kwargs):
  """Returns a beam Writer.

  Args:
    record_format: String record format, e.g., 'tfrecord' to write as.
    file_pattern: String path describing files to be written to.
    value_coder: Coder to use for the values of each written record.
    **kwargs: arguments to pass to the corresponding Writer object constructor.

  Returns:
    A beam writer object.

  Raises:
    ValueError: If an unsupported record_format is provided.
  """
  if record_format == "tfrecord":
    return beam.io.WriteToTFRecord(file_pattern, coder=value_coder, **kwargs)
  raise ValueError("Unsupported record format: {}".format(record_format))


def GetEmitterFn(record_format):
  """Returns an Emitter function for the given record_format.

  An Emitter function takes in a key and value as arguments and returns
  a structure that is compatible with the Beam Writer associated with
  the corresponding record_format.

  Args:
    record_format: String record format, e.g., 'tfrecord' to write as.

  Returns:
    An emitter function of (key, value) -> Writer's input type.

  Raises:
    ValueError: If an unsupported record_format is provided.
  """

  def _ValueEmitter(key, value):
    del key
    return [value]

  if record_format == "tfrecord":
    return _ValueEmitter
  raise ValueError("Unsupported record format: {}".format(record_format))
