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
"""Base models for point-cloud based detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core.ops import record_pb2


class BaseDecoder(base_layer.BaseLayer):
  """A decoder to use for decoding a detector model."""

  @classmethod
  def Params(cls):
    p = super(BaseDecoder, cls).Params()
    p.Define(
        'summarize_boxes_on_image', False,
        'If true, enable the summary metric that projects bounding boxes '
        'to the camera image to view predictions from camera view.')

    p.Define('ap_metric', None, 'Configuration of AP metric for decoding.')

    p.Define(
        'laser_sampling_rate', 0.05,
        'Rate at which real laser outputs are added to decoder output. '
        'Because the laser outputs are large, we only want to output '
        'the lasers on a small number of the batches.')
    return p

  def _SampleLaserForVisualization(self, points_xyz, points_padding):
    """Samples laser points based on configured laser_sampling_rate.

    Args:
      points_xyz: [batch, num_points, 3] float Tensor.
      points_padding: [batch, num_points] float Tensor.

    Returns:
      .NestedMap:

      - points_xyz: 0.0 or points_xyz float Tensor passthrough.
      - points_padding: 0.0 or points_padding float Tensor passthrough.
      - points_sampled: scalar bool Tensor if points were sampled.
        If false, points_xyz and points_padding are scalar 0s to
        reduce the amount of data transferred.
    """
    p = self.params
    rand = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
    sample_laser = rand < p.laser_sampling_rate
    points_xyz, points_padding = tf.cond(
        sample_laser, lambda: (points_xyz, points_padding), lambda: (0.0, 0.0))
    return py_utils.NestedMap({
        'points_xyz': points_xyz,
        'points_padding': points_padding,
        'points_sampled': sample_laser
    })

  def SaveTensors(self, tensor_map):
    """Returns a serialized representation of the contents of `tensor_map`.

    Args:
      tensor_map: A NestedMap of string keys to numpy arrays.

    Returns:
      A serialized record_pb2.Record() of the contents of 'tensor_map'.
    """

    def AddKeyVals(**kwargs):
      record = record_pb2.Record()
      for k, v in kwargs.items():
        record.fields[k].CopyFrom(tf.make_tensor_proto(v))
      return record

    records = AddKeyVals(**tensor_map)
    serialized = records.SerializeToString()
    return serialized
