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
"""Common layers for car models.

These are usually sub-classes of base_layer.BaseLayer used by builder_lib.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.tasks.car import car_lib


class SamplingAndGroupingLayer(base_layer.BaseLayer):
  """Sampling and Grouping layer (based on PointNet++)."""

  @classmethod
  def Params(cls):
    p = super(SamplingAndGroupingLayer, cls).Params()
    p.Define(
        'num_samples', 0,
        'Number of points to be sampled. Each sampled point will be '
        'returned with its corresponding group of neighbors.')
    p.Define('ball_radius', 0,
             'The distance around each sampled point to obtain neighbours.')
    p.Define('group_size', 0, 'Number neighbours for each sampled point.')
    p.Define(
        'sample_neighbors_uniformly', True,
        'Whether to sample neighbors uniformly within the ball radius. '
        'If False, this will pick the nearest neighbors by distance.')
    return p

  def FProp(self, theta, input_data):
    """Apply projection to inputs.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      input_data: A NestedMap object containing 'points', 'features', 'padding'
        Tensors, all of type tf.float32.
        'points': Shape [N, P1, 3]
        'features': Shape [N, P1, F]
        'padding': Shape [N, P1] where 0 indicates real, 1 indicates padded.

    Returns:
      A NestedMap consisting of the following two NestedMaps,
        grouped_points: consists of the grouped points, features and padding.
        query_points: consists of the sampled points and padding.
    """

    p = self.params
    features = input_data.features
    n, p1, c = py_utils.GetShape(features)
    points = py_utils.HasShape(input_data.points, [n, p1, 3])
    padding = py_utils.HasShape(input_data.padding, [n, p1])

    # Sampling
    sampled_idx, _ = car_lib.FarthestPointSampler(
        points, padding, num_sampled_points=p.num_samples)
    query_points = car_lib.MatmulGather(points, tf.expand_dims(sampled_idx, -1))
    query_points = tf.squeeze(query_points, -2)

    # Grouping
    grouped_idx, grouped_padding = car_lib.NeighborhoodIndices(
        points,
        query_points,
        p.group_size,
        points_padding=padding,
        max_distance=p.ball_radius,
        sample_neighbors_uniformly=p.sample_neighbors_uniformly)
    grouped_points = car_lib.MatmulGather(points, grouped_idx)
    # Normalize the grouped points based on the location of the query point.
    grouped_points -= tf.expand_dims(query_points, -2)
    grouped_features = car_lib.MatmulGather(features, grouped_idx)

    # Get the padding for the query points.
    query_padding = tf.array_ops.batch_gather(padding, sampled_idx)

    # Verify the shapes of output tensors.
    query_points = py_utils.HasShape(query_points, [n, p.num_samples, 3])
    query_padding = py_utils.HasShape(query_padding, [n, p.num_samples])
    grouped_features = py_utils.HasShape(grouped_features,
                                         [n, p.num_samples, p.group_size, c])
    grouped_padding = py_utils.HasShape(grouped_padding,
                                        [n, p.num_samples, p.group_size])

    output_grouped_points = py_utils.NestedMap(
        points=grouped_points,
        features=grouped_features,
        padding=grouped_padding)
    output_query = py_utils.NestedMap(
        points=query_points, padding=query_padding)
    output_map = py_utils.NestedMap({
        'grouped_points': output_grouped_points,
        'query_points': output_query
    })
    return output_map
