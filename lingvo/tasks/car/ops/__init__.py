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
"""Car Operations."""

from lingvo import compat as tf

# Try static linking:
try:
  from lingvo.tasks.car.ops import gen_car_ops  # pylint: disable=g-import-not-at-top
except ImportError:
  gen_car_ops = tf.load_op_library(
      tf.resource_loader.get_path_to_datafile('car_ops.so'))

# Set gen_car_ops function module so sphinx generates documentation.
for v in gen_car_ops.__dict__.values():
  try:
    v.__module__ = 'lingvo.tasks.car.ops'
  except:  # pylint: disable=bare-except
    pass

pairwise_iou3d = gen_car_ops.pairwise_iou3d
point_to_grid = gen_car_ops.point_to_grid
non_max_suppression_3d = gen_car_ops.non_max_suppression3d
average_precision3d = gen_car_ops.average_precision3d
sample_points = gen_car_ops.sample_points
