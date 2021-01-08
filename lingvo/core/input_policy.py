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
"""A policy module to place the input generator."""

import lingvo.compat as tf


def Apply(input_params):
  """Possibly wraps input_params according to the input policy.

  Args:
    input_params: An input generator params.

  Returns:
    A possibly updated input_params to use to instantiate the input generator.
  """

  class _UseInputDevice(input_params.cls):
    """Places the input generator on the input device."""

    def __init__(self, params):
      with tf.device(self.cluster.input_device):
        super().__init__(params)

    def SplitInputBatch(self, num_splits):
      with tf.device(self.cluster.input_device):
        return super().SplitInputBatch(num_splits)

  return input_params.Copy().Set(cls=_UseInputDevice)
