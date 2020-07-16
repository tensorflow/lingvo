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
"""Utilities for pruning."""

import lingvo.compat as tf
from model_pruning.python import pruning


def AddToPruningCollections(weight,
                            mask,
                            threshold,
                            gradient=None,
                            old_weight=None,
                            old_old_weight=None):
  """Add mask, threshold, and weight vars to their respective collections."""
  if mask not in tf.get_collection(pruning.MASK_COLLECTION):
    tf.add_to_collection(pruning.WEIGHT_COLLECTION, weight)
    tf.add_to_collection(pruning.MASK_COLLECTION, mask)
    tf.add_to_collection(pruning.THRESHOLD_COLLECTION, threshold)

    # Add gradient, old_weight, and old_old_weight to collections approximating
    # gradient and hessian, where old_weight is the weight tensor one step
    # before and old_old_weight is the weight tensor two steps before.
    if gradient is not None:
      assert old_weight is not None
      assert old_old_weight is not None
      tf.add_to_collection(pruning.WEIGHT_GRADIENT_COLLECTION, gradient)
      tf.add_to_collection(pruning.OLD_WEIGHT_COLLECTION, old_weight)
      tf.add_to_collection(pruning.OLD_OLD_WEIGHT_COLLECTION, old_old_weight)
