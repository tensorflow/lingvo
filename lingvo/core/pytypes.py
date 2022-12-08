# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""PyType Typing utilities for lingvo.

A subset of third_party/py/praxis/pytypes.py.
"""

from typing import List, Tuple, TypeVar, Union, Mapping

import lingvo.compat as tf
from lingvo.core import hyperparams
from lingvo.core import nested_map
from lingvo.core import py_utils
import numpy as np

NpTensor = np.ndarray

NestedMap = nested_map.NestedMap
Params = hyperparams.Params
InstantiableParams = hyperparams.InstantiableParams

T = TypeVar('T')
Nested = Union[T, Tuple[T, ...], List[T], Mapping[str, T], py_utils.NestedMap]
NestedTensor = Nested[tf.Tensor]
NestedBool = Nested[bool]
NestedInt = Nested[int]
