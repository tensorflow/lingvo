# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Common pytype definitions."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import jax
from jax import numpy as jnp
from jax.experimental import pjit
from lingvo.jax import py_utils
import numpy as np

ParamsT = py_utils.Params
JTensor = jnp.ndarray
PRNGKey = JTensor
JTensorOrPartitionSpec = Union[JTensor, pjit.PartitionSpec]
NpTensor = np.ndarray
SummaryDict = Union[py_utils.NestedMap, Dict[str, JTensor]]
PyTreeDef = type(jax.tree_structure(None))

T = TypeVar('T')
Nested = Union[T, Tuple[Any, ...], List[Any], Dict[str, Any],
               py_utils.NestedMap]
NestedJTensor = Nested[JTensor]
NestedBool = Nested[bool]
NestedParams = Nested[ParamsT]
NestedPartitionSpec = Nested[pjit.PartitionSpec]
NestedJTensorOrPartitionSpec = Nested[JTensorOrPartitionSpec]
NestedShapeDtypeStruct = Nested[jax.ShapeDtypeStruct]

# Sharding annotation for a dim can be a single int, or a str, or a sequence of
# (int, str), or None. For example "1", "-1", "None", "data", "(data, replia)"
# are all valid sharding annoations for a particular tensor axis.
DimShardingAnnotation = Optional[Union[Sequence[Union[int, str]], int, str]]
SplitDimsMapping = Optional[Sequence[DimShardingAnnotation]]
