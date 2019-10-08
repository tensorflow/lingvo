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
"""The compatible tensorflow library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compat.v1 import *  # pylint:disable=wildcard-import

# Import absl.flags and absl.logging to overwrite the Tensorflow ones.
# This is the intended behavior in TF 2.0.
# pylint:disable=g-bad-import-order, unused-import, g-import-not-at-top
from absl import flags
from absl import logging
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.compat import v2_compat

from tensorflow.python.framework import function as _function_lib

# The following imports are needed to expose private _Send/_Recv ops
# on TensorFlow 1.X. The could be removed once support for 1.X is dropped.
from google.protobuf import text_format as _text_format
from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.framework import op_def_registry as _op_def_registry
# pylint: enable=g-direct-tensorflow-import

v2_compat.disable_v2_behavior()
Defun = _function_lib.Defun


# TODO(slebedev): Remove after there is no need to support 1.X.
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """\
op {
  name: "_Recv"
  output_arg {
    name: "tensor"
    type_attr: "tensor_type"
  }
  attr {
    name: "tensor_type"
    type: "type"
  }
  attr {
    name: "tensor_name"
    type: "string"
  }
  attr {
    name: "send_device"
    type: "string"
  }
  attr {
    name: "send_device_incarnation"
    type: "int"
  }
  attr {
    name: "recv_device"
    type: "string"
  }
  attr {
    name: "client_terminated"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "_Send"
  input_arg {
    name: "tensor"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "tensor_name"
    type: "string"
  }
  attr {
    name: "send_device"
    type: "string"
  }
  attr {
    name: "send_device_incarnation"
    type: "int"
  }
  attr {
    name: "recv_device"
    type: "string"
  }
  attr {
    name: "client_terminated"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
"""


def _Recv(tensor_type,
          tensor_name,
          send_device,
          send_device_incarnation,
          recv_device,
          name=None):
  return _op_def_lib.apply_op(
      "_Recv",
      tensor_type=tensor_type,
      tensor_name=tensor_name,
      send_device=send_device,
      send_device_incarnation=send_device_incarnation,
      recv_device=recv_device,
      client_terminated=False,
      name=name if name else "Recv")


def _Send(tensor,
          tensor_name,
          send_device,
          send_device_incarnation,
          recv_device,
          name=None):
  return _op_def_lib.apply_op(
      "_Send",
      tensor=tensor,
      tensor_name=tensor_name,
      send_device=send_device,
      send_device_incarnation=send_device_incarnation,
      recv_device=recv_device,
      client_terminated=False,
      name=name if name else "Send")


# pylint: disable=undefined-variable
if not hasattr(raw_ops, "Send") and not hasattr(raw_ops, "Recv"):
  _op_def_lib = _InitOpDefLibrary()
  raw_ops.Send = _Send
  raw_ops.Recv = _Recv
# pylint: enable=undefined-variable

del _Send, _Recv, _InitOpDefLibrary
