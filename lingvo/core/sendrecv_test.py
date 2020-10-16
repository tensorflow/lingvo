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
"""Tests for sendrecv."""

from absl.testing import parameterized
import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import sendrecv
from lingvo.core import test_utils
import numpy as np

FLAGS = tf.flags.FLAGS


def _ListDevices(target):
  with tf.Session(target) as sess:
    devices = sess.list_devices()
  return [_.name for _ in devices]


def _Target():
  return ""


class SendrecvTest(test_utils.TestCase, parameterized.TestCase):

  def testBasic(self):
    devices = _ListDevices(_Target())
    print("\n".join(devices))
    sender, recver = devices[0], devices[-1]
    shape = []

    for dtype in tf.float32, tf.complex64:
      to_send = np.array(3.1415 + 2j).astype(dtype.as_numpy_dtype)
      g = tf.Graph()
      with g.as_default():
        ch = sendrecv.Channel(dtype, shape, sender, recver, "test")
        with tf.device(sender):
          src_val = tf.constant(to_send)
          send_op = ch.Send(src_val)
        with tf.device(recver):
          recv_val = ch.Recv()

      with tf.Session(_Target(), graph=g):
        _, val = self.evaluate([send_op, recv_val])

      self.assertAllClose(to_send, val)

  @parameterized.named_parameters(
      ("_function",) if py_utils._UseTfFunction() else ("_defun",))
  def testInsideFunction(self):
    devices = _ListDevices(_Target())
    sender, recver = devices[0], devices[-1]
    shape = []

    def SendRecv(graph, dtype):
      to_send = np.array(3.1415 + 2j).astype(dtype.as_numpy_dtype)
      with graph.as_default():
        ch = sendrecv.Channel(dtype, shape, sender, recver, "test")
        with tf.device(sender):

          # py_utils.CallDefun requires non-empty inputs. Same below.
          def Send(_):
            src_val = tf.constant(to_send)
            ch.Send(src_val)
            return tf.convert_to_tensor(1.0)

          send_op = py_utils.CallDefun(Send, tf.convert_to_tensor(0))

        with tf.device(recver):

          def Recv(_):
            return ch.Recv()

          recv_val = py_utils.CallDefun(Recv, tf.convert_to_tensor(0))
      return send_op, recv_val, to_send

    for dtype in tf.float32, tf.complex64:
      g = tf.Graph()
      send_op, recv_val, sent_val = SendRecv(g, dtype)

      with tf.Session(_Target(), graph=g):
        _, val = self.evaluate([send_op, recv_val])

      self.assertAllClose(sent_val, val)


if __name__ == "__main__":
  tf.test.main()
