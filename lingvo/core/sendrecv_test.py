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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
from lingvo.core import sendrecv
from lingvo.core import test_utils


def _ListDevices(target):
  with tf.Session(target) as sess:
    devices = sess.list_devices()
  return [_.name for _ in devices]


def _Target():
  return ""


class SendrecvTest(test_utils.TestCase):

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

      with tf.Session(_Target(), graph=g) as sess:
        _, val = sess.run([send_op, recv_val])

      self.assertAllClose(to_send, val)

  def testInsideFunction(self):
    devices = _ListDevices(_Target())
    sender, recver = devices[0], devices[-1]
    shape = []

    def SendRecv(graph, dtype):
      to_send = np.array(3.1415 + 2j).astype(dtype.as_numpy_dtype)
      with graph.as_default():
        ch = sendrecv.Channel(dtype, shape, sender, recver, "test")
        with tf.device(sender):

          @function.Defun()
          def Send():
            src_val = tf.constant(to_send)
            ch.Send(src_val)
            return 1.0

          send_op = Send()

        with tf.device(recver):

          @function.Defun()
          def Recv():
            return ch.Recv()

          recv_val = Recv()
      return send_op, recv_val, to_send

    for dtype in tf.float32, tf.complex64:
      g = tf.Graph()
      send_op, recv_val, sent_val = SendRecv(g, dtype)

      with tf.Session(_Target(), graph=g) as sess:
        _, val = sess.run([send_op, recv_val])

      self.assertAllClose(sent_val, val)


if __name__ == "__main__":
  tf.test.main()
