# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# Lint as: python3
"""Tests for AdaGraft."""

import lingvo.compat as tf
from lingvo.core import adagraft
from lingvo.core import test_utils


class AdagraftTest(test_utils.TestCase):

  def test_identity(self):
    # AdaGraft(1, opt, opt) should do the same thing as opt.
    opt1 = tf.train.AdamOptimizer(0.5, beta1=0.5, beta2=0.5)
    opt2 = tf.train.AdamOptimizer(0.5, beta1=0.5, beta2=0.5)
    opt3 = tf.train.AdamOptimizer(0.5, beta1=0.5, beta2=0.5)
    opt = adagraft.AdaGraftOptimizer(1.0, opt1, opt2)
    with self.cached_session():
      var0 = tf.Variable(2.0, name="var0")
      var1 = tf.Variable(3.0, name="var1")
      loss = (var0 - 1) * (var0 - 1) + (var1 - 1) * (var1 - 1)
      o = opt.minimize(loss)
      oo = opt3.minimize(loss)
      self.evaluate(tf.global_variables_initializer())
      self.evaluate(o)
      l1 = self.evaluate([loss, var0, var1])
      print(l1)
      self.evaluate([tf.assign(var0, 2.0), tf.assign(var1, 3.0)])
      self.evaluate(oo)
      l2 = self.evaluate([loss, var0, var1])
      print(l2)
      self.assertAllClose(l1, l2)

  def test_step(self):
    """Tests grafting of Adam and SGD steps.

    Derivation of one step of Adam and SGD:
    Gradient value is [2,4].
    Adam Derivation:
    Lr_1 = 0.5(1-0.6)^(0.5)/(1-0.5) = 0.63245553203 - Does not matter
    m_1 = 0.5*G = [1,2]
    v_1 = 0.4*G^2 = [1.6,6.4]
    AdamStep = Lr_1*m_1/(sqrt{v_1}+eps) = [0.5, 0.5]
    Normalized AdamStep = [1.0, 1.0]
    SGDStep = [0.6, 1.2] Norm = [0.6, 1.2]
    TotalStep = 0.9*[0.6, 1.2]
    NewVar = [1.46, 1.92]
    """
    opt1 = tf.train.GradientDescentOptimizer(0.3)
    opt2 = tf.train.AdamOptimizer(0.5, beta1=0.5, beta2=0.6)
    opt = adagraft.AdaGraftOptimizer(0.9, opt1, opt2)
    with self.cached_session():
      var0 = tf.Variable(2.0, name="var0")
      var1 = tf.Variable(3.0, name="var1")
      loss = (var0 - 1) * (var0 - 1) + (var1 - 1) * (var1 - 1)
      o = opt.minimize(loss)
      self.evaluate(tf.global_variables_initializer())

      correct_values = [[1.058, 1.46, 1.92], [0.22387284, 1.2116001, 1.4232]]

      for i in range(2):
        self.evaluate(o)
        step_values = self.evaluate([loss, var0, var1])
        print(step_values)
        self.assertAllClose(correct_values[i], step_values)


if __name__ == "__main__":
  tf.test.main()
