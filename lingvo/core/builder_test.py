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
"""Tests for the Builder pattern."""

import inspect
from lingvo import compat as tf
from lingvo.core import builder
from lingvo.core import cluster_factory
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core import tshape
import numpy as np
import sympy


class BuilderTest(test_utils.TestCase):

  def _Expect(self, expected_cost, p, *inputs):
    meta = p.cls.FPropMeta(p, *(tshape.Shape(s) for s in inputs))
    self.assertEqual(meta.flops, expected_cost)

    g = tf.Graph()
    with g.as_default():
      l = p.Instantiate()
      xs = [tf.random.normal(shape=s) for s in inputs]
      ys = l.FPropDefaultTheta(*xs)

    with self.session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      _ = sess.run(ys)

  def testCostEstimate(self):
    b = builder.Base.Params()
    b = b.Instantiate()
    # pyformat: disable
    self._Expect(2400, b._Linear('l', 10, 20), [2, 3, 10])
    self._Expect(60, b._Bias('b', 10), [2, 3, 10])
    self._Expect(150, b._Activation('a'), [3, 5, 10])
    self._Expect(600, b._BN('b', 10), [2, 3, 10])
    self._Expect(
        (2**21) * 9 * 2,
        b._Conv2D('c', (3, 3, 8, 16), (1, 1)),
        [16, 32, 32, 8])
    self._Expect(
        (2**19) * 9 * 2,
        b._Conv2D('c', (3, 3, 8, 16), (2, 2)),
        [16, 32, 32, 8])
    self._Expect(
        (2**20) * 9 * 2,
        b._Conv2D('c', (3, 3, 8, 16), (1, 2)),
        [16, 32, 32, 8])
    self._Expect(
        24 * 10 * 10 * 2 * 7,
        b._Rep(
            'rep',
            7,
            b._Linear('l', 10, 10)),
        [8, 3, 10])
    self._Expect(
        (2**21) * 9 * 2 + (2**14) * 16 * 32 * 2 + (2**14) * 32,
        b._Seq(
            'seq',
            b._Conv2D('c', (3, 3, 8, 16), (1, 1)),
            b._Linear('l', 16, 32),
            b._Activation('a')),
        [16, 32, 32, 8])
    self._Expect(
        (((2**19) * 9 * 2 + (2**12) * 16 * 32 * 2 + (2**12) * 32) * 2),
        b._Par(
            'p',
            b._Seq(
                'b0',
                b._Conv2D('c', (3, 3, 8, 16), (2, 2)),
                b._Linear('l', 16, 32),
                b._Activation('a')),
            b._Seq(
                'b1',
                b._Conv2D('c', (3, 3, 8, 16), (2, 2)),
                b._Linear('l', 16, 32),
                b._Activation('a'))),
        [16, 32, 32, 8])
    # pyformat: enable

  def testFetch(self):
    # Construct a layer w/ two parallel branches.  We want to demonstrate that
    # we can fetch intermediate values from two branches independently.
    # pyformat: disable
    b = builder.Base.Params()
    b = b.Instantiate()
    p = b._Par(
        'p',
        b._Seq(
            'b0',
            b._Conv2D('c', (3, 3, 8, 16), (2, 2)),
            b._Linear('l', 16, 32),
            b._Save('fetch'),
            b._Activation('a')),
        b._Seq(
            'b1',
            b._Conv2D('c', (3, 3, 8, 16), (2, 2)),
            b._Linear('l', 16, 32),
            b._Activation('a'),
            b._Save('fetch')))
    p = b._AddFetches('mh', p, ['b0.fetch', 'b1.fetch'])
    # pyformat: enable
    g = tf.Graph()
    with g.as_default():
      l = p.Instantiate()
      x = tf.random.normal(shape=[17, 64, 64, 8])
      outs = l.FPropDefaultTheta(x)
      x, y, b0, b1 = outs

    with self.session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      v0, v1, u, v = sess.run([b0, b1, x, y])

    self.assertAllClose(np.maximum(0, v0), u)
    self.assertAllClose(v1, v)

  def testFetchGrad(self):
    # Tests we can fetch backprop gradients.
    # pyformat: disable
    b = builder.Base.Params()
    b = b.Instantiate()
    p = b._Seq(
        'seq',
        b._Linear('l', 16, 32),
        b._Bias('b', 32),
        b._Save('fetch'),
        b._Activation('a'))
    # pyformat: enable

    g = tf.Graph()
    with g.as_default():
      l = p.Instantiate()
      x = tf.random.normal(shape=[4, 16])
      y = l.FPropDefaultTheta(x)
      loss = tf.reduce_sum(tf.square(y))
      _ = tf.gradients(ys=loss, xs=x)

      act, dact = l.fetch.activation, l.fetch.gradient

    with self.session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      act_v, dact_v = sess.run([act, dact])

    # The last layer two layers is sum(square(relu(act))).
    # So the dact is simply 2*relu(act).
    self.assertAllClose(2 * np.maximum(0, act_v), dact_v)

  def testBatchParallel(self):
    # pyformat: disable
    b = builder.Base.Params()
    b = b.Instantiate()
    p = b._BatchParallel(
        'bp',
        b._Seq(
            'main',
            b._Linear('l', 8, 4),
            b._PrintShape('debug'),
            b._Bias('b', 4)))
    # pyformat: enable
    g = tf.Graph()
    with g.as_default():
      l = p.Instantiate()
      x = tf.random.normal(shape=[4, 8])
      with cluster_factory.ForTestingWorker(cpus=4, split_size=1):
        y1 = l.FPropDefaultTheta(x)
      with cluster_factory.ForTestingWorker(cpus=4, split_size=2):
        y2 = l.FPropDefaultTheta(x)
      with cluster_factory.ForTestingWorker(cpus=4, split_size=4):
        y4 = l.FPropDefaultTheta(x)

    cfg = tf.config_pb2.ConfigProto()
    cfg.device_count['CPU'] = 4
    with self.session(config=cfg, graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      v1, v2, v4 = sess.run([y1, y2, y4])

    self.assertAllClose(v1, v2)
    self.assertAllClose(v1, v4)

  def testFn(self):
    b = builder.Base.Params()
    b = b.Instantiate()

    p = b._Fn('fn', lambda x, y: x + y, fn_out=lambda x, y: x)

    meta = p.cls.FPropMeta(p, tshape.Shape([4, 6]), tshape.Shape([4, 6]))
    self.assertEqual(meta.flops, 48)
    self.assertEqual(meta.out_shapes[0].ToTensorShape().as_list(), [4, 6])

    g = tf.Graph()
    with g.as_default():
      l = p.Instantiate()
      x = tf.random.normal(shape=[4, 8])
      y = tf.random.normal(shape=[4, 1])
      z = l.FPropDefaultTheta(x, y)

    with self.session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      v = sess.run([x, y, z])

    self.assertAllClose(v[0] + v[1], v[2])

  def testRematerialize(self):
    # Test the dropout consistency between fprop and bprop.
    b = builder.Base.Params()
    b = b.Instantiate()
    start_block = layers.DeterministicDropoutLayer.Params().Set(
        name='start_dropout', keep_prob=0.7)
    # Build 4 dropout layers, each wrapped by RematerializeFn.
    num_blocks = 4
    blocks = []
    blocks_per_cell = 2
    for i in range(num_blocks):
      blocks.append(layers.DeterministicDropoutLayer.Params().Set(
          name='dropout_{}'.format(i), keep_prob=0.7))
    cells = []
    while blocks:
      heads, blocks = blocks[:blocks_per_cell], blocks[blocks_per_cell:]
      cell_name = 'cell_{}'.format(len(cells))
      cells.append(
          b._Rematerialize(name=cell_name, body=b._Seq(cell_name, *heads)))
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.random.set_seed(12345)
      p = b._Seq('test', start_block, *cells)
      mdl = p.Instantiate()
      # y = mdl.Frop(x * w)
      # Fake input
      x = tf.ones([4, 5])
      # Construct weights.
      w = tf.get_variable(
          'w', shape=[4, 5], initializer=tf.constant_initializer([[1] * 5] * 4))
      y = mdl.FPropDefaultTheta(x * w)
      # Construct loss function such that gradients = final activation.
      # dy/dw = y = mdl.Frop(x * w) when w is 1.
      loss = tf.reduce_sum(y)
      grads = py_utils.ComputeGradients(loss, py_utils.NestedMap(w=w))
      tf.global_variables_initializer().run()
      y_val, grads_val = sess.run([y, grads.Transform(tuple)])
      grads_val = grads_val['w'][1]
      self.assertAllClose(y_val, grads_val)
      self.assertEqual(py_utils.GetStepSeed().eval(), 441070709)

  def testFnDefaultMeta(self):
    b = builder.Base.Params()
    b = b.Instantiate()

    def Foo(x, y):
      return x * x, y * 2

    p = b._Fn('fn', Foo)

    meta = p.cls.FPropMeta(p, tshape.Shape([4, 6]), tshape.Shape([3, 3]))
    self.assertEqual(meta.flops, 33)
    self.assertEqual(meta.out_shapes[0].ToTensorShape().as_list(), [4, 6])
    self.assertEqual(meta.out_shapes[1].ToTensorShape().as_list(), [3, 3])

    g = tf.Graph()
    with g.as_default():
      l = p.Instantiate()
      x = tf.random.normal(shape=[4, 8])
      y = tf.random.normal(shape=[3, 3])
      z0, z1 = l.FPropDefaultTheta(x, y)

    with self.session(graph=g) as sess:
      sess.run(tf.global_variables_initializer())
      vx, vy, vz0, vz1 = sess.run([x, y, z0, z1])

    self.assertAllClose(vx * vx, vz0)
    self.assertAllClose(vy * 2, vz1)

  def testSymbolicDims(self):
    p = builder.Base.Params()
    b = p.Instantiate()

    f1 = tshape.Shape(['kh', 'kw', 'idims', 'odims'])
    kh, kw, idims, odims = f1
    f2 = tshape.Shape([kh, kw, odims, odims])
    p = b._Seq('test', b._Conv2D('conv', f1, (2, 2)),
               b._Conv2D('conv', f2, (2, 2)), b._Bias('bias', odims))

    inp = tshape.Shape(['b', 'h', 'w', idims])
    b, h, w, _ = inp
    meta = p.cls.FPropMeta(p, inp)
    print('flops = ', meta.flops)
    out = meta.out_shapes[0]
    print('outputs = ', out)

    # sympy.lambdify can help us to do faster numerical evaluation.
    # Might be useful to build a "cost" model given a builder layer.
    f = sympy.lambdify([b, h, w, kh, kw, idims, odims], meta.flops, 'numpy')
    print('f.source = ', inspect.getsource(f))
    self.assertEqual(f(8, 224, 224, 3, 3, 8, 32), 925646848)
    self.assertEqual(f(8, 224, 224, 5, 5, 8, 32), 2569814016)

  def testDoEval(self):
    p = builder.Base.Params().Instantiate()._Dropout('dropout', 0.5)

    g = tf.Graph()
    with g.as_default():
      l = p.Instantiate()
      x = tf.random.normal(shape=[16, 16])
      # FProp three times each with different do_eval mode.
      with cluster_factory.SetEval(mode=None):
        a = l.FPropDefaultTheta(x)
      with cluster_factory.SetEval(mode=False):
        b = l.FPropDefaultTheta(x)
      with cluster_factory.SetEval(mode=True):
        c = l.FPropDefaultTheta(x)

    with self.session(graph=g) as sess:
      x, a, b, c = sess.run([x, a, b, c])

    self.assertGreater(np.linalg.norm(x - a), 0)
    self.assertGreater(np.linalg.norm(x - b), 0)
    self.assertAllEqual(x, c)


if __name__ == '__main__':
  tf.test.main()
