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
"""Tests for builder_layers."""

from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import builder_layers as layers
from lingvo.core import cluster_factory
from lingvo.core import layers as lingvo_layers
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core import tshape
import numpy as np


class FCLayerTestNestedMapFPropInput(lingvo_layers.FCLayer):
  """lingvo_layers.FCLayer with nested map as input signature in FProp.

  This is for testing compliance of RepeatLayer with NestedMap inputs in FProp.
  """

  def FProp(self, theta, in_nmap):
    """Overriding FProp input signature for FCLayer.

    Args:
     theta: NestedMap containing weights of layer.
     in_nmap: NestedMap containing at least the following:
      - features: The inputs tensor. Shaped [..., input_dim].
      - paddings: The paddings tensor. Shaped [..., 1], where all but the last
        dimension match.

    Returns:
     out_nmap: NestedMap containing the following:
      - features: Output after applying projection (see super() for details).
      - paddings: Output (unused) paddings.
    """
    outputs = super().FProp(theta, in_nmap.features, in_nmap.paddings)
    in_nmap.features = outputs
    return in_nmap


class BuilderLayerTest(test_utils.TestCase, parameterized.TestCase):

  def testCreateNestedMapLayerFProp(self):
    with self.session():
      x = tf.constant(1)
      y = tf.constant(2)
      params = layers.CreateNestedMapLayer.Params().Set(
          name='map', keys=['x', 'y'])
      layer = params.Instantiate()
      layer_out = self.evaluate(layer.FPropDefaultTheta(x, y))
      self.assertEqual(1, layer_out.x)
      self.assertEqual(2, layer_out.y)

  def testFirstNLayerFProp(self):
    with self.session():
      params = layers.FirstNLayer.Params()
      params.name = 'fn'
      params.n = 2

      fn_layer = layers.FirstNLayer(params)
      a = tf.constant(1)
      b = tf.constant(2)
      c = tf.constant(3)

      fn_out = self.evaluate(fn_layer.FPropDefaultTheta(a, b, c))

      self.assertEqual((1, 2), fn_out)

  def testArgIndexLayerFProp(self):
    with self.session():
      params = layers.ArgIndexLayer.Params().Set(name='argidx', idx=[1, 3])
      argidx_layer = layers.ArgIndexLayer(params)
      args = [tf.constant(i) for i in range(5)]

      argidx_out = self.evaluate(argidx_layer.FPropDefaultTheta(*args))
      self.assertEqual((1, 3), argidx_out)

  def testSequentialLayer(self):
    g = tf.Graph()
    with g.as_default(), self.SetEval(True):
      tf.random.set_seed(24332)
      p = layers.SequentialLayer.Params().Set(
          name='seq',
          repeat=2,
          sub=[
              lingvo_layers.FCLayer.Params().Set(
                  name='foo', input_dim=32, output_dim=8),
              lingvo_layers.FCLayer.Params().Set(
                  name='bar', input_dim=8, output_dim=8),
              lingvo_layers.FCLayer.Params().Set(
                  name='baz', input_dim=8, output_dim=32),
              lingvo_layers.DropoutLayer.Params().Set(
                  name='dropout', keep_prob=0.5)
          ])
      l = p.Instantiate()
      x = tf.random.normal(shape=[2, 32])
      y = l.FPropDefaultTheta(x)
      l.vars.Transform(lambda x: x.shape).VLog(0, 'vars: ')

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      x_val, y_val, w = self.evaluate([x, y, l.vars])

    act = x_val
    # relu(act \dot w + b)
    for i in range(2):
      act = np.maximum(0, np.dot(act, w.rep[i].foo.w) + w.rep[i].foo.b)
      act = np.maximum(0, np.dot(act, w.rep[i].bar.w) + w.rep[i].bar.b)
      act = np.maximum(0, np.dot(act, w.rep[i].baz.w) + w.rep[i].baz.b)
    self.assertAllClose(act, y_val)

  def testEmptySequentialLayer(self):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(24332)
      p = layers.SequentialLayer.Params().Set(name='seq')
      l = p.Instantiate()
      x = tf.random.normal(shape=[2, 32])
      y = l.FPropDefaultTheta(x)
      self.assertIsInstance(y, tf.Tensor)

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      x_val, y_val = self.evaluate([x, y])
      self.assertAllEqual(x_val, y_val)

  def testEmptySequentialLayerFPropMeta(self):
    g = tf.Graph()
    with g.as_default():
      p = layers.SequentialLayer.Params().Set(name='seq')
      l = p.Instantiate()
      x = py_utils.NestedMap(val=tf.random.normal(shape=[2, 32]))
      y = l.FPropDefaultTheta(x)
      self.assertIsInstance(y.val, tf.Tensor)
      y_shape = l.FPropMeta(
          p, py_utils.Transform(lambda t: tshape.Shape(t.shape),
                                x)).out_shapes[0]
      self.assertEqual(y.val.shape.as_list(),
                       y_shape.val.ToTensorShape().as_list())

  def testUnarySequentialLayer(self):
    g = tf.Graph()
    with g.as_default(), self.SetEval(True):
      tf.random.set_seed(24332)
      p = layers.UnarySequentialLayer.Params().Set(
          name='seq',
          sub=[
              lingvo_layers.FCLayer.Params().Set(
                  name='foo', input_dim=32, output_dim=8),
              lingvo_layers.FCLayer.Params().Set(
                  name='bar', input_dim=8, output_dim=8),
              lingvo_layers.FCLayer.Params().Set(
                  name='baz', input_dim=8, output_dim=32),
              lingvo_layers.DropoutLayer.Params().Set(
                  name='dropout', keep_prob=0.5)
          ])
      l = p.Instantiate()
      x = tf.random.normal(shape=[2, 32])
      y = l.FPropDefaultTheta(x)
      l.vars.Transform(lambda x: x.shape).VLog(0, 'vars: ')

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      x_val, y_val, w = self.evaluate([x, y, l.vars])

    act = x_val
    # relu(act \dot w + b)
    act = np.maximum(0, np.dot(act, w.foo.w) + w.foo.b)
    act = np.maximum(0, np.dot(act, w.bar.w) + w.bar.b)
    act = np.maximum(0, np.dot(act, w.baz.w) + w.baz.b)
    self.assertAllClose(act, y_val)

  def testParallelLayer(self):
    g = tf.Graph()
    with g.as_default(), self.SetEval(True):
      tf.random.set_seed(24332)
      p = layers.ParallelLayer.Params().Set(
          name='test',
          merge=lambda xs: tuple([tf.add_n(x) for x in zip(*xs)]),
          sub=[
              lingvo_layers.FCLayer.Params().Set(
                  name='foo', input_dim=32, output_dim=4),
              lingvo_layers.FCLayer.Params().Set(
                  name='bar', input_dim=32, output_dim=4),
              layers.SequentialLayer.Params().Set(
                  name='seq',
                  sub=[
                      lingvo_layers.FCLayer.Params().Set(
                          name='baz', input_dim=32, output_dim=4),
                      lingvo_layers.DropoutLayer.Params().Set(
                          name='dropout', keep_prob=0.5)
                  ])
          ])
      l = p.Instantiate()
      x = tf.random.normal(shape=[2, 32])
      y = l.FPropDefaultTheta(x)

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      x_val, y_val, w = self.evaluate([x, y, l.vars])

    out = []
    act = x_val
    # relu(act \dot w + b)
    out += [np.maximum(0, np.matmul(act, w.foo.w) + w.foo.b)]
    self.assertEqual(out[-1].shape, (2, 4))
    out += [np.maximum(0, np.matmul(act, w.bar.w) + w.bar.b)]
    self.assertEqual(out[-1].shape, (2, 4))
    out += [np.maximum(0, np.matmul(act, w.seq.baz.w) + w.seq.baz.b)]
    self.assertEqual(out[-1].shape, (2, 4))

    np_result = out[0]
    for v in out[1:]:
      np_result = np.add(np_result, v)
    self.assertAllClose(np_result, y_val)

  def testParallelMatmulLayer(self):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(24332)

      def MergeFn(xs):
        result = []
        for x in zip(*xs):
          val = x[0]
          for v in x[1:]:
            val = tf.matmul(val, v)
          result.append(val)
        return tuple(result)

      p = layers.ParallelLayer.Params().Set(
          name='parallel',
          merge=MergeFn,
          sub=[
              lingvo_layers.FCLayer.Params().Set(
                  name='foo', input_dim=32, output_dim=4),
              lingvo_layers.FCLayer.Params().Set(
                  name='bar', input_dim=32, output_dim=4),
              lingvo_layers.FCLayer.Params().Set(
                  name='baz', input_dim=32, output_dim=4)
          ])
      l = p.Instantiate()
      x = tf.random.normal(shape=[2, 4, 32])
      y = l.FPropDefaultTheta(x)

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      x_val, y_val, w = self.evaluate([x, y, l.vars])

    out = []
    act = x_val
    # relu(act \dot w + b)
    out += [np.maximum(0, np.matmul(act, w.foo.w) + w.foo.b)]
    self.assertEqual(out[-1].shape, (2, 4, 4))
    out += [np.maximum(0, np.matmul(act, w.bar.w) + w.bar.b)]
    self.assertEqual(out[-1].shape, (2, 4, 4))
    out += [np.maximum(0, np.matmul(act, w.baz.w) + w.baz.b)]
    self.assertEqual(out[-1].shape, (2, 4, 4))

    np_result = out[0]
    for v in out[1:]:
      np_result = np.matmul(np_result, v)
    self.assertAllClose(np_result, y_val, atol=1e-5, rtol=1e-5)

  def testParalellMultiOutputsLayer(self):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(24332)

      def Merge(xs):
        rets = []
        for x in zip(*xs):
          if x[0] is None:
            rets.append(None)
          else:
            rets.append(tf.add_n(list(x)))
        return tuple(rets)

      p = layers.ParallelLayer.Params().Set(
          name='parallel',
          merge=Merge,
          sub=[
              lingvo_layers.ConvLayer.Params().Set(
                  name='p%d' % i,
                  filter_shape=(3, 3, 3, 5),
                  filter_stride=(1, 1),
                  batch_norm=False) for i in range(3)
          ])
      l = p.Instantiate()
      x = tf.zeros(shape=[2, 32, 32, 3])
      y0, y1 = l.FPropDefaultTheta(x)
      y_sum = tf.reduce_sum(y0)
      # Ensures the 2nd return value (None) are handled properly.
      self.assertEqual(None, y1)

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      y_sum_val = self.evaluate(y_sum)

    self.assertEqual(y_sum_val, 0.)

  def testMapLayer(self):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(24332)
      p = layers.MapLayer.Params().Set(
          name='map', fn=tf.reduce_max, kwargs={'axis': 1})
      l = p.Instantiate()
      x0, x1 = [tf.random.normal(shape=[2, 3, 5])] * 2
      y0, y1 = l.FPropDefaultTheta(x0, x1)

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      vx0, vx1, vy0, vy1 = self.evaluate([x0, x1, y0, y1])

    self.assertAllClose(np.max(vx0, 1), vy0)
    self.assertAllClose(np.max(vx1, 1), vy1)

  def testLinearLayer(self):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(24332)
      p = layers.LinearLayer.Params().Set(
          name='test', input_dims=10, output_dims=5)
      l = p.Instantiate()
      xs = []
      ys = []
      for shape in ([2, 10], [2, 3, 10], [2, 3, 5, 10], [2, 3, 5, 7, 10]):
        x = tf.random.normal(shape=shape)
        y = l.FPropDefaultTheta(x)
        xs += [x]
        ys += [y]

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      xs_val, ys_val, w_val = self.evaluate([xs, ys, l.vars])

    self.assertEqual(w_val.w.shape, (10, 5))
    for (xv, yv) in zip(xs_val, ys_val):
      self.assertAllClose(np.matmul(xv, w_val.w), yv)

  def testBiasLayer(self):
    g = tf.Graph()
    with g.as_default():
      tf.random.set_seed(24332)
      p = layers.BiasLayer.Params().Set(name='test', dims=10)
      l = p.Instantiate()
      x = tf.random.normal(shape=[2, 10])
      y = l.FPropDefaultTheta(x)

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      x_val, y_val, w_val = self.evaluate([x, y, l.vars])

    self.assertEqual(w_val.b.shape, (10,))
    self.assertAllClose(x_val + w_val.b, y_val)

  def testGraphTensors(self):
    graph_tensors = layers.GraphTensors()
    graph_tensors.StoreTensor('t',
                              py_utils.NestedMap(a=py_utils.NestedMap(b='c')))
    self.assertEqual('c', graph_tensors.GetTensor('t.a.b'))

  def testSignatureParsing(self):
    sig = layers.GraphSignature('a,b->c')
    self.assertEqual(['a', 'b'], sig.inputs)
    self.assertEqual(['c'], sig.outputs)

    sig = layers.GraphSignature('[a,b],d->c')
    self.assertEqual([['a', 'b'], 'd'], sig.inputs)
    self.assertEqual(['c'], sig.outputs)

    # also test nested structures, like nested lists and dicts.
    sig = layers.GraphSignature('(x=a,y=b)->c')
    self.assertEqual([{'x': 'a', 'y': 'b'}], sig.inputs)
    self.assertEqual(['c'], sig.outputs)

    # Make sure that empty lists and dicts work.
    sig = layers.GraphSignature('(x=[]),()->d')
    self.assertEqual([{'x': []}, {}], sig.inputs)

    sig = layers.GraphSignature('(x=a,y=[f,(z=g.h)]),[d,e]->j')
    self.assertEqual([{
        'x': 'a',
        'y': ['f', {
            'z': 'g.h'
        }]
    }, ['d', 'e']], sig.inputs)
    self.assertEqual(['j'], sig.outputs)

  def testGraphLayer(self):
    g = tf.Graph()
    with g.as_default(), self.SetEval(True):
      tf.random.set_seed(24332)

      def _FnMeta(*shapes):
        return py_utils.NestedMap(flops=1, out_shapes=shapes)

      p = layers.GraphLayer.Params().Set(
          name='graph',
          input_endpoints=['x'],
          output_endpoints=['y'],
          sub=[
              ('x.a->y.c',
               layers.FnLayer.Params().Set(fn=lambda x: 2 * x,
                                           fn_meta=_FnMeta)),
              ('x.b->y.d', layers.FnLayer.Params().Set(
                  name='bar', fn=lambda x: x + 2, fn_meta=_FnMeta)),
              ('y.c,y.d->y.e, y.f', layers.FnLayer.Params().Set(
                  name='baz', fn=lambda x, y: (x + y, x - y), fn_meta=_FnMeta)),
          ])
      l = p.Instantiate()
      x = py_utils.NestedMap(a=tf.constant(1.0), b=tf.constant(2.0))
      y = l.FProp(l.theta, x)
      y_shape = l.FPropMeta(
          p, py_utils.Transform(lambda t: tshape.Shape(t.shape),
                                x)).out_shapes[0]
      self.assertDictEqual(
          py_utils.Transform(lambda t: t.shape.as_list(), y),
          py_utils.Transform(lambda t: t.ToTensorShape().as_list(), y_shape))

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      y_val = self.evaluate(y)
      print(y_val)
      self.assertEqual(py_utils.NestedMap(c=2.0, d=4.0, e=6.0, f=-2.0), y_val)

  def testGraphLayerReusedLayers(self):
    g = tf.Graph()
    with g.as_default(), self.SetEval(True):
      tf.random.set_seed(24332)

      p = layers.GraphLayer.Params().Set(
          name='graph',
          input_endpoints=['x'],
          output_endpoints=['y'],
          sub=[
              # These three lines are the same as testGraphLayer.
              ('x.a->y.c', 0),
              ('x.b->y.d', 1),
              ('y.c,y.d->y.e,y.f', 2),
              # But here, we run layer 0 again to generate y.g.
              ('y.f->y.g', 0),
          ],
          sub_layers=[
              layers.FnLayer.Params().Set(fn=lambda x: 2 * x),
              layers.FnLayer.Params().Set(name='bar', fn=lambda x: x + 2),
              layers.FnLayer.Params().Set(
                  name='baz', fn=lambda x, y: (x + y, x - y)),
          ])
      l = p.Instantiate()
      x = py_utils.NestedMap(a=tf.constant(1.0), b=tf.constant(2.0))
      y = l.FProp(l.theta, x)

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      y_val = self.evaluate(y)
      print(y_val)
      self.assertEqual(
          py_utils.NestedMap(c=2.0, d=4.0, e=6.0, f=-2.0, g=-4.0), y_val)

  def testSoftCondLayer(self):
    num_experts = 100
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(24332)
      p = layers.SoftCondLayer.Params().Set(
          name='soft_cond',
          cond_dim=2,
          num_experts=num_experts,
          body=lingvo_layers.FCLayer.Params().Set(input_dim=2, output_dim=2))
      l = p.Instantiate()
      x = tf.random.normal(shape=[1, 2, 2])
      y = l.FPropDefaultTheta(x)
      self.evaluate(tf.global_variables_initializer())
      x_val, y_val, vars_val = self.evaluate([x, y, l.vars])

      p_nz = layers.SoftCondLayer.Params().Set(
          name='soft_cond_nonzeros',
          cond_dim=2,
          num_experts=num_experts,
          nonzeros_mean=True,
          body=lingvo_layers.FCLayer.Params().Set(input_dim=2, output_dim=2))
      l_nz = p_nz.Instantiate()
      x_nz = tf.random.normal(shape=[1, 2, 2])
      y_nz = l_nz.FPropDefaultTheta(x_nz)
      self.evaluate(tf.global_variables_initializer())
      x_nz_val, y_nz_val, vars_nz_val = self.evaluate([x_nz, y_nz, l_nz.vars])

    np_val = x_val[0]
    np_nz_val = x_nz_val[0]
    taks_weight = np.exp(-1.0 * np.dot(np.mean(np_val, 0), vars_val.w))
    taks_weight = 1.0 / (1.0 + taks_weight)
    nzs = np.count_nonzero(np_nz_val, 0).astype('float32') + 1e-10
    taks_weight_nz = np.exp(-1.0 *
                            np.dot(np.sum(np_nz_val, 0) / nzs, vars_nz_val.w))
    taks_weight_nz = 1.0 / (1.0 + taks_weight_nz)
    weighted_weight = np.einsum('i,ijk->jk', taks_weight, vars_val.body.w)
    weighted_weight_nz = np.einsum('i,ijk->jk', taks_weight_nz,
                                   vars_nz_val.body.w)
    weighted_bias = np.einsum('i,ij->j', taks_weight, vars_val.body.b)
    weighted_bias_nz = np.einsum('i,ij->j', taks_weight_nz, vars_nz_val.body.b)
    np_val_out = np.maximum(0, np.dot(np_val, weighted_weight) + weighted_bias)
    np_val_out_nz = np.maximum(
        0,
        np.dot(np_nz_val, weighted_weight_nz) + weighted_bias_nz)
    self.assertAllClose(np_val_out, y_val[0])
    self.assertAllClose(np_val_out_nz, y_nz_val[0])

  def testRepeatLayer(self):
    repeat = 100
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(24332)
      p = layers.RepeatLayer.Params().Set(
          name='recurrent',
          repeat=repeat,
          body=lingvo_layers.FCLayer.Params().Set(input_dim=2, output_dim=2))
      l = p.Instantiate()
      x = tf.random.normal(shape=[2, 2])
      y = l.FPropDefaultTheta(x)
      self.evaluate(tf.global_variables_initializer())
      x_val, y_val, w = self.evaluate([x, y, l.vars])

    np_val = x_val

    # relu(act \dot w + b)
    for i in range(repeat):
      np_val = np.maximum(0, np.dot(np_val, w.body.w[i]) + w.body.b[i])
    self.assertAllClose(np_val, y_val)

  @parameterized.parameters(('eval_only', True), ('always', False))
  def testRepeatLayerUnrolledEval(self, unroll, do_eval):
    repeat = 100
    with cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', do_eval=do_eval):
      tf.random.set_seed(24332)
      p = layers.RepeatLayer.Params().Set(
          name='recurrent',
          repeat=repeat,
          per_layer_vars=True,
          unroll=unroll,
          body=lingvo_layers.FCLayer.Params().Set(input_dim=2, output_dim=2))
      l = p.Instantiate()
      x = tf.random.normal(shape=[2, 2])
      y = l.FPropDefaultTheta(x)
      self.evaluate(tf.global_variables_initializer())
      x_val, y_val, w = self.evaluate([x, y, l.vars])

    np_val = x_val

    # relu(act \dot w + b)
    for i in range(repeat):
      body_i = w['body_iter_%05d' % i]
      np_val = np.maximum(0, np.dot(np_val, body_i.w) + body_i.b)
    self.assertAllClose(np_val, y_val)

  def testRepeatLayerNestedMapFPropInputSignature(self):
    """Tests RepeatLayer having body layer with NestedMap in FProp signature."""
    repeat = 100
    input_dim, output_dim = 2, 2
    # Reference RepeatLayer.
    ref_p = layers.RepeatLayer.Params().Set(
        name='ref_recurrent',
        repeat=repeat,
        body=lingvo_layers.FCLayer.Params().Set(
            input_dim=input_dim, output_dim=output_dim))
    # RepeatLayer with NestedMap in `body` FProp input signature.
    new_p = layers.RepeatLayer.Params().Set(
        name='nested_map_recurrent',
        repeat=repeat,
        body=FCLayerTestNestedMapFPropInput.Params().Set(
            input_dim=input_dim, output_dim=output_dim))
    # Verify FProp output equality for both layers.
    ref_layer = ref_p.Instantiate()
    new_layer = new_p.Instantiate()
    assign_op = [
        tf.assign(dst, src)
        for (src,
             dst) in zip(ref_layer.vars.Flatten(), new_layer.vars.Flatten())
    ]
    with self.session() as sess:
      tf.random.set_seed(24332)
      sess.run(tf.global_variables_initializer())
      sess.run(assign_op)
      inputs = tf.random.normal(shape=[2, 2])
      paddings = tf.zeros((2, 1))
      ref_outputs = ref_layer.FPropDefaultTheta(inputs)
      new_out_nmap = new_layer.FPropDefaultTheta(
          py_utils.NestedMap(features=inputs, paddings=paddings))
      ref_out_vals = sess.run(ref_outputs)
      new_out_vals = sess.run(new_out_nmap.features)
      self.assertAllClose(ref_out_vals, new_out_vals)

  def testRepeatLayerNestedMapBProp(self):
    """Tests RepeatLayer having body layer with mutable NestedMap."""
    repeat = 3
    input_dim, output_dim = 2, 2
    # RepeatLayer with NestedMap in `body` FProp input signature.
    p = layers.RepeatLayer.Params().Set(
        name='nested_map_recurrent',
        repeat=repeat,
        body=FCLayerTestNestedMapFPropInput.Params().Set(
            input_dim=input_dim, output_dim=output_dim))
    # Verify FProp output equality for both layers.
    layer = p.Instantiate()
    with self.session() as sess:
      tf.random.set_seed(24332)
      sess.run(tf.global_variables_initializer())
      inputs = tf.random.normal(shape=[2, 5, 2])
      paddings = tf.zeros((2, 5, 1))
      args = py_utils.NestedMap(features=inputs, paddings=paddings)
      outputs = layer.FPropDefaultTheta(args)
      # Mutate 'args' before the bprop.
      args.features = tf.transpose(args.features, [1, 0, 2])
      args.paddings = tf.transpose(args.paddings, [1, 0, 2])
      in_grads = tf.gradients(ys=tf.nest.flatten(outputs), xs=[inputs])
      sess.run(in_grads)

  def testRepeatLayerNestedMapFPropInputRaisesErrorWithNoneInput(self):
    """Tests RepeatLayer raise ValueError with None values in input map."""
    repeat = 100
    # RpeatLayer with NestedMap in FProp input signature.
    p = layers.RepeatLayer.Params().Set(
        name='nested_map_recurrent',
        repeat=repeat,
        body=FCLayerTestNestedMapFPropInput.Params().Set(
            input_dim=2, output_dim=2))
    layer = p.Instantiate()
    with self.session() as sess:
      tf.random.set_seed(24332)
      sess.run(tf.global_variables_initializer())
      inputs = tf.random.normal(shape=[2, 2])
      # Set paddings to None.
      paddings = None
      with self.assertRaisesRegex(
          ValueError, 'Each value in the input NestedMap must be a tensor.'):
        layer.FPropDefaultTheta(
            py_utils.NestedMap(features=inputs, paddings=paddings))

  def testParallelRepeatLayerLayer(self):
    repeat = 100
    body_p = layers.SequentialLayer.Params().Set(
        name='body',
        sub=[
            layers.LinearLayer.Params().Set(
                name='ln1', input_dims=2, output_dims=4),
            layers.FnLayer.Params().Set(
                name='relu',
                fn=tf.nn.relu,
                fn_meta=lambda x: py_utils.NestedMap(flops=1, out_shapes=(x,))),
            layers.LinearLayer.Params().Set(
                name='ln2', input_dims=4, output_dims=2)
        ])
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(24332)
      p = layers.ParallelRepeatLayer.Params().Set(
          name='moe', repeat=repeat, body=body_p)
      l = p.Instantiate()
      x = tf.random.normal(shape=[repeat, 2, 2])
      y = l.FPropDefaultTheta(x)
      self.evaluate(tf.global_variables_initializer())
      x_val, y_val, w = self.evaluate([x, y, l.vars])

    np_val = []

    for i in range(repeat):
      # relu(act \dot w_1) \dot w_2
      np_val.append(
          np.dot(
              np.maximum(0, np.dot(x_val[i], w.body.ln1.w[i])),
              w.body.ln2.w[i]))
    np_val = np.stack(np_val)
    self.assertAllClose(np_val, y_val)

  def testRematerializationLayer(self):
    with self.session(use_gpu=False, graph=tf.Graph()):
      tf.random.set_seed(24332)

      def MulSumFnMeta(x):
        return py_utils.NestedMap(flops=2, out_shapes=(x,))

      def AddFnMeta(x, y):
        del y
        return py_utils.NestedMap(flops=2, out_shapes=(x,))

      p = layers.GraphLayer.Params().Set(
          name='graph',
          input_endpoints=['a', 'b'],
          output_endpoints=['e'],
          sub=[
              ('a->c', layers.FnLayer.Params().Set(
                  fn=lambda x: 2 * x, fn_meta=MulSumFnMeta)),
              ('b->d', layers.FnLayer.Params().Set(
                  name='bar', fn=lambda x: x + 2, fn_meta=MulSumFnMeta)),
              ('c,d->e', layers.FnLayer.Params().Set(
                  name='baz', fn=lambda x, y: x + y, fn_meta=AddFnMeta)),
          ])
      p = layers.RematerializationLayer.Params().Set(name='remat', body=p)
      l = p.Instantiate()
      x = tf.constant(1.0)
      y = tf.constant(2.0)
      z = l.FProp(l.theta, x, y)
      self.evaluate(tf.global_variables_initializer())
      z_val = self.evaluate(z)
      print(z_val)
      self.assertAllClose(6.0, z_val)

  def testPrintShapeLayer(self):
    g = tf.Graph()
    with g.as_default():
      p = layers.PrintShapeLayer.Params().Set(name='test')
      l = p.Instantiate()
      x = tf.constant(1.0)
      y = l.FPropDefaultTheta(x)

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      x_val, y_val = self.evaluate([x, y])
    self.assertEqual(x_val, y_val)

  def testReshapeLayer(self):
    g = tf.Graph()
    with g.as_default():
      p = layers.ReshapeLayer.Params().Set(name='test', shape=[-1, 2, 1])
      l = p.Instantiate()
      x = tf.constant([[1.0, 2.0], [3.0, 2.0]])
      y = l.FPropDefaultTheta(x)

    with self.session(graph=g):
      self.evaluate(tf.global_variables_initializer())
      self.assertAllEqual(self.evaluate(tf.shape(x)), [2, 2])
      self.assertAllEqual(self.evaluate(tf.shape(y)), [2, 2, 1])


if __name__ == '__main__':
  tf.test.main()
