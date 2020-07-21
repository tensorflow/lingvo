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
"""Tests for lingvo gpipe."""

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core import tshape
from lingvo.core.gpipe import FeatureExtractionLayer
from lingvo.core.gpipe import PartitionSequentialLayers
from lingvo.core.gpipe import PipeliningLayer
from lingvo.core.layers import Conv2DLayerNoPadding
from lingvo.core.layers import FetchLayer


class _TimestepAccumulator(base_layer.Accumulator):
  """Simple accumulator for counting timesteps in the pipeline."""

  def DefaultValue(self):
    return tf.convert_to_tensor(0.0)

  def Increment(self):
    self.SetValue(self.GetValue() + 1.0)


class _SimpyLayer(base_layer.BaseLayer):
  """Simpy Layer with accumulator that counts time step."""

  def __init__(self, params):
    super().__init__(params)
    conv = Conv2DLayerNoPadding.Params().Set(
        name='conv',
        filter_shape=(3, 3, 1, 1),
        filter_stride=(1, 1),
        params_init=py_utils.WeightInit.Constant(0.1))
    self.CreateChild('conv', conv)
    self.RegisterAccumulator('ts_count', _TimestepAccumulator())

  def FProp(self, theta, inputs):
    p = self.params
    with tf.name_scope(p.name):
      inputs = self.conv.FProp(theta.conv, inputs)
      self.accumulators.ts_count.Increment()
      return tf.nn.relu(inputs)

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes((inputs,))
    return py_utils.NestedMap(flops=1, out_shapes=(inputs,))


class _SimpyLayerWithNestedMapInput(_SimpyLayer):

  def FProp(self, theta, inputs):
    p = self.params
    with tf.name_scope(p.name):
      vec = self.conv.FProp(theta.conv, inputs.vec)
      return py_utils.NestedMap(vec=vec, paddings=inputs.paddings)

  @classmethod
  def FPropMeta(cls, p, inputs):
    py_utils.CheckShapes(
        tuple(inputs.Filter(lambda x: x is not None).Flatten()))
    return py_utils.NestedMap(flops=1, out_shapes=(inputs,))


def _Partition(params, num_splits, *shapes):
  seqs = PartitionSequentialLayers(params, num_splits, *shapes)
  return [
      FeatureExtractionLayer.Params().Set(name='d%d' % i, sub=seqs[i].sub)
      for i in range(len(seqs))
  ]


def _BuildDummyPipelineCnn(num_splits=4,
                           num_micro_batches=8,
                           micro_batch_size=None):
  """Construct a dummy layer that consist of 16 3x3 conv layers.

  In addition, each conv layer increments a count every time step.

  Args:
    num_splits: number of cells for pipeline cnn
    num_micro_batches: number of time steps.
    micro_batch_size: Size of a micro batch.

  Returns:
    A PipeliningLayer layer.
  """
  assert num_splits in [1, 2, 4, 8, 16]
  num_layers = 16
  layers = []
  for i in range(num_layers):
    layers.append(_SimpyLayer.Params().Set(name='layer_{}'.format(i)))

  if num_splits == 1:
    p = FeatureExtractionLayer.Params().Set(name='seq', sub=layers)
  else:
    cell_tpl = []
    layers_per_split = num_layers // num_splits
    num_act_outputs = 0
    num_act_inputs = 0
    act_fetch_layers = None
    for split in range(num_splits):
      sub = layers[split * layers_per_split:(split + 1) * layers_per_split]
      if split == 0:
        sub.append(FetchLayer.Params().Set(name='fetch'))
        num_act_outputs = 1
        act_fetch_layers = ['fetch']
      else:
        num_act_inputs = 1
        act_fetch_layers = []
      split_layer = FeatureExtractionLayer.Params().Set(
          name='split_{}'.format(split),
          sub=sub,
          act_fetch_layers=act_fetch_layers,
          num_act_inputs=num_act_inputs,
          num_act_outputs=num_act_outputs)
      cell_tpl.append(split_layer)
    p = PipeliningLayer.Params().Set(
        name='pipeline',
        num_micro_batches=num_micro_batches,
        micro_batch_size=micro_batch_size,
        cell_tpl=cell_tpl,
        before_tpl=[])
  layer = p.Instantiate()
  return layer


class DummyPipelineCnnTest(test_utils.TestCase):

  def _verify_timestep_counts(self,
                              num_splits,
                              auto_partition=False,
                              micro_batch_size=None):
    num_micro_batches = 8
    batch_size = 16
    with self.session(graph=tf.Graph()):
      tf.random.set_seed(1245)
      inputs = tf.random.uniform([batch_size, 8, 8, 1], seed=12345)
      if auto_partition:
        layers = [
            _SimpyLayer.Params().Set(name='layer_{}'.format(i))
            for i in range(16)
        ]
        net = PipeliningLayer.Params().Set(
            name='pipeline',
            num_micro_batches=num_micro_batches,
            cell_tpl=_Partition(layers, num_splits,
                                tshape.Shape([batch_size, 8, 8,
                                              1]))).Instantiate()
      else:
        net = _BuildDummyPipelineCnn(
            num_splits=num_splits,
            micro_batch_size=micro_batch_size,
            num_micro_batches=num_micro_batches)
      endpoints = net.FPropDefaultTheta(inputs)
      if isinstance(endpoints, (list, tuple)):
        logits, aux_logits = endpoints
      else:
        logits = endpoints
        aux_logits = None
      loss = tf.reduce_mean(logits)
      grads = tf.gradients(loss, tf.trainable_variables())
      grad_norm = tf.sqrt(py_utils.SumSquared(grads))
      ts = net.GetAccumulatorValues().Flatten()

      self.evaluate(tf.global_variables_initializer())
      grad_norm_val, ts_vals = self.evaluate([grad_norm, ts])
      test_utils.CompareToGoldenSingleFloat(self, 0.268087, grad_norm_val)
      # Accumulator values should be equal to number of time steps in pipeline.
      for ts_val in list(ts_vals):
        expected_ts = num_micro_batches if num_splits > 1 else 1
        self.assertEqual(ts_val, expected_ts)
      if aux_logits is not None:
        aux_logit_tensor = self.evaluate(aux_logits)
        self.assertEqual(aux_logit_tensor.shape, (batch_size, 8, 8, 1))

  def testDummyPipelineCnnOneSplit(self):
    self._verify_timestep_counts(num_splits=1)

  def testDummyPipelineCnnTwoSplits(self):
    self._verify_timestep_counts(num_splits=2)

  def testDummyPipelineCnnTwoSplitsMicroBatchSize1(self):
    with self.assertRaises(ValueError):
      self._verify_timestep_counts(num_splits=2, micro_batch_size=1)

  def testDummyPipelineCnnTwoSplitsMicroBatchSize2(self):
    self._verify_timestep_counts(num_splits=2, micro_batch_size=2)

  def testDummyPipelineCnnFourSplits(self):
    self._verify_timestep_counts(num_splits=4)

  def testDummyPipelineCnnAutoPartitionTwoSplits(self):
    self._verify_timestep_counts(num_splits=2, auto_partition=True)

  def testDummyPipelineCnnAutoPartitionFourSplits(self):
    self._verify_timestep_counts(num_splits=4, auto_partition=True)

  def testDummyPipelineCnnNestedMapInput(self):
    batch_size = 16
    num_layers = 4
    cells = []
    with self.session(graph=tf.Graph()):
      for i in range(num_layers):
        cells.append(_SimpyLayerWithNestedMapInput.Params().Set(
            name='layer_{}'.format(i)))
      p = PipeliningLayer.Params().Set(
          name='pipeline',
          num_micro_batches=8,
          micro_batch_size=2,
          nested_map_fprop=True,
          cell_tpl=cells,
          before_tpl=[])
      layer = p.Instantiate()
      tf.random.set_seed(1245)
      inputs = tf.random.uniform([batch_size, 8, 8, 1], seed=12345)
      outputs = layer.FPropDefaultTheta(
          py_utils.NestedMap(vec=inputs, paddings=None))
      self.evaluate(tf.global_variables_initializer())
      self.evaluate(outputs.vec)
      self.assertEqual(outputs.vec.shape, (batch_size, 8, 8, 1))


if __name__ == '__main__':
  tf.test.main()
