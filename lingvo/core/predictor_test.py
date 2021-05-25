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
"""Tests for lingvo.core.predictor."""

import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import inference_graph_exporter
from lingvo.core import inference_graph_pb2
from lingvo.core import predictor
from lingvo.core import test_utils


class DummyModel(base_model.BaseTask):

  def Inference(self):
    with tf.name_scope('inference'):
      feed1 = tf.placeholder(name='feed1_node', dtype=tf.float32, shape=[1])
      fetch1 = tf.identity(feed1, name='fetch1_node')
      feed2 = tf.placeholder(name='feed2_node', dtype=tf.float32, shape=[2])
      fetch2 = tf.identity(feed2, name='fetch2_node')
      inference_graph = inference_graph_pb2.InferenceGraph()
      subgraph = inference_graph.subgraphs['default']
      subgraph.feeds['feed1'] = feed1.name
      subgraph.fetches['fetch1'] = fetch1.name
      subgraph = inference_graph.subgraphs['subgraph2']
      subgraph.feeds['feed1'] = feed2.name
      subgraph.fetches['fetch1'] = fetch2.name
      return inference_graph


class PredictorTest(test_utils.TestCase):

  def _testInferenceGraph(self):
    p = base_model.SingleTaskModel.Params(DummyModel.Params().Set(name='test'))
    p.input = base_input_generator.BaseInputGenerator.Params().Set(name='test')
    inference_graph = inference_graph_exporter.InferenceGraphExporter.Export(p)
    return inference_graph

  def testPredictorFeedShapes(self):
    pred = predictor.Predictor(self._testInferenceGraph())
    self.assertEqual([1], pred.feed_shapes.feed1)
    self.assertEqual([2], pred.subgraph_feed_shapes('subgraph2').feed1)

  def testPredictorFetchShapes(self):
    pred = predictor.Predictor(self._testInferenceGraph())
    self.assertEqual([1], pred.fetch_shapes.fetch1)
    self.assertEqual([2], pred.subgraph_fetch_shapes('subgraph2').fetch1)

  def testPredictor(self):
    pred = predictor.Predictor(self._testInferenceGraph())
    fetch1 = pred.Run('fetch1', feed1=[12345])
    self.assertEqual(12345, fetch1)

  def testPredictorSubgraph(self):
    pred = predictor.Predictor(self._testInferenceGraph())
    fetch1 = pred.Run('fetch1', feed1=[12345, 23456], subgraph_name='subgraph2')
    self.assertAllEqual([12345, 23456], fetch1)

  def testPredictorNoLoadGraphDefFromInferenceGraph(self):
    p = base_model.SingleTaskModel.Params(DummyModel.Params().Set(name='test'))
    p.input = base_input_generator.BaseInputGenerator.Params().Set(name='test')
    pred = predictor.Predictor(
        p.Instantiate().GetTask().Inference(),
        load_graph_def_from_inference_graph=False)
    fetch1 = pred.Run('fetch1', feed1=[12345])
    self.assertEqual(12345, fetch1)

  def testMissingFeedRaisesInvalidArgumentError(self):
    pred = predictor.Predictor(self._testInferenceGraph())
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'feed1'):
      pred.Run('fetch1')

  def testInvalidFetchRaisesKeyError(self):
    pred = predictor.Predictor(self._testInferenceGraph())
    with self.assertRaisesRegex(KeyError, 'nonexistent'):
      pred.Run(['fetch1', 'nonexistent'], feed1=[12345])

  def testInvalidFetchWithoutValidateFetchesReturnsNone(self):
    pred = predictor.Predictor(self._testInferenceGraph())
    fetch1, nonexistent = pred.Run(['fetch1', 'nonexistent'],
                                   feed1=[12345],
                                   validate_fetches=False)
    self.assertEqual(12345, fetch1)
    self.assertIsNone(nonexistent)


if __name__ == '__main__':
  tf.test.main()
