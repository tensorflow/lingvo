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
# pylint: disable=line-too-long
"""Predictor does inference using a saved inference graph.

Example::

  params = model_registry.GetParams('MyModel', 'Test')
  inference_graph = inference_graph_exporter.InferenceGraphExporter.Export(
      params)
  pred = Predictor(inference_graph=inference_graph)
  pred.Load("/tmp/logdir/train/ckpt-00000000")
  [topk_hyps] = pred.Run(["topk_hyps"], src_strings=["Hello World"])
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import six
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.core.protobuf import config_pb2
from lingvo.core import inference_graph_pb2
from lingvo.core import py_utils


def LoadInferenceGraph(path):
  """Parse the given path as an InferenceGraph proto.

  Args:
    path: The path to the file to load.

  Returns:
    An InferenceGraph object.
  """
  inference_graph = inference_graph_pb2.InferenceGraph()
  with tf.gfile.Open(path, "r") as f:
    text_format.Parse(f.read(), inference_graph)
  return inference_graph


class Predictor(object):
  """Loads a model and does inference.

  See model.Inference() documentation for list of fetches and feeds.

  Args:
    inference_graph: A saved InferenceGraph proto.
    subgraph_name: The subgraph to use for prediction.
    checkpoint: An optional checkpoint to load.
    device_type: Device type string. Either "cpu", "gpu", or "tpu".
    tf_master: The tf_master.
  """

  def __init__(self,
               inference_graph,
               subgraph_name=None,
               checkpoint=None,
               device_type="gpu",
               tf_master=""):
    assert device_type in ["cpu", "gpu", "tpu"]
    subgraph_name = subgraph_name or "default"
    if isinstance(inference_graph, six.string_types):
      tf.logging.info("Reading inference graph from %s.", inference_graph)
      inference_graph = LoadInferenceGraph(inference_graph)
    self._inference_graph = inference_graph
    self._checkpoint = checkpoint
    self._device_type = device_type
    self._tf_master = tf_master

    self._graph = tf.Graph()
    with self._graph.as_default():
      tf.logging.info("Loading inference graph for prediction.")
      self._saver = tf.train.Saver(saver_def=inference_graph.saver_def)
      with tf.device("/%s:0" % "cpu" if device_type == "tpu" else device_type):
        tf.import_graph_def(inference_graph.graph_def, name="")
      self._graph.finalize()

    if inference_graph.subgraphs:
      if subgraph_name not in inference_graph.subgraphs:
        raise ValueError(
            "Subgraph %s not defined. Valid subgraphs: %s" %
            (subgraph_name, list(inference_graph.subgraphs.keys())))
      subgraph = inference_graph.subgraphs[subgraph_name]
      self._fetches = subgraph.fetches
      self._feeds = subgraph.feeds
    else:
      self._fetches = inference_graph.fetches
      self._feeds = inference_graph.feeds

    # Lock for creating new sessions.
    self._sess_lock = threading.Lock()
    self._cur_sess_id = 0
    self._CreateNewSession()

  @property
  def fetch_keys(self):
    return list(self._fetches.keys())

  @property
  def feed_keys(self):
    return list(self._feeds.keys())

  @py_utils.RetryOnTransientTfError()
  def _CreateNewSession(self):
    """Updates self._sess with a new session."""
    sess = tf.Session(
        self._tf_master, graph=self._graph, config=py_utils.SessionConfig())
    try:
      sess.run(self._graph.get_operation_by_name("init_all_tables"))
    except KeyError:
      tf.logging.info("Could not find tables initializer in graph.")
    if self._device_type == "tpu":
      sess.run(self._graph.get_operation_by_name("tpu_init_op"))
    if self._checkpoint:
      self._saver.restore(sess, self._checkpoint)
    else:
      try:
        init_op = self._graph.get_operation_by_name("init_all_variables")
        sess.run(init_op)
      except KeyError:
        tf.logging.warn("No checkpoint provided and the graph has no default "
                        "variable_init op.")
    tf.logging.info("Created new predictor session.")
    self._sess = sess

  def _MaybeCreateNewSession(self, sess_id):
    """Create a new session if sess_id is the current session.

    Args:
      sess_id: The id of a session that no longer works.
    """
    with self._sess_lock:
      if sess_id == self._cur_sess_id:
        self._CreateNewSession()
        self._cur_sess_id += 1

  @py_utils.RetryOnTransientTfError()
  def _RunWithValidSession(self, fn, *args, **kwargs):
    """Ensures `fn` is called while self._sess is a valid session."""
    sess_id = self._cur_sess_id
    try:
      return fn(self._sess, *args, **kwargs)
    except py_utils.transient_tf_errors:
      # self._sess is invalid, most likely due to the worker being preempted.
      # Make sure a new session is created before re-raising the exception and
      # triggering the py_utils.Retry loop.
      self._MaybeCreateNewSession(sess_id)
      raise

  def Load(self, checkpoint):
    """Loads parameters from a checkpoint.

    Args:
      checkpoint: The checkpoint path to restore.
    """
    self._RunWithValidSession(self._saver.restore, checkpoint)
    self._checkpoint = checkpoint

  def Run(self, fetch_keys, **kwargs):
    """Runs predictor.

    Args:
      fetch_keys: a list of keys in the fetch dictionary to fetch.
      **kwargs: a dict of inputs to feed.

    Returns:
      A list of predictions corresponding to the order of fetch_keys.

    Raises:
      ValueError: the number of inputs does not meet requirements.
      KeyError: a key specified in fetch_keys is invalid.
    """
    for x in fetch_keys:
      if x not in self._fetches:
        raise KeyError(
            "Key %s is not in the list of available fetches. Available keys: %s"
            % (x, list(self._fetches.keys())))
    fetches = [self._fetches[x] for x in fetch_keys]

    for k in kwargs:
      if k not in self._feeds:
        raise KeyError(
            """kwarg %s is not in the list of available feeds. Available kwargs:
             %s""" % (k, list(self._feeds.keys())))
    feeds = {self._feeds[k]: v for k, v in six.iteritems(kwargs)}

    run_options = config_pb2.RunOptions(
        report_tensor_allocations_upon_oom=False)
    return self._RunWithValidSession(
        tf.Session.run, fetches, feed_dict=feeds, options=run_options)


def main(_):
  # pylint: disable=g-import-not-at-top
  # pylint: disable=unused-variable
  import lingvo.model_imports
  import IPython
  IPython.start_ipython(argv=["--colors", "NoColor"], user_ns=globals())


if __name__ == "__main__":
  tf.app.run(main)
