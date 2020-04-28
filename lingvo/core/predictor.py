# Lint as: python2, python3
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
"""Predictor does inference using a saved inference graph.

Example::

  params = model_registry.GetParams('MyModel', 'Test')
  inference_graph = inference_graph_exporter.InferenceGraphExporter.Export(
      params)
  pred = Predictor(inference_graph=inference_graph)
  pred.Load("/tmp/logdir/train/ckpt-00000000")
  [topk_hyps] = pred.Run(["topk_hyps"], src_strings=["Hello World"])
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
from lingvo import model_imports
import lingvo.compat as tf
from lingvo.core import inference_graph_pb2
from lingvo.core import py_utils
import six
from six.moves import zip

from google.protobuf import text_format


def LoadInferenceGraph(path, clear_device_placement=False):
  """Parse the given path as an InferenceGraph proto.

  Args:
    path: The path to the file to load.
    clear_device_placement: If true, clears device field from nodes in graph.

  Returns:
    An InferenceGraph object.
  """
  inference_graph = inference_graph_pb2.InferenceGraph()
  with tf.io.gfile.GFile(path, "r") as f:
    text_format.Parse(f.read(), inference_graph)
  if clear_device_placement:
    for node in inference_graph.graph_def.node:
      node.ClearField("device")
    for function in inference_graph.graph_def.library.function:
      for node_def in function.node_def:
        node_def.ClearField("device")
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
    session_config: A tf.SessionConfig to use. By default
      py_utils.SessionConfig() is used.
    clear_device_placement: If set, clears device field of loaded inference
      graph.
  """

  def __init__(self,
               inference_graph,
               subgraph_name=None,
               checkpoint=None,
               device_type="gpu",
               tf_master="",
               session_config=None,
               clear_device_placement=False):
    assert device_type in ["cpu", "gpu", "tpu"]
    subgraph_name = subgraph_name or "default"
    if isinstance(inference_graph, six.string_types):
      tf.logging.info("Reading inference graph from %s.", inference_graph)
      inference_graph = LoadInferenceGraph(inference_graph,
                                           clear_device_placement)
    self._inference_graph = inference_graph
    self._checkpoint = checkpoint
    self._device_type = device_type
    self._tf_master = tf_master
    self._session_config = session_config

    self._graph = tf.Graph()
    with self._graph.as_default():
      tf.logging.info(
          "Loading inference graph for prediction subgraph_name={}.".format(
              subgraph_name))
      self._saver = tf.train.Saver(saver_def=inference_graph.saver_def)
      with tf.device("/%s:0" % "cpu" if device_type == "tpu" else device_type):
        tf.import_graph_def(inference_graph.graph_def, name="")
      if device_type == "tpu":
        # If no tpu init op exists, create it here.
        try:
          self._graph.get_operation_by_name("tpu_init_op")
        except KeyError:
          tf.group(tf.tpu.initialize_system(), name="tpu_init_op")

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
    config = self._session_config
    if not config:
      config = py_utils.SessionConfig()
    sess = tf.Session(self._tf_master, graph=self._graph, config=config)

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
        tf.logging.warning(
            "No checkpoint provided and the graph has no default "
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
        tf.logging.info("Current session id {}.".format(self._cur_sess_id))

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
    if checkpoint != self._checkpoint:
      self._RunWithValidSession(self._saver.restore, checkpoint)
      self._checkpoint = checkpoint

  def Run(self,
          fetch_keys,
          validate_fetches=True,
          session_run_options=None,
          run_metadata=None,
          **kwargs):
    """Runs predictor.

    Args:
      fetch_keys: a list of keys in the fetch dictionary to fetch.
      validate_fetches: if True, raises a KeyError if a specified fetch is
        invalid. If False, returns None for invalid fetches instead.
      session_run_options: Optional tf.RunOptions() to use in the session.
      run_metadata: Optional tf.RunMetadata() to use in the session.
      **kwargs: a dict of inputs to feed.

    Returns:
      A list of predictions corresponding to the order of fetch_keys.

    Raises:
      InvalidArgumentError: the number of inputs does not meet requirements.
      KeyError: a feed specified in kwargs is invalid, or a fetch in fetch_keys
        is invalid and validate_fetches is True.
    """
    if validate_fetches:
      for x in fetch_keys:
        if x not in self._fetches:
          raise KeyError(
              "%s is not in the list of available fetches. Available keys: %s" %
              (x, list(self._fetches.keys())))
    valid_fetch_idxs, valid_fetches = zip(*[(i, self._fetches[k])
                                            for i, k in enumerate(fetch_keys)
                                            if k in self._fetches.keys()])

    for k in kwargs:
      if k not in self._feeds:
        raise KeyError(
            "%s is not in the list of available feeds. Available keys: %s" %
            (k, list(self._feeds.keys())))
    feeds = {self._feeds[k]: v for k, v in six.iteritems(kwargs)}

    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=False)
    if session_run_options:
      run_options = session_run_options

    fetched_results = self._RunWithValidSession(
        tf.Session.run,
        valid_fetches,
        feed_dict=feeds,
        options=run_options,
        run_metadata=run_metadata)
    results = [None] * len(fetch_keys)
    for i, fetch in zip(valid_fetch_idxs, fetched_results):
      results[i] = fetch
    return results


def main(_):
  import IPython  # pylint: disable=g-import-not-at-top
  IPython.start_ipython(argv=["--colors", "NoColor"], user_ns=globals())


if __name__ == "__main__":
  tf.disable_eager_execution()
  model_imports.ImportAllParams()
  tf.app.run(main)
