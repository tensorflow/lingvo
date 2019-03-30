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
"""Utility for exporting an InferenceGraph proto from model params."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import re

import six
import tensorflow as tf

from google.protobuf import text_format
from lingvo.core import base_model
from lingvo.core import bfloat16_variables
from lingvo.core import inference_graph_pb2
from lingvo.core import py_utils

FLAGS = tf.flags.FLAGS

# InferenceDeviceOptions contains options to configure inference on the device.
# device: Device to infer on.
# retain_device_placement: If true, the specified device in the generated
#   inference graph nodes will be retained. Otherwise, the specified device
#   will be cleared, so that the runtime can choose automatically.
# var_options: Options on handling variables. For TPUs, variables can be
#   either placed on device through 'ON_DEVICE' option, or treated as
#   constants with AS_CONSTANTS.
# gen_init_op: Whether to serialize initialization ops for the device. For TPUs,
#   servers can be initialized globally once, in which case this should be
#   turned off to avoid tripping initialization checks.
# dtype_override: Whether to override the dtype to use for activations and
# weights in the model. Options supported are None or tf.bfloat16.
InferenceDeviceOptions = collections.namedtuple('InferenceDeviceOptions', [
    'device', 'retain_device_placement', 'var_options', 'gen_init_op',
    'dtype_override'
])

_CONST_GUARANTEE = None


@contextlib.contextmanager
def NoConstGuaranteeScope():
  """Disallow const gauranteeing variable with-in scope."""
  global _CONST_GUARANTEE
  var_scope = tf.get_variable_scope()
  old_caching_device = var_scope.caching_device
  old_val = _CONST_GUARANTEE
  var_scope.set_caching_device(None)
  _CONST_GUARANTEE = False
  yield
  _CONST_GUARANTEE = old_val
  var_scope.set_caching_device(old_caching_device)


# Marks variable as constants for compilation
def MaybeGuaranteeConstGetter(getter, name, *args, **kwargs):
  global _CONST_GUARANTEE
  if _CONST_GUARANTEE:
    with tf.control_dependencies(None):
      return tf.guarantee_const(
          getter(name, *args, **kwargs), name=name + '/GuaranteeConst')
  else:
    return getter(name, *args, **kwargs)


@contextlib.contextmanager
def ConstGuaranteeScope():
  """Treats all variables under this scope as constants."""
  global _CONST_GUARANTEE
  var_scope = tf.get_variable_scope()
  old_custom_getter = var_scope.custom_getter
  old_caching_device = var_scope.caching_device
  old_val = _CONST_GUARANTEE
  var_scope.set_custom_getter(MaybeGuaranteeConstGetter)
  var_scope.set_caching_device(lambda op: op.device)
  _CONST_GUARANTEE = True
  yield
  _CONST_GUARANTEE = old_val
  var_scope.set_custom_getter(old_custom_getter)
  var_scope.set_caching_device(old_caching_device)


@contextlib.contextmanager
def _DummyScope():
  yield None


def _GetVarName(v):
  return v.name[:-len(':0')]


def _MakeVariableDictionary(variables):
  """Returns a dictionary with name -> tf.Variable() mapping."""
  vars_dict = {}
  for v in variables:
    vars_dict[_GetVarName(v)] = v
  return vars_dict


def IsTpu(device_options):
  return device_options.device == 'tpu'


def ShouldForceBfloat16ForWeightsAndActivations(device_options):
  return device_options.dtype_override == tf.bfloat16


def ConvertSubgraphDictToProto(subgraphs_dict):
  """Converts dict of subgraphs/feeds/fetches to InferenceGraph.

  Args:
    subgraphs_dict: Dict of (fetches, feeds) where each fetches/feeds is a
      NestedMap.

  Returns:
    Equivalent InferenceGraph.
  """
  # Build the output inference graph.
  inference_graph_proto = inference_graph_pb2.InferenceGraph()
  for subgraph_name, tensors in six.iteritems(subgraphs_dict):
    fetches = tensors[0]
    feeds = tensors[1]

    # Rewrite fetches and feeds to map to their tensor name instead of
    # Tensor instance.
    named_fetches = {
        k: v.name for k, v in six.iteritems(fetches) if v is not None
    }
    named_feeds = {k: v.name for k, v in six.iteritems(feeds)}

    # Export as subgraph.
    inference_graph_proto.subgraphs[subgraph_name].fetches.update(named_fetches)
    inference_graph_proto.subgraphs[subgraph_name].feeds.update(named_feeds)
  return inference_graph_proto


def GetOutputOpNames(graph,
                     inference_graph_proto,
                     preserve_colocation_nodes=True):
  """Gets output op names from an inference graph.

  Args:
    graph: The tf graph.
    inference_graph_proto: an InferenceGraph proto.
    preserve_colocation_nodes: a Python bool, default to True. Preserves nodes
      colocating with the closure of output ops in the returned array.

  Returns:
    Array of tf op names that should be preserved in the graph.
  """
  output_op_names = set()
  for subgraph in six.itervalues(inference_graph_proto.subgraphs):
    # Sometimes feeds aren't connected to any outputs but keep them in the graph
    # anyways to avoid errors.
    for tensor_or_op_name in (
        list(subgraph.feeds.values()) + list(subgraph.fetches.values())):
      # Tensor names have format <op_name>:<output_index>. Some inference
      # graphs put tensors and others put ops in the feeds/fetches (depends
      # on how it is used). We differentiate here. We still do the lookup in
      # the graph to sanity check (versus relying on the text manipulation).
      # If this logic ever breaks, TensorFlow will raise a ValueError with
      # a description of the syntax of each.
      if re.search(r':[0-9]+$', tensor_or_op_name):
        # Tensor-name.
        t = graph.get_tensor_by_name(tensor_or_op_name)
        output_op_names.add(t.op.name)
      else:
        op = graph.get_operation_by_name(tensor_or_op_name)
        output_op_names.add(op.name)

  # We also need to preserve any nodes that are used for colocation.
  # E.g., a node may have this attr:
  #   attr {
  #     key: "_class"
  #     value {
  #       list {
  #         s: "loc:@inference/embedding_lookup/Read/ReadVariableOp"
  #       }
  #     }
  #   }
  #
  # In this case, we need to make sure the node
  # inference/embedding_lookup/Read/ReadVariableOp is not pruned.
  #
  # TODO(zhifengc): It's possible that it's better to fix in
  # tf.graph_util.extract_sub_graph.
  graph_def = tf.graph_util.extract_sub_graph(graph.as_graph_def(),
                                              list(output_op_names))
  reachable_vars = [node.name for node in graph_def.node]

  if not preserve_colocation_nodes:
    return sorted(list(output_op_names))

  for node in graph.get_operations():
    if '_class' in node.node_def.attr:
      for loc in node.node_def.attr['_class'].list.s:
        loc = loc.decode('utf-8')
        if loc.startswith('loc:@'):
          loc_name = loc[5:]
          if loc_name not in reachable_vars:
            # Skip nodes that cannot be reached from the pruned graph.
            continue
          output_op_names.add(node.name)

  return sorted(list(output_op_names))


def _ParamExists(param_obj, param_name):
  """Tests whether param_name is contained in param_obj."""
  if not param_obj:
    return
  for k, _ in param_obj.IterParams():
    if k == param_name:
      return True
  return False


def _FreezeGraphFromCheckpoint(graph, saver, checkpoint, output_op_names):
  """Freezes a graph from a checkpoint.

  Args:
    graph: tf.Graph.
    saver: The tf.Saver to use for restoration.
    checkpoint: The checkpoint to restore.
    output_op_names: Names of output ops.

  Returns:
    Resulting tf.GraphDef.
  """
  sess = tf.Session(graph=graph, config=py_utils.SessionConfig())
  saver.restore(sess, checkpoint)
  return tf.graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), output_op_names)


def _FreezeDefaults(graph, output_op_names):
  """Default initializes a graph and freezes it.

  Args:
    graph: tf.Graph.
    output_op_names: Names of output ops.

  Returns:
    Resulting tf.GraphDef.
  """
  with tf.Session(graph=graph, config=py_utils.SessionConfig()) as sess:
    sess.run(graph.get_operation_by_name('init_all_variables'))
    return tf.graph_util.convert_variables_to_constants(sess,
                                                        graph.as_graph_def(),
                                                        output_op_names)


class InferenceGraphExporter(object):
  """Class for exporting inference graphs."""

  @classmethod
  def Export(cls,
             model_cfg,
             model_task_name=None,
             device_options=InferenceDeviceOptions(
                 device='',
                 retain_device_placement=False,
                 var_options=None,
                 gen_init_op=True,
                 dtype_override=None),
             freeze_checkpoint=None,
             freeze_defaults=False,
             export_path=None,
             subgraph_filter=None,
             random_seed=None):
    """Exports a InferenceGraph proto with piecewise subgraphs.

    Sets FLAGS.enable_asserts to False unless user explicitly sets it to True.

    Args:
      model_cfg: a Params instance as returned by
        model_registry.GetParams(modelname, 'Test') or model_params.Model().
      model_task_name: The task to generate an inference graph for. Should be
        None for single-task models.
      device_options: Device options for the accelerator used for serving.
      freeze_checkpoint: The checkpoint to load. Loads and freezes the model if
        given.
      freeze_defaults: Default initializes the graph and freeze. Useful for
        early testing of downstream tools without having a checkpoint.
      export_path: If not None, write the inference graph in ASCII to this path.
      subgraph_filter: If not None or empty, export only this list of inference
        subgraphs.
      random_seed: Fixes the random seed in the exported inference graph.

    Returns:
      InferenceGraph proto.

    Raises:
      ValueError: if the model does not support the listed subgraphs.
    """
    assert issubclass(model_cfg.cls, base_model.BaseModel)

    # Disable assertions unless user explicitly enables it.
    if FLAGS['enable_asserts'].using_default_value:
      FLAGS.enable_asserts = False

    # TODO(laurenzo): Work out how much we need to specify here in terms of
    # cluster configuration.
    cls._SetClusterParams(model_cfg.cluster, device_options)

    # Disable packed inputs for inference writing purposes.
    def _DisablePackedInput(task):
      if (_ParamExists(task, 'encoder') and
          _ParamExists(task.encoder, 'packed_input')):
        task.encoder.packed_input = False
      if (_ParamExists(task, 'decoder') and
          _ParamExists(task.decoder, 'packed_input')):
        task.decoder.packed_input = False

    # Configure the model.
    model_cfg.random_seed = random_seed
    model_cfg.is_eval = True
    model_cfg.is_inference = True

    if issubclass(model_cfg.cls, base_model.MultiTaskModel):
      for _, task_param in model_cfg.task_params.IterParams():
        _DisablePackedInput(task_param)
    else:
      _DisablePackedInput(model_cfg.task)

    tf.logging.info('Model %s. Params: %s', model_cfg.name, model_cfg.ToText())

    # Instantiate the graph.
    graph = tf.Graph()
    with graph.as_default():
      tf.set_random_seed(random_seed)
      cluster = model_cfg.cluster.cls(model_cfg.cluster)
      device = cluster.GetPlacer()
      tpu_const_scope = _DummyScope()
      if (IsTpu(device_options) and
          device_options.var_options == 'AS_CONSTANTS'):
        # Do not specify devices for variables if we are marking them as
        # constants.
        device = ''
        tpu_const_scope = ConstGuaranteeScope()

      with cluster, tf.device(device), tpu_const_scope:

        bfloat16_override = ShouldForceBfloat16ForWeightsAndActivations(
            device_options)

        if bfloat16_override:
          py_utils.UpdateDtype(model_cfg, tf.bfloat16)
          py_utils.UpdateFpropDtype(model_cfg, tf.bfloat16)

        # Hard-code TPU-related flags prior to instantiating model.
        old_enable_asserts = FLAGS.enable_asserts
        old_xla_device = FLAGS.xla_device
        if IsTpu(device_options):
          FLAGS.enable_asserts = False
          FLAGS.xla_device = 'tpu'

        try:
          mdl = model_cfg.cls(model_cfg)
          variables_to_restore = (
              _MakeVariableDictionary(tf.global_variables())
              if not mdl.ema else mdl.ema.variables_to_restore())

          if bfloat16_override:
            saver_var_spec = (
                bfloat16_variables
                .get_saver_spec_for_variables_with_bf16_overrides(
                    variables_to_restore))
          else:
            saver_var_spec = variables_to_restore

          saver = tf.train.Saver(saver_var_spec)
          tf.variables_initializer(
              tf.global_variables(), name='init_all_variables')
          if IsTpu(device_options) and device_options.gen_init_op:
            tf.group(tf.contrib.tpu.initialize_system(), name='tpu_init_op')

          model_task = mdl.GetTask(model_task_name)

          inference_graph_proto = inference_graph_pb2.InferenceGraph()
          subgraphs_proto = model_task.Inference()
          if isinstance(subgraphs_proto, dict):
            subgraphs_proto = ConvertSubgraphDictToProto(subgraphs_proto)
          for name, subgraph in subgraphs_proto.subgraphs.items():
            if not subgraph_filter or name in subgraph_filter:
              inference_graph_proto.subgraphs[name].CopyFrom(subgraph)

          # Add a table init op and global variable init op to the graph.
          # Tables can be declared anywhere in the graph, so this op has to be
          # added last.
          tf.tables_initializer(name='init_all_tables')
        finally:
          # Reset TPU-related flags after model instantiation.
          FLAGS.enable_asserts = old_enable_asserts
          FLAGS.xla_device = old_xla_device

    tf.logging.info('Graph contains ops: %r',
                    [op.name for op in graph.get_operations()])

    inference_graph_proto.saver_def.CopyFrom(saver.as_saver_def())

    # Freezing.
    if freeze_defaults or freeze_checkpoint:
      output_op_names = GetOutputOpNames(
          graph, inference_graph_proto, preserve_colocation_nodes=False)
      if cls._DeviceSupportsFreezing(device_options):
        raise ValueError('freeze_checkpoint cannot be used with device ' +
                         device_options.device)
      if freeze_checkpoint:
        tf.logging.info('Freezing graph from checkpoint: %s', freeze_checkpoint)
        graph_def = _FreezeGraphFromCheckpoint(graph, saver, freeze_checkpoint,
                                               output_op_names)
      elif freeze_defaults:
        tf.logging.info('Default initializing graph and freezing.')
        graph_def = _FreezeDefaults(graph, output_op_names)
    else:
      output_op_names = GetOutputOpNames(graph, inference_graph_proto)

      # Prune the graph to just the parts we need.
      # To support restoring, we have to not prune out the restore node.
      output_op_names.append('init_all_tables')
      output_op_names.append('init_all_variables')
      output_op_names.append('save/control_dependency')
      output_op_names.append('save/restore_all')
      if IsTpu(device_options) and device_options.gen_init_op:
        output_op_names.append('tpu_init_op')
      graph_def = graph.as_graph_def()
      tf.logging.info('Pruning graph to output ops: %r', output_op_names)
      graph_def = tf.graph_util.extract_sub_graph(graph_def, output_op_names)

    if not device_options.retain_device_placement:
      # Clear the device so that the runtime can choose.
      tf.logging.info('Clearing device placement for: %s',
                      device_options.device)
      for node in graph_def.node:
        node.ClearField('device')
      for function in graph_def.library.function:
        for node_def in function.node_def:
          node_def.ClearField('device')

    inference_graph_proto.graph_def.CopyFrom(graph_def)

    if export_path:
      with tf.gfile.Open(export_path, 'w') as f:
        f.write(text_format.MessageToString(inference_graph_proto))
    return inference_graph_proto

  @classmethod
  def _SetClusterParams(cls, cluster_params, device_options):
    """Sets cluster params.

    Args:
      cluster_params: Model().cluster config.
      device_options: InferenceDeviceOptions.
    """

    def Update(p):
      """Update cluster params `p`."""
      p.name = '/job:localhost'
      p.replicas = 1
      p.tpus_per_replica = 1 if IsTpu(device_options) else 0
      p.gpus_per_replica = 0
      p.devices_per_split = 1

    cluster_params.mode = 'sync'
    cluster_params.job = 'decoder'
    cluster_params.add_summary = False
    Update(cluster_params.controller)
    Update(cluster_params.worker)
    Update(cluster_params.ps)
    Update(cluster_params.evaler)
    Update(cluster_params.decoder)
    Update(cluster_params.input)

  @classmethod
  def _DeviceSupportsFreezing(cls, device_options):
    return IsTpu(device_options)
