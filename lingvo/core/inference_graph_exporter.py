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

import collections
import contextlib
import re
from typing import Optional

import lingvo.compat as tf
from lingvo.core import base_model
from lingvo.core import bfloat16_variables
from lingvo.core import inference_graph_pb2
from lingvo.core import py_utils
from lingvo.core import tpu_embedding_layers_v1
import six

from google.protobuf import text_format

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
    'dtype_override', 'fprop_dtype_override'
])


_CONST_GUARANTEE = None


# Marks variable as constants for compilation
def GuaranteeConstGetter(next_creator, **kwargs):
  if _CONST_GUARANTEE:
    with tf.control_dependencies(None):
      name = kwargs['var_name'] + '/GuaranteeConst'
      return tf.guarantee_const(next_creator(**kwargs), name=name)
  return next_creator(**kwargs)


@contextlib.contextmanager
def ConstGuaranteeScope():
  """Treats all variables under this scope as constants."""
  global _CONST_GUARANTEE
  old_val = _CONST_GUARANTEE
  _CONST_GUARANTEE = True
  with py_utils.VariableCreatorScope(GuaranteeConstGetter):
    yield
  _CONST_GUARANTEE = old_val


@contextlib.contextmanager
def NoConstGuaranteeScope():
  """Disallow const guaranteeing variable with-in scope."""
  global _CONST_GUARANTEE
  old_val = _CONST_GUARANTEE
  _CONST_GUARANTEE = False
  yield
  _CONST_GUARANTEE = old_val


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


def IsGpu(device_options):
  return device_options.device == 'gpu'


def ShouldForceBfloat16ForWeightsAndActivations(device_options):
  return device_options.dtype_override == tf.bfloat16


def ShouldForceBfloat16ForActivations(device_options):
  return device_options.fprop_dtype_override == tf.bfloat16


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
  for subgraph_name, tensors in subgraphs_dict.items():
    fetches = tensors[0]
    feeds = tensors[1]

    # Rewrite fetches and feeds to map to their tensor name instead of
    # Tensor instance.
    named_fetches = {k: v.name for k, v in fetches.items() if v is not None}
    named_feeds = {k: v.name for k, v in feeds.items() if v is not None}

    # Export as subgraph.
    inference_graph_proto.subgraphs[subgraph_name].fetches.update(named_fetches)
    inference_graph_proto.subgraphs[subgraph_name].feeds.update(named_feeds)
  return inference_graph_proto


def GetOutputOpNames(
    graph,
    inference_graph_proto,
    subgraphs=None,
    preserve_colocation_nodes=True,
    preserve_saver_restore_nodes=False,
    preserve_extra_ops=None,
    return_name_with_op=False,
):
  """Gets output op names and/or ops from an inference graph.

  Args:
    graph: The tf graph.
    inference_graph_proto: an InferenceGraph proto.
    subgraphs: an optional list of subgraph names. If provided, only output ops
      from these subgraphs are preserved. Otherwise, all subgraphs are included.
    preserve_colocation_nodes: a Python bool, default to True. Preserves nodes
      colocating with the closure of output ops in the returned array.
    preserve_saver_restore_nodes: a Python bool, default to False. Preserves
      nodes for restoring according to inference_graph_proto.saver_def.
    preserve_extra_ops: an optional list of extra op names to preserve as long
      as they present in the graph.
    return_name_with_op: If true, return list of (op name, op). If false, return
      list of op names.

  Returns:
    List of (op name, op) or (op name) that should be preserved in the graph.
  """
  name_op_tuples = set()

  def _GetOpName(tensor_or_op_name):
    """Returns (op name, op) of the given node name."""
    # Tensor names have format <op_name>:<output_index>. Some inference
    # graphs put tensors and others put ops in the feeds/fetches (depends
    # on how it is used). We differentiate here. We still do the lookup in
    # the graph to sanity check (versus relying on the text manipulation).
    # If this logic ever breaks, TensorFlow will raise a ValueError with
    # a description of the syntax of each.
    if re.search(r':[0-9]+$', tensor_or_op_name):
      # Tensor-name.
      t = graph.get_tensor_by_name(tensor_or_op_name)
      return (t.op.name, t.op)
    else:
      op = graph.get_operation_by_name(tensor_or_op_name)
      return (op.name, op)

  def _PostProcess(name_op_tuples):
    if return_name_with_op:
      return sorted(list(name_op_tuples), key=lambda tup: tup[0])
    else:
      return sorted([name for name, _ in name_op_tuples])

  for subgraph_name, subgraph in inference_graph_proto.subgraphs.items():
    if subgraphs and subgraph_name not in subgraphs:
      tf.logging.info('Skip subgraph %s.', subgraph_name)
      continue
    # Sometimes feeds aren't connected to any outputs but keep them in the graph
    # anyways to avoid errors.
    for tensor_or_op_name in (list(subgraph.feeds.values()) +
                              list(subgraph.fetches.values())):
      name_op_tuples.add(_GetOpName(tensor_or_op_name))

  if preserve_saver_restore_nodes:
    # Only nodes for restoring is preserved. saver_def.save_tensor_name is
    # skipped because it's only used for saving.
    saver_def = inference_graph_proto.saver_def
    for op_name in [saver_def.filename_tensor_name, saver_def.restore_op_name]:
      try:
        name_op_tuples.add(_GetOpName(op_name))
      except KeyError:
        tf.logging.info('Op/tensor %s not in the graph. Ignoring.' % op_name)

  if not preserve_colocation_nodes and not preserve_extra_ops:
    return _PostProcess(name_op_tuples)

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
  # tf.compat.v1.graph_util.extract_sub_graph.
  graph_def = tf.compat.v1.graph_util.extract_sub_graph(
      graph.as_graph_def(), [name for (name, _) in name_op_tuples]
  )
  reachable_vars = [node.name for node in graph_def.node]

  for node in graph.get_operations():
    if preserve_extra_ops and node.name in preserve_extra_ops:
      name_op_tuples.add((node.name, node))
    elif preserve_colocation_nodes and '_class' in node.node_def.attr:
      for loc in node.node_def.attr['_class'].list.s:
        loc = six.ensure_text(loc, 'utf-8')
        if loc.startswith('loc:@'):
          loc_name = loc[5:]
          if loc_name not in reachable_vars:
            # Skip nodes that cannot be reached from the pruned graph.
            continue
          name_op_tuples.add((node.name, node))

  return _PostProcess(name_op_tuples)


def _GetEmaVarToNameDict(mdl: base_model.BaseModel) -> dict[tf.Variable, str]:
  """Return EMA variables to names dictionary."""
  ema_vars_name_dict = {}
  if mdl.ema:
    # tf.Variable can't be key of dict. Need to use ref().
    ema_vars_name_dict = {
        v.ref(): k
        for k, v in mdl.ema.variables_to_restore(mdl.variables_for_ema).items()
    }
  return ema_vars_name_dict


def _GetReachableVarsToRestore(
    ema_vars_name_dict: dict[tf.Variable, str],
    name_op_tuples: list[tuple[str, tf.Operation]],
    graph: tf.Graph,
) -> dict[str, tf.Variable]:
  """Get reachable variables to restore dictionary."""
  with graph.as_default():
    vars_to_restore_set = set()
    for name, op in name_op_tuples:
      if op.type == 'VarHandleOp':
        vars_to_restore_set.add(name)
    variables_to_restore = {}
    # Can't get tf.Variables that are not created by tf.get_variable().
    for var in tf.global_variables():
      if var.op.name in vars_to_restore_set:
        if var.ref() in ema_vars_name_dict:
          variables_to_restore[ema_vars_name_dict[var.ref()]] = var
        else:
          variables_to_restore[var.op.name] = var
    return variables_to_restore


def _MaybeCreateSaver(
    variables_to_restore: dict[str, tf.Variable],
    bfloat16_override: bool,
    bfloat16_ckpt: bool,
) -> Optional[tf.train.Saver]:
  """Create tf.train.Saver if there's variable to restore.

  Args:
    variables_to_restore: A dictionary specifying variables to be restored. The
      keys are variables names and the values are variables.
    bfloat16_override: Whether to overrides float32 variables' dtype to
      bfloat16.
    bfloat16_ckpt: Whether the floating-type variables in the checkpoint are
      bfloat16.

  Returns:
    A saver that will save and restore the variables specified in
    `variables_to_restore`.
  """
  if not variables_to_restore:
    return None
  if bfloat16_override:
    saver_var_spec = (
        bfloat16_variables.get_saver_spec_for_variables_with_bf16_overrides(
            variables_to_restore, bfloat16_ckpt
        )
    )
    # For TPU embedding layers, if the table explicitly specifies the
    # inference dtype as bfloat16, the variables in the checkpoint
    # must already be in bfloat16, so we change back to bfloat16 to
    # avoid dtype mismatch.
    tpu_emb_coll = tpu_embedding_layers_v1.TpuEmbeddingCollection.Get()
    for var_name in tpu_emb_coll.inference_with_bfloat16_var_names:
      saver_var_spec[var_name] = variables_to_restore[var_name]
  else:
    saver_var_spec = variables_to_restore
  tf.logging.info('Saving variables %r', saver_var_spec)
  return tf.train.Saver(saver_var_spec)


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
  return tf.compat.v1.graph_util.convert_variables_to_constants(
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
    return tf.compat.v1.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), output_op_names)


class InferenceGraphExporter:
  """Class for exporting inference graphs."""

  @classmethod
  def Export(
      cls,
      model_cfg,
      model_task_name=None,
      inference_fn_name='Inference',
      inference_fn_kwargs=None,
      device_options=InferenceDeviceOptions(
          device='',
          retain_device_placement=False,
          var_options=None,
          gen_init_op=True,
          dtype_override=None,
          fprop_dtype_override=None,
      ),
      freeze_checkpoint=None,
      freeze_defaults=False,
      export_path=None,
      subgraph_filter=None,
      random_seed=None,
      disable_packed_input=True,
      prune_graph=True,
      export_graph_collections=False,
      bfloat16_ckpt=False,
  ):
    """Exports a InferenceGraph proto with piecewise subgraphs.

    Sets FLAGS.enable_asserts to False unless user explicitly sets it to True.

    Note: Enable FLAGS.pin_vars_to_cpu (default false) to make weight-sharing
    and multi-core inference on TPUs work properly.

    Args:
      model_cfg: a Params instance as returned by
        model_registry.GetParams(modelname, 'Test') or model_params.Model().
      model_task_name: The task to generate an inference graph for. Should be
        None for single-task models.
      inference_fn_name: Task's inference function name, which generates
        subgraphs.
      inference_fn_kwargs: kwargs for inference_fn_name.
      device_options: Device options for the accelerator used for serving.
      freeze_checkpoint: The checkpoint to load. Loads and freezes the model if
        given.
      freeze_defaults: Default initializes the graph and freeze. Useful for
        early testing of downstream tools without having a checkpoint.
      export_path: If not None, write the inference graph in ASCII to this path.
      subgraph_filter: A string or a list of subgraph names. If not None or
        empty, export only this list of inference subgraphs.
      random_seed: Fixes the random seed in the exported inference graph.
      disable_packed_input: Disable packed input for inference writing purposes.
      prune_graph: If true, prune the graph to just the parts we need.
      export_graph_collections: If true, export graph collections to the
        InferenceGraph proto.
      bfloat16_ckpt: Whether the checkpoint is of type bfloat16.

    Returns:
      InferenceGraph proto.

    Raises:
      ValueError: if the model does not support the listed subgraphs.
    """
    assert issubclass(model_cfg.cls, base_model.BaseModel)
    if device_options.dtype_override and device_options.fprop_dtype_override:
      raise ValueError(
          'device_options{dtype_override,fprop_dtype_override) can not both be'
          'set.')
    if subgraph_filter and not isinstance(subgraph_filter, (tuple, list)):
      subgraph_filter = [subgraph_filter]

    # Disable assertions unless user explicitly enables it.
    if FLAGS['enable_asserts'].using_default_value:
      FLAGS.enable_asserts = False

    # TODO(laurenzo): Work out how much we need to specify here in terms of
    # cluster configuration.
    cls._SetClusterParams(model_cfg.cluster, device_options)

    # Configure the model.
    model_cfg.random_seed = random_seed
    model_cfg.is_inference = True

    if disable_packed_input:

      def _DisablePackedInput(task):
        if (_ParamExists(task, 'encoder') and
            _ParamExists(task.encoder, 'packed_input')):
          task.encoder.packed_input = False
        if (_ParamExists(task, 'decoder') and
            _ParamExists(task.decoder, 'packed_input')):
          task.decoder.packed_input = False

      if issubclass(model_cfg.cls, base_model.MultiTaskModel):
        for _, task_param in model_cfg.task_params.IterParams():
          _DisablePackedInput(task_param)
      else:
        _DisablePackedInput(model_cfg.task)

    tf.logging.debug('Model %s params:', model_cfg.name)
    for line in model_cfg.ToText().split('\n'):
      tf.logging.debug('%s', line)

    # Instantiate the graph.
    graph = tf.Graph()
    with graph.as_default():
      tf.random.set_seed(random_seed)
      cluster = model_cfg.cluster.Instantiate()
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

        act_bfloat16_override = ShouldForceBfloat16ForActivations(
            device_options)
        if act_bfloat16_override:
          py_utils.UpdateFpropDtype(model_cfg, tf.bfloat16)

        # Hard-code TPU-related flags prior to instantiating model.
        old_enable_asserts = FLAGS.enable_asserts
        old_xla_device = FLAGS.xla_device
        if IsTpu(device_options) or IsGpu(device_options):
          FLAGS.enable_asserts = False
          FLAGS.xla_device = device_options.device

        try:
          mdl = model_cfg.Instantiate()
          task = mdl.GetTask(model_task_name)

          tf.variables_initializer(
              tf.global_variables(), name='init_all_variables')
          if IsTpu(device_options) and device_options.gen_init_op:
            tf.group(tf.tpu.initialize_system(), name='tpu_init_op')

          if freeze_checkpoint or freeze_defaults:
            # Replace variables with tensors using tf.identity in theta before
            # freezing to avoid the graph referencing types of DT_RESOURCE.
            def AddIdentityToTheta(layer):
              # pylint: disable=protected-access
              layer._private_theta = py_utils.Transform(tf.identity,
                                                        layer._private_theta)
              # pylint: enable=protected-access
              layer.children.Transform(AddIdentityToTheta)

            AddIdentityToTheta(task)

          inference_graph_proto = inference_graph_pb2.InferenceGraph()
          if inference_fn_kwargs is not None:
            subgraphs_proto = getattr(task,
                                      inference_fn_name)(**inference_fn_kwargs)
          else:
            subgraphs_proto = getattr(task, inference_fn_name)()

          if isinstance(subgraphs_proto, dict):
            subgraphs_proto = ConvertSubgraphDictToProto(subgraphs_proto)
          for name, subgraph in subgraphs_proto.subgraphs.items():
            if not subgraph_filter or name in subgraph_filter:
              inference_graph_proto.subgraphs[name].CopyFrom(subgraph)

          if not inference_graph_proto.subgraphs and subgraph_filter:
            raise ValueError(
                f'Subgraph filters {subgraph_filter} filtered out all '
                'subgraphs. Defined subgraphs: '
                f'{list(subgraphs_proto.subgraphs.keys())}')

          ema_vars_name_dict = _GetEmaVarToNameDict(mdl)
          name_op_tuples = GetOutputOpNames(
              graph, inference_graph_proto, return_name_with_op=True
          )
          variables_to_restore = _GetReachableVarsToRestore(
              ema_vars_name_dict, name_op_tuples, graph
          )
          saver = _MaybeCreateSaver(
              variables_to_restore, bfloat16_override, bfloat16_ckpt
          )

          assets_collection = py_utils.GetSavedModelAssets()
          for asset in assets_collection:
            if asset.op.type == 'Const' and asset.op.get_attr(
                'dtype') == tf.dtypes.string:
              constant_value = asset.op.get_attr('value')
              if constant_value.string_val:
                tf.logging.info('Found asset file_path: %s',
                                constant_value.string_val[0])
                asset_file_def = inference_graph_proto.asset_file_def.add()
                asset_file_def.tensor_info.name = asset.name
                asset_file_def.filename = constant_value.string_val[0]

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

    # Collection defs
    if not tf.executing_eagerly():
      if export_graph_collections:
        meta_graph = tf.train.export_meta_graph(graph=graph)
        for key in meta_graph.collection_def:
          tf.logging.info('copying collection %s', key)
          inference_graph_proto.collection_def[key].CopyFrom(
              meta_graph.collection_def[key])
    else:
      tf.logging.warning('Not exporting collection defs '
                         'since operating in eager mode.')

    # Freezing.
    if freeze_defaults or freeze_checkpoint:
      if saver is None:
        tf.logging.info('No variables to restore.')
        graph_def = graph.as_graph_def()
      else:
        output_op_names = GetOutputOpNames(
            graph,
            inference_graph_proto,
            preserve_colocation_nodes=False,
            preserve_saver_restore_nodes=False,
        )
        if cls._DeviceSupportsFreezing(device_options):
          raise ValueError(
              'freeze_checkpoint cannot be used with device '
              + device_options.device
          )
        if freeze_checkpoint:
          tf.logging.info(
              'Freezing graph from checkpoint: %s', freeze_checkpoint
          )
          graph_def = _FreezeGraphFromCheckpoint(
              graph, saver, freeze_checkpoint, output_op_names
          )
        elif freeze_defaults:
          tf.logging.info('Default initializing graph and freezing.')
          graph_def = _FreezeDefaults(graph, output_op_names)
    else:
      if saver is not None:
        inference_graph_proto.saver_def.CopyFrom(saver.as_saver_def())
      graph_def = graph.as_graph_def()

      if prune_graph:
        output_op_names = [name for name, _ in name_op_tuples]

        # Prune the graph to just the parts we need.
        # To support restoring, we have to not prune out the restore node.
        output_op_names.append('init_all_tables')
        output_op_names.append('init_all_variables')
        if saver is not None:
          output_op_names.append('save/control_dependency')
          output_op_names.append('save/restore_all')
        if IsTpu(device_options) and device_options.gen_init_op:
          output_op_names.append('tpu_init_op')

        tf.logging.info('Pruning graph to output ops: %r', output_op_names)
        graph_def = tf.compat.v1.graph_util.extract_sub_graph(
            graph_def, output_op_names)

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
      with tf.io.gfile.GFile(export_path, 'w') as f:
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
      p.gpus_per_replica = 1 if IsGpu(device_options) else 0
      p.devices_per_split = 1

    cluster_params.mode = 'sync'
    cluster_params.job = 'decoder'
    cluster_params.add_summary = False
    cluster_params.do_eval = True
    Update(cluster_params.controller)
    Update(cluster_params.worker)
    Update(cluster_params.ps)
    Update(cluster_params.evaler)
    Update(cluster_params.decoder)
    Update(cluster_params.input)

  @classmethod
  def _DeviceSupportsFreezing(cls, device_options):
    return IsTpu(device_options)
