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
"""TPU embedding layers."""

import math
from typing import Dict

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import schedule

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_embedding as tpu_embedding_lib
# pylint:enable=g-direct-tensorflow-import


def _ShouldUseTpu(p):
  """Whether we should create embedding tables and run lookup on tpu."""
  return not p.is_inference and py_utils.use_tpu()


class TpuEmbeddingCollection:
  """Manage various TPU embedding related ops and tensors."""

  GRAPH_COLLECTION_NAME = '__tpu_embedding_collection'

  @classmethod
  def Get(cls):
    """Returns the TpuEmbeddingCollection associated with the current graph."""
    emb_collection = tf.get_collection(cls.GRAPH_COLLECTION_NAME)
    assert len(emb_collection) <= 1
    if len(emb_collection) == 1:
      tf.logging.info(
          'TpuEmbeddingCollection singleton already exists, reusing')
      return emb_collection[0]
    else:
      singleton = cls()
      tf.add_to_collection(cls.GRAPH_COLLECTION_NAME, singleton)
      return singleton

  def __init__(self):
    # Maps table name to the list of variables for the corresponding table.
    self._table_vars = py_utils.NestedMap()

    # The TPUEmbedding configuration.
    self._tpu_embedding = None

    # Ops to load/retrieve embedding tables to/from HBM.
    self._load_ops_list = []
    self._retrieve_ops_list = []

    # Maps task name to the (feature_name -> activation_tensor) dict for the
    # corresponding task.
    self._activations_by_task = {}

    # List of (name, value, weight) tuples for summary.
    self._summary_tensors = []

    # Set of embedding feature names.
    self._feature_names = None

    # Schedule for the value that is used as TPU embedding gradient multiplier.
    self._gradient_multiplier_schedule = None

  def AddTableVariables(self, table_name, var_list):
    """Add TPU embedding table variable list to the collection."""
    if table_name in self._table_vars:
      existing_var_list = self._table_vars[table_name]
      if var_list != existing_var_list:
        raise ValueError(
            f'Table {table_name} with a different variable list already '
            f'exists. Existing variable list: {existing_var_list}, '
            f'variable list being added: {var_list}')
      return
    self._table_vars[table_name] = var_list

  @property
  def table_variables(self):
    """Returns a NestedMap mapping table names to variables."""
    return self._table_vars

  @property
  def tpu_embedding(self):
    return self._tpu_embedding

  @tpu_embedding.setter
  def tpu_embedding(self, tpu_embedding):
    self._tpu_embedding = tpu_embedding

  def AddLoadOps(self, load_ops):
    self._load_ops_list.append(load_ops)

  @property
  def load_ops(self):
    return self._load_ops_list

  def AddRetrieveOps(self, retrieve_ops):
    self._retrieve_ops_list.append(retrieve_ops)

  @property
  def retrieve_ops(self):
    return self._retrieve_ops_list

  def AddActivations(self, task, activations):
    if task in self._activations_by_task:
      existing_activations = self._activations_by_task[task]
      raise ValueError(f'Activations for task {task} already exists. '
                       f'Existing activations: {existing_activations}, '
                       f'activations being added: {activations}')
    self._activations_by_task[task] = activations

  @property
  def activations_by_task(self):
    return self._activations_by_task

  def AddSummaryTensor(self, name, value, weight=1.0):
    self._summary_tensors.append((name, value, tf.convert_to_tensor(weight)))

  @property
  def summary_tensors(self):
    return self._summary_tensors

  @property
  def feature_names(self):
    return self._feature_names

  @feature_names.setter
  def feature_names(self, feature_names):
    if self._feature_names and self._feature_names != feature_names:
      raise ValueError('feature_names already exists. '
                       f'Existing feature names: {self._feature_names}, '
                       f'feature names being added: {feature_names}')
    self._feature_names = feature_names

  @property
  def gradient_multiplier_schedule(self):
    return self._gradient_multiplier_schedule

  @gradient_multiplier_schedule.setter
  def gradient_multiplier_schedule(self, multiplier_schedule):
    self._gradient_multiplier_schedule = multiplier_schedule


# TODO(jeffreyzhao): Add the rest of the TPU Embedding optimizers.
class _TPUEmbeddingOptimizer(base_layer.BaseLayer):
  """Base class for TPUEmbeddingLayer, TPUEmbeddingTable optimizers."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('clip_weight_min', None,
             'The minimum value to clip by; None means -infinity.')
    p.Define('clip_weight_max', None,
             'The maximum value to clip by; None means +infinity.')
    p.Define(
        'weight_decay_factor', None,
        'Amount of weight decay to apply; None means that the weights are not '
        'decayed.')
    p.Define(
        'multiply_weight_decay_factor_by_learning_rate', None,
        'If true, weight_decay_factor is multiplied by the current learning '
        'rate.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name

  def CreateOptimizerParameters(self, learning_rate):
    """Create TPUEmbedding API optimzier parameters."""
    return NotImplementedError()

  def CreateSlotVariablesAndOps(self, table_vars, tpu_embedding_table):
    """Create slot variables and infeed/retrieval ops.

    Args:
      table_vars: A list of all embedding table shard variables.
      tpu_embedding_table: Parent TPUEmbeddingTable layer.

    Returns:
      List of load ops
      List of retrieve ops
    """
    return NotImplementedError()


class TPUEmbeddingSGDOptimizer(_TPUEmbeddingOptimizer):
  """SGD optimizer for TPUEmbeddingLayer, TPUEmbeddingTable."""

  def CreateOptimizerParameters(self, learning_rate):
    p = self.params
    return tpu_embedding_lib.StochasticGradientDescentParameters(
        learning_rate=learning_rate,
        clip_weight_min=p.clip_weight_min,
        clip_weight_max=p.clip_weight_max,
        weight_decay_factor=p.weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=p
        .multiply_weight_decay_factor_by_learning_rate)

  def CreateSlotVariablesAndOps(self, table_vars, tpu_embedding_table):
    load_op_list = []
    retrieve_op_list = []

    num_tpu_hosts = tpu_embedding_table.params.num_tpu_hosts
    table_name = tpu_embedding_table.table_name

    for host_id, table_var in zip(range(num_tpu_hosts), table_vars):
      # The slot vars should be on the same device as the table var.
      device_name = tpu_embedding_table.GetDeviceName(host_id)
      with tf.device(device_name), py_utils.outside_all_rewrites():
        # Only the Trainer needs these ops.
        if py_utils.use_tpu():
          # TPU Embedding load/retrieve ops need to be in the outer graph scope.
          with tf.init_scope():
            tf.logging.info('creating load and retrieve ops.')
            load_parameters_op = (
                tpu_embedding_lib.tpu_ops
                .load_tpu_embedding_stochastic_gradient_descent_parameters(
                    parameters=table_var,
                    table_name=table_name,
                    num_shards=num_tpu_hosts,
                    shard_id=host_id))
            load_op_list.append(load_parameters_op)

            retrieved_table = (
                tpu_embedding_lib.tpu_ops
                .retrieve_tpu_embedding_stochastic_gradient_descent_parameters(
                    table_name=table_name,
                    num_shards=num_tpu_hosts,
                    shard_id=host_id))
            retrieve_parameters_op = tpu_embedding_lib.control_flow_ops.group(
                tf.assign(table_var, retrieved_table))
            retrieve_op_list.append(retrieve_parameters_op)

    return load_op_list, retrieve_op_list


class TPUEmbeddingAdagradOptimizer(_TPUEmbeddingOptimizer):
  """Adagrad optimizer for TPUEmbeddingLayer, TPUEmbeddingTable."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('initial_accumulator', 0.1,
             'Initial value of Adagrad accumulator.')
    p.Define(
        'use_gradient_accumulation', True,
        'Setting this to False makes embedding gradients calculation less '
        'accurate but faster. See tpu_embedding_lib for more details.')
    return p

  def CreateOptimizerParameters(self, learning_rate):
    p = self.params
    return tpu_embedding_lib.AdagradParameters(
        learning_rate=learning_rate,
        initial_accumulator=p.initial_accumulator,
        clip_weight_min=p.clip_weight_min,
        clip_weight_max=p.clip_weight_max,
        weight_decay_factor=p.weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=p
        .multiply_weight_decay_factor_by_learning_rate)

  def CreateSlotVariablesAndOps(self, table_vars, tpu_embedding_table):
    p = self.params

    load_op_list = []
    retrieve_op_list = []

    num_tpu_hosts = tpu_embedding_table.params.num_tpu_hosts
    table_name = tpu_embedding_table.table_name
    slot_var_collections = [tpu_embedding_table.__class__.__name__ + '_vars']

    for host_id, table_var in zip(range(num_tpu_hosts), table_vars):
      # The slot vars should be on the same device as the table var.
      device_name = tpu_embedding_table.GetDeviceName(host_id)
      with tf.device(device_name), py_utils.outside_all_rewrites():
        w_ada = py_utils.WeightParams(
            shape=table_var.shape.as_list(),
            init=py_utils.WeightInit.Constant(p.initial_accumulator),
            dtype=p.dtype,
            collections=slot_var_collections)
        var_name = tpu_embedding_table.GetVariableName(host_id) + '/Adagrad'
        tpu_embedding_table.CreateVariable(var_name, w_ada, trainable=False)
        accumulator_var = tpu_embedding_table.vars[var_name]

        # Only the Trainer needs these ops.
        if py_utils.use_tpu():
          # Remove the slot vars from the variable list to void copying them
          # to TPU (by the tf.cast in tpu_embedding_table.theta).
          # pylint: disable=protected-access
          del tpu_embedding_table._private_vars[var_name]
          del tpu_embedding_table._private_theta[var_name]
          # pylint: enable=protected-access

          # TPU Embedding load/retrieve ops need to be in the outer graph scope.
          with tf.init_scope():
            tf.logging.info('creating load and retrieve ops.')
            load_parameters_op = (
                tpu_embedding_lib.tpu_ops.load_tpu_embedding_adagrad_parameters(
                    parameters=table_var,
                    accumulators=accumulator_var,
                    table_name=table_name,
                    num_shards=num_tpu_hosts,
                    shard_id=host_id))
            load_op_list.append(load_parameters_op)

            retrieved_table, retrieved_accumulator = (
                tpu_embedding_lib.tpu_ops
                .retrieve_tpu_embedding_adagrad_parameters(
                    table_name=table_name,
                    num_shards=num_tpu_hosts,
                    shard_id=host_id))
            retrieve_parameters_op = tpu_embedding_lib.control_flow_ops.group(
                tf.assign(table_var, retrieved_table),
                tf.assign(accumulator_var, retrieved_accumulator))
            retrieve_op_list.append(retrieve_parameters_op)

    return load_op_list, retrieve_op_list


class TPUEmbeddingTable(base_layer.BaseLayer):
  """An embedding table controlled by TPUEmbeddingLayer.

  Note that all input_keys needs to be declared upfront.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('vocab_size', 0, 'Depth of the input.')
    p.Define('embedding_dim', 0, 'Depth of the output.')
    p.Define('input_keys', None, 'Name of inputs in InputBatch.')
    p.Define(
        'combiner', 'mean',
        'Must be "sum", "sqrtn", "mean" or None in the case of a '
        '"sequence embedding "')
    p.Define(
        'max_sequence_length', None,
        'If not None or 0, embedding lookup will return a '
        '"sequence embedding" of shape '
        '`[batch, max_sequence_length, embedding_dim]` without applying a '
        'sequence  reducing combiner')
    p.Define('num_tpu_hosts', 0, 'Total number of TPU hosts.')
    p.Define(
        'optimizer', None,
        'Table optimizer parameters. Will override the optimizer parameters '
        'defined in this table\'s TPUEmbeddingLayer.')
    p.Define('learning_rate', None,
             'Overrides TPUEmbeddingLayer\'s learning_rate.')
    p.Define('lr_schedule', None, 'Overrides TPUEmbeddingLayer\'s lr_schedule.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.vocab_size > 0
    assert p.embedding_dim > 0
    assert p.input_keys
    assert p.name
    assert p.num_tpu_hosts > 0
    if p.combiner is None:
      assert p.max_sequence_length
    if p.max_sequence_length is not None and p.max_sequence_length > 0:
      assert p.combiner is None
    assert p.optimizer
    assert p.learning_rate
    assert p.lr_schedule

    self._ids_per_shard = int(math.ceil(float(p.vocab_size) / p.num_tpu_hosts))
    self._padded_vocab_size = self._ids_per_shard * p.num_tpu_hosts
    self._input_keys = p.input_keys

    self._max_sequence_length = 0
    if p.max_sequence_length:
      self._max_sequence_length = p.max_sequence_length

    self.CreateChild('optimizer', p.optimizer)
    self.CreateChild('schedule', p.lr_schedule)
    self._tpu_embedding_collection = TpuEmbeddingCollection.Get()

    def LearningRateFn(step):
      with py_utils.GlobalStepContext(step):
        lr = self.schedule.Value() * p.learning_rate
      self._tpu_embedding_collection.AddSummaryTensor(
          'tpu_embedding_lr/{}'.format(p.name), lr)
      return lr

    self._table_name = '{}_table'.format(p.name)
    self._table_config = tpu_embedding_lib.TableConfig(
        self._padded_vocab_size,
        p.embedding_dim,
        combiner=p.combiner,
        learning_rate=None,
        learning_rate_fn=LearningRateFn,
        # All TableConfigs passed to API will have a learning rate function,
        # so the learning_rate in the optimization_parameters is not used.
        optimization_parameters=self.optimizer.CreateOptimizerParameters(
            p.learning_rate))

    self._load_op_list = []
    self._retrieve_op_list = []

  def _CreateLayerVariables(self):
    p = self.params
    w_pc = py_utils.WeightParams(
        shape=[self._ids_per_shard, p.embedding_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    embedding_table_vars = []
    for i in range(p.num_tpu_hosts):
      device_name = self.GetDeviceName(i)
      with tf.device(device_name), py_utils.outside_all_rewrites():
        var_name = self.GetVariableName(i)
        self.CreateVariable(var_name, w_pc)
        embedding_var = self.vars[var_name]
        embedding_table_vars.append(embedding_var)
        # Remove from _private_vars / _private_thetas to be added later as wm.
        del self._private_vars[var_name]
        del self._private_theta[var_name]

    self._tpu_embedding_collection.AddTableVariables(self.table_name,
                                                     embedding_table_vars)

    if not _ShouldUseTpu(p):
      # We don't want to add this for TrainerTpu, otherwise the identity
      # reference leads to copying the embedding to the TPU for no reason.
      # However, this is needed for CPU (eval/decode/controller).
      self._private_vars['wm'] = embedding_table_vars
      self._private_theta['wm'] = [tf.identity(v) for v in embedding_table_vars]

    # Only trainer and controller need slot variables and load/retrieve ops.
    if not self.do_eval and not p.is_inference:
      self._load_op_list, self._retrieve_op_list = (
          self.optimizer.CreateSlotVariablesAndOps(embedding_table_vars, self))

  # Return device to place sharded variables on.
  def GetDeviceName(self, host_id):
    if self.do_eval:
      return None
    else:
      return '{}/replica:0/task:{}/device:CPU:0'.format(
          self.cluster.params.worker.name, host_id)

  # Return variable name for embedding table shards.
  def GetVariableName(self, host_id):
    return 'var_%d' % host_id

  @property
  def table_config(self):
    return self._table_config

  @property
  def table_name(self):
    return self._table_name

  @property
  def retrieve_op_list(self):
    return self._retrieve_op_list

  @property
  def load_op_list(self):
    return self._load_op_list

  @property
  def input_keys(self):
    return self._input_keys

  @property
  def max_sequence_length(self):
    return self._max_sequence_length

  def _SequenceEmbLookup(self, dense_ids: tf.Tensor,
                         partition_strategy: str) -> Dict[str, tf.Tensor]:
    """Sequence embedding lookup.

    Note that we do not support padding ids in sequence embeddings.

    Args:
      dense_ids: A dict of `input_key` string -> [batch, sequence] int32 Tensor.
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      An activations dict of string -> float32 Tensor of dimension [batch,
      max_sequence_length, embedding_dim]
    """
    p = self.params
    embs = tf.nn.embedding_lookup(
        self.theta.wm,
        tf.reshape(dense_ids, [-1]),
        partition_strategy=partition_strategy)
    out_shape = tf.concat([tf.shape(dense_ids), [p.embedding_dim]], 0)
    return tf.reshape(embs, out_shape)

  def _CombinerEmbLookup(self, sparse_ids: tf.SparseTensor,
                         partition_strategy: str) -> Dict[str, tf.Tensor]:
    """Combiner embedding lookup.

    Args:
      sparse_ids: A dict of `input_key` string -> [batch, ...] int32
        SparseTensor.
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      An activations dict of string -> float32 Tensor of dimension [batch, 1,
      embedding_dim]
    """
    p = self.params
    embs = tf.nn.embedding_lookup_sparse(
        self.theta.wm,
        sparse_ids,
        None,  # sp_weights
        combiner=p.combiner,
        partition_strategy=partition_strategy)
    batch_size = sparse_ids.dense_shape[0]
    # For tf.nn.embedding_lookup_sparse, output.dim0 might be different from
    # sparse_ids.dense_shape.dim0.
    # Explicitly pad results to maintain dim0=batch.
    dim0_padlen = tf.cast(batch_size, tf.int32) - tf.shape(embs)[0]
    embs = tf.pad(embs, [[0, dim0_padlen], [0, 0]])
    # [batch, 1, embedding_dim]
    embs = py_utils.HasShape(embs, [batch_size], ndims=1)
    return tf.expand_dims(embs, 1)

  def CpuEmbLookup(self, ids_map: Dict[str, tf.Tensor],
                   partition_strategy: str) -> Dict[str, tf.Tensor]:
    """CPU evaluation embedding lookup for dense tensors.

    Args:
      ids_map: A dict of `input_key` string -> [batch, sequence] int32 Tensor.
        For sequence embeddings, -1 is used as a padding id. Non-sequence
        embeddings do not support padded ids.
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      An activations dict of string -> float32 Tensor.
      For non-sequence embeddings: [batch, 1, embedding_dim]
      For sequence embeddings: [batch, max_sequence_length, embedding_dim]
    """
    rets = py_utils.NestedMap()
    if self.max_sequence_length > 0:
      # "Sequence embedding", no combiner case
      for k, ids in ids_map.items():
        rets[k] = self._SequenceEmbLookup(ids, partition_strategy)
    else:
      # Non-"Sequence embedding", combiner case
      for k, ids in ids_map.items():
        # Dense to sparse.
        dense_shape = tf.shape(ids, out_type=tf.int64)
        sample_indices = tf.cast(tf.where(tf.not_equal(ids, -1)), tf.int64)
        embedding_indices = tf.cast(tf.gather_nd(ids, sample_indices), tf.int64)
        # [?, embedding_dim]
        sparse_ids = tf.SparseTensor(
            indices=sample_indices,
            values=embedding_indices,
            dense_shape=dense_shape)
        rets[k] = self._CombinerEmbLookup(sparse_ids, partition_strategy)
    return rets

  def CpuEmbLookupSparse(self, ids_map: Dict[str, tf.SparseTensor],
                         partition_strategy: str) -> Dict[str, tf.Tensor]:
    """CPU evaluation embedding lookup for SparseTensors.

    Args:
      ids_map: A dict of `input_key` string -> [batch, ...] int32 SparseTensor.
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      An activations dict of string -> float32 Tensor.
      For non-sequence embeddings: [batch, 1, embedding_dim]
      For sequence embeddings: [batch, max_sequence_length, embedding_dim]
    """
    rets = py_utils.NestedMap()
    if self.max_sequence_length > 0:
      # "Sequence embedding", no combiner case
      for k, ids in ids_map.items():
        # Sparse to dense.
        dense_ids = tf.sparse.to_dense(ids, default_value=-1)
        rets[k] = self._SequenceEmbLookup(dense_ids, partition_strategy)
    else:
      # Non-"Sequence embedding", combiner case
      for k, ids in ids_map.items():
        rets[k] = self._CombinerEmbLookup(ids, partition_strategy)
    return rets


class TPUEmbeddingLayer(base_layer.BaseLayer):
  """Monolithic interface to TPU embedding.

  This layer has some important caveats, due to the interface of the
  TPU embedding hardware. Its behavior most closely mimics that of
  tf.nn.embedding_lookup_sparse.

  Supports multiple tables and multiple input_keys per table.
  Requires its own optimizer parameters.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('tables', None, 'TPUEmbeddingTables')
    p.Define('pipeline_execution_with_tensor_core', False,
             'Set to True to be faster. See tpu_embedding.py for details.')
    p.Define('batch_size', 0, 'Per-core batch size.')
    p.Define(
        'optimizer', TPUEmbeddingAdagradOptimizer.Params(),
        'Layer optimizer parameters. Will be used for any TPUEmbeddingTables '
        'with None optimizer parameters.')
    p.Define('learning_rate', 0.0, 'Learning rate.')
    p.Define(
        'lr_schedule', schedule.ContinuousSchedule.Params(),
        'Lingvo learning rate schedule. Will be multiplied to learning rate.')
    p.Define(
        'partition_strategy', 'div', 'A string, either "mod" or "div", '
        'specifying how to map the lookup id to the embedding tensor. For '
        'more information see `tf.nn.embedding_lookup_sparse`.')
    p.Define(
        'gradient_multiplier_schedule', schedule.ConstantOne.Params(),
        'Values from this schedule will be multiplied to the embedding '
        'gradients. Gradients from Tensorcore will not be affected.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    assert p.tables
    assert p.batch_size > 0
    assert p.name
    assert p.optimizer
    assert p.learning_rate
    assert p.lr_schedule
    assert p.gradient_multiplier_schedule
    assert p.partition_strategy in ['mod', 'div']

    num_tpu_hosts = p.tables[0].num_tpu_hosts
    assert all([t.num_tpu_hosts == num_tpu_hosts for t in p.tables])

    # Stop if a table has no optimizer parameters and the layer also has no
    # optimizer parameters
    table_optimizer_missing = any(
        table_params.optimizer is None for table_params in p.tables)
    if not p.optimizer and table_optimizer_missing:
      raise ValueError(
          'A table is missing optimizer parameters, and no layer-level '
          'optimizer parameters were given.')
    elif table_optimizer_missing:
      for table_params in p.tables:
        if table_params.optimizer is None:
          table_params.optimizer = p.optimizer.Copy()
        if table_params.learning_rate is None:
          table_params.learning_rate = p.learning_rate
        if table_params.lr_schedule is None:
          table_params.lr_schedule = p.lr_schedule.Copy()

    self.CreateChildren('tables', p.tables)
    self.CreateChild('gradient_multiplier_schedule',
                     p.gradient_multiplier_schedule)
    self._tpu_embedding_collection = TpuEmbeddingCollection.Get()

    # Save embedding feature names in the collection.
    feature_names = set()
    for table in self.tables:
      for feature in table.input_keys:
        feature_names.add(feature)
    self._tpu_embedding_collection.feature_names = feature_names

  def _CreateChildrenVariables(self):
    # Backwards compatibility: manually call child.InstantiateVariables()
    # outside of tf.variable_scope(p.name).
    for table in self.tables:
      table.InstantiateVariables()
    super()._CreateChildrenVariables()

  def _CheckTPUEmbeddingConfig(self, tpu_embedding, table_to_config_dict,
                               feature_to_config_dict, global_batch_size):
    """Check that the existing tpu_embedding config matches the given ones."""

    def _Match(d1, d2, namedtuple_attrs_to_check):
      if len(d1) != len(d2):
        return False
      for k, v1 in d1.items():
        if k not in d2:
          return False
        v2 = d2[k]
        for attr in namedtuple_attrs_to_check:
          if getattr(v1, attr) != getattr(v2, attr):
            return False
      return True

    # We just check numeric/string settings for simplicity, this excludes things
    # like learning_rate_fn, optimization_parameters, etc since it's hard to
    # compare them.
    if not _Match(tpu_embedding.table_to_config_dict, table_to_config_dict,
                  ['vocabulary_size', 'dimension', 'combiner']):
      raise ValueError('table_to_config_dict mismatch. '
                       f'Expecting {tpu_embedding.table_to_config_dict}, '
                       f'got {table_to_config_dict}')
    if not _Match(tpu_embedding.feature_to_config_dict, feature_to_config_dict,
                  ['table_id', 'max_sequence_length']):
      raise ValueError('feature_to_config_dict mismatch. '
                       f'Expecting {tpu_embedding.feature_to_config_dict}, '
                       f'got {feature_to_config_dict}')
    if (tpu_embedding.batch_size_per_core * tpu_embedding.num_cores !=
        global_batch_size):
      raise ValueError(
          'global_batch_size mismatch. '
          f'batch_size_per_core: {tpu_embedding.batch_size_per_core}, '
          f'num_cores: {tpu_embedding.num_cores}, '
          f'global_batch_size: {global_batch_size}')

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params

    load_op_list = []
    retrieve_op_list = []

    # At the feature level, track which are associated
    # with "sequence embeddings".
    self._sequence_features = {}

    if _ShouldUseTpu(p):
      num_cores = self.cluster.params.worker.tpus_per_replica
      global_batch_size = (
          self.params.batch_size * self.cluster.num_splits_per_client)
      table_to_config_dict = {}
      feature_to_config_dict = {}
      for table in self.tables:
        table_to_config_dict[table.table_name] = table.table_config
        load_op_list += table.load_op_list
        retrieve_op_list += table.retrieve_op_list
        for feature in table.input_keys:
          if table.max_sequence_length > 0:
            self._sequence_features[feature] = True
          feature_to_config_dict[feature] = tpu_embedding_lib.FeatureConfig(
              table.table_name, max_sequence_length=table.max_sequence_length)

      tpu_embedding = self._tpu_embedding_collection.tpu_embedding
      if tpu_embedding:
        self._CheckTPUEmbeddingConfig(tpu_embedding, table_to_config_dict,
                                      feature_to_config_dict, global_batch_size)
        tf.logging.info('TPUEmbedding API singleton already exists, reusing')
        self._tpu_embedding = tpu_embedding
      else:
        tf.logging.info('adding load and retrieve ops to collection.')
        self._tpu_embedding_collection.AddLoadOps(load_op_list)
        self._tpu_embedding_collection.AddRetrieveOps(retrieve_op_list)

        mode = tpu_embedding_lib.TRAINING
        device_config = tpu_embedding_lib.DeviceConfig(
            num_cores=num_cores,
            num_hosts=self.params.tables[0].num_tpu_hosts,
            job_name=self.cluster.params.worker.name)
        self._tpu_embedding = tpu_embedding_lib.TPUEmbedding(
            table_to_config_dict,
            feature_to_config_dict,
            global_batch_size,
            mode,
            master=None,
            pipeline_execution_with_tensor_core=(
                self.params.pipeline_execution_with_tensor_core),
            partition_strategy=p.partition_strategy,
            device_config=device_config)
        self._tpu_embedding_collection.tpu_embedding = self._tpu_embedding
        self._tpu_embedding_collection.gradient_multiplier_schedule = (
            self.gradient_multiplier_schedule)

  def _TpuEmbLookup(self) -> Dict[str, tf.Tensor]:
    """TPU Embedding lookup."""
    activations = self._tpu_embedding.get_activations()
    task = py_utils.GetTaskCallScope()
    self._tpu_embedding_collection.AddActivations(task, activations)

    ret = py_utils.NestedMap()
    for k, v in activations.items():
      if k in self._sequence_features:
        ret[k] = v
      else:
        # Non-sequence embeddings, we fill the "time" dimension with 1.
        with tf.name_scope(k):
          ret[k] = tf.expand_dims(v, axis=[1])
    return ret

  def EmbLookup(self, ids_map: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Looks up embedding vectors for each entry in dense Tensor ids_map.

    Since the TPUEmbedding is monolothic, and consulted once per
    FProp/BProp, we must centralize the lookup. Thus, for multiple
    features, we contain them into a single-lookup rather than allowing
    the caller to call Lookup multiple times.

    Args:
      ids_map: A dict of `input_key` string -> [batch, sequence] int32 Tensor.
        For sequence embeddings, -1 is used as a padding id. Non-sequence
        embeddings do not support padded ids.

    Returns:
      Activations dict of string ->
      For non-sequence embeddings:  [batch, 1, embedding_dim],
      For sequence embeddings: [batch, max_sequence_length, embedding_dim]
      float32 Tensor.
    """
    p = self.params

    def CpuEmbLookup(ids_map):
      """CPU evaluation embedding lookup."""
      rets = py_utils.NestedMap()
      for table in self.tables:
        table_id_map = {}
        for key in table.input_keys:
          table_id_map[key] = ids_map[key]
        table_rets = table.CpuEmbLookup(table_id_map, p.partition_strategy)
        # Merge table_rets with rets
        for k, v in table_rets.items():
          rets[k] = v
      return rets

    if _ShouldUseTpu(p):
      return self._TpuEmbLookup()
    else:
      return CpuEmbLookup(ids_map)

  def EmbLookupSparse(
      self, ids_map: Dict[str, tf.SparseTensor]) -> Dict[str, tf.Tensor]:
    """Looks up embedding vectors for each entry in SparseTensor ids_map.

    Since the TPUEmbedding is monolothic, and consulted once per
    FProp/BProp, we must centralize the lookup. Thus, for multiple
    features, we contain them into a single-lookup rather than allowing
    the caller to call Lookup multiple times.

    Args:
      ids_map: A dict of `input_key` string -> [batch, ...] int32 SparseTensor.

    Returns:
      Activations dict of string ->
      For non-sequence embeddings:  [batch, 1, embedding_dim],
      For sequence embeddings: [batch, max_sequence_length, embedding_dim]
      float32 Tensor.
    """
    p = self.params

    def CpuEmbLookupSparse(ids_map):
      """CPU evaluation embedding lookup."""
      rets = py_utils.NestedMap()
      for table in self.tables:
        table_id_map = {}
        for key in table.input_keys:
          table_id_map[key] = ids_map[key]
        table_rets = table.CpuEmbLookupSparse(table_id_map,
                                              p.partition_strategy)
        # Merge table_rets with rets
        for k, v in table_rets.items():
          rets[k] = v
      return rets

    if _ShouldUseTpu(p):
      return self._TpuEmbLookup()
    else:
      return CpuEmbLookupSparse(ids_map)
