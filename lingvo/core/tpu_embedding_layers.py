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

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import schedule

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_embedding as tpu_embedding_lib
# pylint:enable=g-direct-tensorflow-import


def _IsTpuTraining(p):
  """Whether we should create embedding tables and run lookup on tpu."""
  return not p.is_inference and py_utils.use_tpu()


def _RemovePrivateVar(layer, var_name):
  """Remove a variable by name from `layer`.

  This is usually used to avoid copying the variable to TPU, for example, by the
  tf.cast when accessing layer.theta.

  Args:
    layer: The layer to remove the variable from.
    var_name: The name of the variable to remove.
  """
  # pylint: disable=protected-access
  del layer._private_vars[var_name]
  del layer._private_theta[var_name]
  # pylint: enable=protected-access


class TpuEmbeddingCollection:
  """Manage various TPU embedding related ops and tensors."""

  @classmethod
  def Get(cls):
    """Returns the TpuEmbeddingCollection associated with the current graph."""
    emb_collection = py_utils.GetTpuEmbeddingGraphCollection()
    assert len(emb_collection) <= 1
    if len(emb_collection) == 1:
      tf.logging.info(
          'TpuEmbeddingCollection singleton already exists, reusing')
      return emb_collection[0]
    else:
      singleton = cls()
      emb_collection.append(singleton)
      return singleton

  def __init__(self):
    # Maps table name to the list of variables for the corresponding table.
    self._table_vars = py_utils.NestedMap()

    # The TPUEmbedding configuration.
    self._tpu_embedding = None

    # Maps table name to the list of ops that loads/retrieves embedding tables
    # to/from TPU.
    self._load_ops_map = py_utils.NestedMap()
    self._retrieve_ops_map = py_utils.NestedMap()

    # Maps task name to the (feature_name -> activation_tensor) dict for the
    # corresponding task.
    self._activations_by_task = {}

    # List of (name, value, weight) tuples for summary.
    self._summary_tensors = []

    # Set of embedding feature names.
    self._feature_names = None

    # Schedule for the value that is used as TPU embedding gradient multiplier.
    self._gradient_multiplier_schedule = None

    # Maps task name to the mode used by that task.
    self._mode_by_task = {}

    # Maps task name to the send gradient op for that task. Mainly used to
    # ensure that send gradient op is created only once for each task.
    self._send_gradient_op_by_task = {}

  def AddTableVariables(self, table_name, var_list):
    """Add TPU embedding table variable list to the collection."""
    if table_name in self._table_vars:
      raise ValueError(f'Variables for table {table_name} already exist.')
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
    if self._tpu_embedding is not None:
      raise ValueError('TPUEmbedding already set before.')
    self._tpu_embedding = tpu_embedding

  def AddLoadRetrieveOps(self, table_name, load_ops, retrieve_ops):
    if table_name in self._load_ops_map:
      raise ValueError(f'Load ops for table {table_name} already exist.')
    assert table_name not in self._retrieve_ops_map
    self._load_ops_map[table_name] = load_ops
    self._retrieve_ops_map[table_name] = retrieve_ops

  @property
  def load_ops(self):
    return self._load_ops_map

  @property
  def retrieve_ops(self):
    return self._retrieve_ops_map

  def _ValidateTaskScope(self, task_call_scope):
    if not task_call_scope:
      raise ValueError(
          'It expects a non-empty task call scope name, but get '
          f'{task_call_scope}. This usually means the current code is not run '
          'under a py_utils.TaskCallScope() context.')

  def AddActivations(self, task_call_scope):
    self._ValidateTaskScope(task_call_scope)
    tf.logging.info(
        f'Adding TPU embedding activations for task {task_call_scope}.')
    if task_call_scope not in self._activations_by_task:
      activations = self._tpu_embedding.get_activations()
      self._activations_by_task[task_call_scope] = activations
    return self._activations_by_task[task_call_scope]

  def GetActivations(self, task_call_scope):
    tf.logging.info(
        f'Getting TPU embedding activations for task {task_call_scope}.')
    if task_call_scope in self._activations_by_task:
      self._ValidateTaskScope(task_call_scope)
      return self._activations_by_task[task_call_scope]
    return None

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

  def SetGradientMultiplierSchedule(self, multiplier_schedule):
    if self._gradient_multiplier_schedule is not None:
      raise ValueError('gradient_multiplier_schedule was set before.')
    self._gradient_multiplier_schedule = multiplier_schedule

  def SetTaskMode(self, task_call_scope, mode):
    self._ValidateTaskScope(task_call_scope)
    tf.logging.info(
        f'Setting TPU embedding mode for task {task_call_scope} as {mode}.')
    self._mode_by_task[task_call_scope] = mode

  def ShouldStopGradient(self, task_call_scope):
    self._ValidateTaskScope(task_call_scope)
    if task_call_scope not in self._mode_by_task:
      raise ValueError(
          f'TPU embedding mode for task {task_call_scope} not found.')
    should_stop_gradient = (self._mode_by_task[task_call_scope] != 'train')
    tf.logging.info(('Disabled' if should_stop_gradient else 'Enabled') +
                    f' TPU embedding gradient for task {task_call_scope}.')
    return should_stop_gradient

  def ApplyGradients(self, task_call_scope, feature_to_gradient_dict):
    """Apply tpu embedding gradient updates.

    Args:
      task_call_scope: The current task call scope name.
      feature_to_gradient_dict: A `py_utils.NestedMap` of: tpu embedding feature
        name -> gradient tensor for the embedding feature.

    Returns:
      The gradient update op and a dict of eval metrics.

    Raises:
      ValueError: if gradients have been applied before for the current task.
    """
    self._ValidateTaskScope(task_call_scope)
    if task_call_scope in self._send_gradient_op_by_task:
      raise ValueError(
          f'Send gradient op for task {task_call_scope} already exist.')
    tf.logging.info(
        f'Applying TPU embedding gradients for task {task_call_scope}.')

    # Apply gradient multiplier schedule.
    grad_multiplier = self._gradient_multiplier_schedule.Value()
    feature_to_gradient_dict = feature_to_gradient_dict.Transform(
        lambda g: g * grad_multiplier)

    send_gradient_op = (
        self._tpu_embedding.generate_send_gradients_op(
            feature_to_gradient_dict, step=py_utils.GetGlobalStep()))
    self._send_gradient_op_by_task[task_call_scope] = send_gradient_op

    activations = self.GetActivations(task_call_scope).values()
    eval_metrics = {
        'tpu_embedding_activation_norm':
            (tf.sqrt(py_utils.SumSquared(activations)), tf.constant(1.0)),
        'tpu_embedding_grad_norm':
            (tf.sqrt(py_utils.SumSquared(feature_to_gradient_dict.Flatten())),
             tf.constant(1.0)),
        'tpu_embedding_gradient_multiplier':
            (grad_multiplier, tf.constant(1.0)),
    }
    return send_gradient_op, eval_metrics


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
    """Create slot variables and load/retrieve ops.

    Args:
      table_vars: A list of all embedding table shard variables.
      tpu_embedding_table: Parent TPUEmbeddingTable layer.

    Returns:
      List of load ops
      List of retrieve ops
    """
    raise NotImplementedError()


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
          # Remove the slot vars from the variable list to avoid them being
          # copied to TPU.
          _RemovePrivateVar(tpu_embedding_table, var_name)

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


class TPUEmbeddingAdamOptimizer(_TPUEmbeddingOptimizer):
  """Adam optimizer for TPUEmbeddingLayer, TPUEmbeddingTable."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'clip_gradient_min', None,
        'Controls clipping of the gradient. The minimum value to clip by.')
    p.Define('clip_gradient_max', None, 'The max value to clip by.')
    p.Define(
        'sum_inside_sqrt', True, 'When this is true, the Adam update'
        'formula is changed from m / (sqrt(v) + epsilon) to m / '
        'sqrt(v + epsilon**2). This option improves the performance of'
        'TPU training and is not expected to harm model quality.')
    p.Define('lazy_adam', True, 'Use lazy Adam instead of Adam. Lazy Adam'
             'trains faster.')
    p.Define('beta1', 0.9, 'The exponential decay rate for the 1st moment'
             'estimates')
    p.Define('beta2', 0.999, 'The exponential decay rate for the 2nd moment'
             'estimates')
    p.Define('epsilon', 1e-08, 'A small constant for numerical stability')
    p.Define(
        'use_gradient_accumulation', True, 'Setting this to False makes'
        'embedding gradients calculation less accurate but faster')

    return p

  def CreateOptimizerParameters(self, learning_rate):
    p = self.params
    return tpu_embedding_lib.AdamParameters(
        learning_rate=learning_rate,
        beta1=p.beta1,
        beta2=p.beta2,
        epsilon=p.epsilon,
        lazy_adam=p.lazy_adam,
        sum_inside_sqrt=p.sum_inside_sqrt,
        use_gradient_accumulation=p.use_gradient_accumulation,
        clip_weight_min=p.clip_weight_min,
        clip_weight_max=p.clip_weight_max,
        weight_decay_factor=p.weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=p
        .multiply_weight_decay_factor_by_learning_rate,
        clip_gradient_min=p.clip_gradient_min,
        clip_gradient_max=p.clip_gradient_max)

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
        m_adam = py_utils.WeightParams(
            shape=table_var.shape.as_list(),
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=slot_var_collections)
        var_name_m = tpu_embedding_table.GetVariableName(host_id) + '/Adam/m'
        tpu_embedding_table.CreateVariable(var_name_m, m_adam, trainable=False)
        m_var = tpu_embedding_table.vars[var_name_m]

        v_adam = py_utils.WeightParams(
            shape=table_var.shape.as_list(),
            init=py_utils.WeightInit.Constant(0.0),
            dtype=p.dtype,
            collections=slot_var_collections)
        var_name_v = tpu_embedding_table.GetVariableName(host_id) + '/Adam/v'
        tpu_embedding_table.CreateVariable(var_name_v, v_adam, trainable=False)
        v_var = tpu_embedding_table.vars[var_name_v]

        # Only the Trainer needs these ops.
        if py_utils.use_tpu():
          # Remove the slot vars from the variable list to avoid them being
          # copied to TPU.
          _RemovePrivateVar(tpu_embedding_table, var_name_m)
          _RemovePrivateVar(tpu_embedding_table, var_name_v)

          # TPU Embedding load/retrieve ops need to be in the outer graph scope.
          with tf.init_scope():
            tf.logging.info('creating load and retrieve ops.')
            load_parameters_op = (
                tpu_embedding_lib.tpu_ops.load_tpu_embedding_adam_parameters(
                    parameters=table_var,
                    momenta=m_var,
                    velocities=v_var,
                    table_name=table_name,
                    num_shards=num_tpu_hosts,
                    shard_id=host_id))
            load_op_list.append(load_parameters_op)

            retrieved_table, retrieved_m, retrieved_v = (
                tpu_embedding_lib.tpu_ops
                .retrieve_tpu_embedding_adam_parameters(
                    table_name=table_name,
                    num_shards=num_tpu_hosts,
                    shard_id=host_id))
            retrieve_parameters_op = tpu_embedding_lib.control_flow_ops.group(
                tf.assign(table_var, retrieved_table),
                tf.assign(m_var, retrieved_m), tf.assign(v_var, retrieved_v))
            retrieve_op_list.append(retrieve_parameters_op)

    return load_op_list, retrieve_op_list


class TPUEmbeddingFTRLOptimizer(_TPUEmbeddingOptimizer):
  """FTRL optimizer for TPUEmbeddingLayer, TPUEmbeddingTable."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'learning_rate_power', -0.5,
        'A float value, must be less or equal to zero. Controls how the'
        'learning rate decreases during training. Use zero for a fixed learning'
        'rate.')
    p.Define(
        'initial_accumulator_value', 0.1, 'The starting value for'
        'accumulators. Only zero or positive values are allowed.')
    p.Define(
        'l1_regularization_strength', 0.0, 'A float value, must be greater'
        'than or equal to zero. Defaults to 0.0.')
    p.Define(
        'l2_regularization_strength', 0.0, 'A float value, must be greater'
        'than or equal to zero. Defaults to 0.0.')
    p.Define('multiply_linear_by_learning_rate', False, 'Whether multiply'
             'linear by learning rate.')
    p.Define(
        'beta', 0.0, 'A float value, representing the beta value from the'
        'FTLR paper. Defaults to 0.0.')
    p.Define('allow_zero_accumulator', False, 'Whether allowing zero'
             'accumulator.')
    p.Define('clip_gradient_min', None, 'Clip gradient min value.')
    p.Define('clip_gradient_max', None, 'Clip gradient max value.')
    p.Define('use_gradient_accumulation', True, 'Use gradient accumulation.')
    p.Define('initial_linear_value', 0.0, 'Initial linear value.')

    return p

  def CreateOptimizerParameters(self, learning_rate):
    p = self.params
    return tpu_embedding_lib.FtrlParameters(
        learning_rate=learning_rate,
        learning_rate_power=p.learning_rate_power,
        initial_accumulator_value=p.initial_accumulator_value,
        l1_regularization_strength=p.l1_regularization_strength,
        l2_regularization_strength=p.l2_regularization_strength,
        use_gradient_accumulation=p.use_gradient_accumulation,
        clip_weight_min=p.clip_weight_min,
        clip_weight_max=p.clip_weight_max,
        weight_decay_factor=p.weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=p
        .multiply_weight_decay_factor_by_learning_rate,
        multiply_linear_by_learning_rate=p.multiply_linear_by_learning_rate,
        beta=p.beta,
        allow_zero_accumulator=p.allow_zero_accumulator,
        clip_gradient_min=p.clip_gradient_min,
        clip_gradient_max=p.clip_gradient_max)

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
        accumulator = py_utils.WeightParams(
            shape=table_var.shape.as_list(),
            init=py_utils.WeightInit.Constant(p.initial_accumulator_value),
            dtype=p.dtype,
            collections=slot_var_collections)
        accumulator_name = (
            tpu_embedding_table.GetVariableName(host_id) + '/Ftrl')
        tpu_embedding_table.CreateVariable(
            accumulator_name, accumulator, trainable=False)
        accumulator_var = tpu_embedding_table.vars[accumulator_name]

        linear = py_utils.WeightParams(
            shape=table_var.shape.as_list(),
            init=py_utils.WeightInit.Constant(p.initial_linear_value),
            dtype=p.dtype,
            collections=slot_var_collections)
        linear_name = tpu_embedding_table.GetVariableName(host_id) + '/Ftrl_1'
        tpu_embedding_table.CreateVariable(linear_name, linear, trainable=False)
        linear_var = tpu_embedding_table.vars[linear_name]

        # Only the Trainer needs these ops.
        if py_utils.use_tpu():
          # Remove the slot vars from the variable list to avoid them being
          # copied to TPU.
          _RemovePrivateVar(tpu_embedding_table, accumulator_name)
          _RemovePrivateVar(tpu_embedding_table, linear_name)

          # TPU Embedding load/retrieve ops need to be in the outer graph scope.
          with tf.init_scope():
            tf.logging.info('creating load and retrieve ops.')
            load_parameters_op = (
                tpu_embedding_lib.tpu_ops.load_tpu_embedding_ftrl_parameters(
                    parameters=table_var,
                    accumulators=accumulator_var,
                    linears=linear_var,
                    table_name=table_name,
                    num_shards=num_tpu_hosts,
                    shard_id=host_id))
            load_op_list.append(load_parameters_op)

            retrieved_table, retrieved_accumulator, retrieved_linear = (
                tpu_embedding_lib.tpu_ops
                .retrieve_tpu_embedding_ftrl_parameters(
                    table_name=table_name,
                    num_shards=num_tpu_hosts,
                    shard_id=host_id))
            retrieve_parameters_op = tpu_embedding_lib.control_flow_ops.group(
                tf.assign(table_var, retrieved_table),
                tf.assign(accumulator_var, retrieved_accumulator),
                tf.assign(linear_var, retrieved_linear))
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
    p.Define(
        'inference_use_merged_variable', False,
        'Whether to use merged embedding table variable during inference. '
        'If set to True, only one table variable will be created, and '
        'the user will need to manually merge the sharded table variables '
        'in the trained checkpoint before generating the inference graph.')
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

  def _CreateLayerVariables(self):
    p = self.params

    # Reuse the singleton table variables if they were created before.
    all_table_vars = self._tpu_embedding_collection.table_variables
    if self.table_name in all_table_vars:
      embedding_table_vars = all_table_vars[self.table_name]
    else:
      w_pc = py_utils.WeightParams(
          shape=[self._ids_per_shard, p.embedding_dim],
          init=p.params_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])

      embedding_table_vars = []
      if p.is_inference and p.inference_use_merged_variable:
        with py_utils.outside_all_rewrites():
          var_name = 'merged_var'
          self.CreateVariable(var_name, w_pc)
          embedding_var = self.vars[var_name]
          embedding_table_vars.append(embedding_var)
          # Remove from _private_vars / _private_thetas to be added later as wm.
          _RemovePrivateVar(self, var_name)
      else:
        for i in range(p.num_tpu_hosts):
          device_name = self.GetDeviceName(i)
          with tf.device(device_name), py_utils.outside_all_rewrites():
            var_name = self.GetVariableName(i)
            self.CreateVariable(var_name, w_pc)
            embedding_var = self.vars[var_name]
            embedding_table_vars.append(embedding_var)
            # Remove from _private_vars / _private_thetas to be added later as
            # wm.
            _RemovePrivateVar(self, var_name)

      # Track the table variables so they can be excluded from EMA.
      self._tpu_embedding_collection.AddTableVariables(self.table_name,
                                                       embedding_table_vars)

    if not _IsTpuTraining(p):
      # We don't need this for TrainerTpu, as the vars are not directly
      # accessed besides in the TPU embeddding load/retrieve ops.
      # However, this is needed for CPU (eval/decode/controller).
      self._private_vars['wm'] = embedding_table_vars
      self._private_theta['wm'] = embedding_table_vars

    # If slot variables and load/retrieve ops were created before, maybe by a
    # different program or task, don't create it again.
    # Note that there should be only one copy of slot variables and
    # load/retrieve ops in the graph and they're shared by different
    # tasks/programs.
    all_load_ops = self._tpu_embedding_collection.load_ops
    if self.table_name not in all_load_ops:
      assert self.table_name not in self._tpu_embedding_collection.retrieve_ops
      # Only trainer and controller (for checkpointing) need slot variables.
      # Only trainer needs load/retrieve ops.
      if not self.do_eval and not p.is_inference:
        load_ops, retrieve_ops = self.optimizer.CreateSlotVariablesAndOps(
            embedding_table_vars, self)
        self._tpu_embedding_collection.AddLoadRetrieveOps(
            self.table_name, load_ops, retrieve_ops)

  # Return device to place sharded variables on.
  def GetDeviceName(self, host_id):
    if self.params.is_inference:
      # This is to place variables on the same device as other variables.
      return None
    if self.do_eval:
      return '/cpu:0'
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
  def input_keys(self):
    return self._input_keys

  @property
  def max_sequence_length(self):
    return self._max_sequence_length

  def _SequenceEmbLookup(self, dense_ids: tf.Tensor,
                         partition_strategy: str) -> tf.Tensor:
    """Sequence embedding lookup.

    Note that we do not support padding ids in sequence embeddings.

    Args:
      dense_ids: An int Tensor of shape [batch, sequence].
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      A float32 activations Tensor of shape
      [batch, max_sequence_length, embedding_dim].
    """
    p = self.params
    embs = tf.nn.embedding_lookup(
        self.theta.wm,
        tf.reshape(dense_ids, [-1]),
        partition_strategy=partition_strategy)
    out_shape = tf.concat([tf.shape(dense_ids), [p.embedding_dim]], 0)
    return tf.reshape(embs, out_shape)

  def _CombinerEmbLookup(self, sparse_ids: tf.SparseTensor,
                         partition_strategy: str) -> tf.Tensor:
    """Combiner embedding lookup.

    Args:
      sparse_ids: An int SparseTensor of shape [batch, ...].
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      A float32 activations Tensor of shape [batch, 1, embedding_dim].
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

  def CpuEmbLookup(self, ids_map: py_utils.NestedMap,
                   partition_strategy: str) -> py_utils.NestedMap:
    """CPU evaluation embedding lookup for dense tensors.

    Args:
      ids_map: A NestedMap of nested `input_key` string -> [batch, sequence]
        int Tensor. For sequence embeddings, -1 is used as a padding id.
        Non-sequence embeddings do not support padded ids.
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      An activations NestedMap of nested string -> float32 Tensor.
      For non-sequence embeddings: [batch, 1, embedding_dim]
      For sequence embeddings: [batch, max_sequence_length, embedding_dim]
    """
    if self.max_sequence_length > 0:
      # "Sequence embedding", no combiner case
      return ids_map.Transform(
          lambda ids: self._SequenceEmbLookup(ids, partition_strategy))
    else:
      # Non-"Sequence embedding", combiner case
      def _Lookup(ids):
        # Dense to sparse.
        dense_shape = tf.shape(ids, out_type=tf.int64)
        sample_indices = tf.cast(tf.where(tf.not_equal(ids, -1)), tf.int64)
        embedding_indices = tf.cast(tf.gather_nd(ids, sample_indices), tf.int64)
        # [?, embedding_dim]
        sparse_ids = tf.SparseTensor(
            indices=sample_indices,
            values=embedding_indices,
            dense_shape=dense_shape)
        return self._CombinerEmbLookup(sparse_ids, partition_strategy)

      return ids_map.Transform(_Lookup)

  def CpuEmbLookupSparse(self, ids_map: py_utils.NestedMap,
                         partition_strategy: str) -> py_utils.NestedMap:
    """CPU evaluation embedding lookup for SparseTensors.

    Args:
      ids_map: A NestedMap of nested `input_key` string -> [batch, ...] int
        SparseTensor.
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      An activations NestedMap of nested string -> float32 Tensor.
      For non-sequence embeddings: [batch, 1, embedding_dim]
      For sequence embeddings: [batch, max_sequence_length, embedding_dim]
    """
    if self.max_sequence_length > 0:
      # "Sequence embedding", no combiner case
      def _Lookup(ids):
        # Sparse to dense.
        dense_ids = tf.sparse.to_dense(ids, default_value=-1)
        return self._SequenceEmbLookup(dense_ids, partition_strategy)

      return ids_map.Transform(_Lookup)
    else:
      # Non-"Sequence embedding", combiner case
      return ids_map.Transform(
          lambda ids: self._CombinerEmbLookup(ids, partition_strategy))


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
    p.Define('num_tpu_hosts', 0, 'Total number of TPU hosts.')
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
    assert p.gradient_multiplier_schedule
    assert p.partition_strategy in ['mod', 'div']

    if p.num_tpu_hosts > 0:
      for table_params in p.tables:
        num_tpu_hosts = table_params.num_tpu_hosts
        if num_tpu_hosts > 0 and num_tpu_hosts != p.num_tpu_hosts:
          raise ValueError(
              f'num_tpu_hosts mismatch: {num_tpu_hosts} vs {p.num_tpu_hosts}')
        table_params.num_tpu_hosts = p.num_tpu_hosts
    else:
      num_tpu_hosts = p.tables[0].num_tpu_hosts
      assert all([t.num_tpu_hosts == num_tpu_hosts for t in p.tables])

    # Stop if a table has no optimizer related parameters and the layer also
    # has no optimizer parameters
    for param_name in ['optimizer', 'learning_rate', 'lr_schedule']:
      table_param_missing = any(
          table_params.Get(param_name) is None for table_params in p.tables)
      if not p.Get(param_name) and table_param_missing:
        raise ValueError(
            f'A table is missing {param_name} parameters, and no layer-level '
            f'{param_name} parameters were given.')
      elif table_param_missing:
        for table_params in p.tables:
          if table_params.Get(param_name) is None:
            value = p.Get(param_name)
            if isinstance(value, hyperparams.Params):
              value = value.Copy()  # Avoid mutating the original copy.
            table_params.Set(**{param_name: value})

    self.CreateChildren('tables', p.tables)
    self.CreateChild('gradient_multiplier_schedule',
                     p.gradient_multiplier_schedule)
    self._tpu_embedding_collection = TpuEmbeddingCollection.Get()

    # Save embedding feature names in the collection.
    feature_names = set()
    for table in self.tables:
      for feature in table.input_keys:
        if feature in feature_names:
          raise ValueError(f'Input key {feature} was used by multiple tables.')
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

    # At the feature level, track which are associated
    # with "sequence embeddings".
    self._sequence_features = {}

    if _IsTpuTraining(p):
      num_cores = self.cluster.params.worker.tpus_per_replica
      global_batch_size = (
          self.params.batch_size * self.cluster.num_splits_per_client)
      table_to_config_dict = {}
      feature_to_config_dict = {}
      for table in self.tables:
        table_to_config_dict[table.table_name] = table.table_config
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
        self._tpu_embedding_collection.SetGradientMultiplierSchedule(
            self.gradient_multiplier_schedule)

  def _TpuEmbLookup(self, ids_map: py_utils.NestedMap) -> py_utils.NestedMap:
    """TPU Embedding lookup."""
    task_call_scope = py_utils.GetTaskCallScope()
    activations = self._tpu_embedding_collection.AddActivations(task_call_scope)

    ret = py_utils.NestedMap()
    for k, v in activations.items():
      if ids_map.Get(k) is not None:
        if k in self._sequence_features:
          ret.Set(k, v)
        else:
          # Non-sequence embeddings, we fill the "time" dimension with 1.
          with tf.name_scope(k):
            ret.Set(k, tf.expand_dims(v, axis=[1]))
    return ret

  def EmbLookup(self, ids_map: py_utils.NestedMap) -> py_utils.NestedMap:
    """Looks up embedding vectors for each entry in dense Tensor ids_map.

    Since the TPUEmbedding is monolothic, and consulted once per
    FProp/BProp, we must centralize the lookup. Thus, for multiple
    features, we contain them into a single-lookup rather than allowing
    the caller to call Lookup multiple times.

    Args:
      ids_map: A NestedMap of nested `input_key` string -> [batch, sequence] int
        Tensor.
        For sequence embeddings, -1 is used as a padding id. Non-sequence
        embeddings do not support padded ids.

    Returns:
      Activations NestedMap of nested string ->
      For non-sequence embeddings:  [batch, 1, embedding_dim],
      For sequence embeddings: [batch, max_sequence_length, embedding_dim]
      float32 Tensor.
    """
    assert isinstance(ids_map, py_utils.NestedMap)
    p = self.params

    def CpuEmbLookup(ids_map):
      """CPU evaluation embedding lookup."""
      rets = py_utils.NestedMap()
      for table in self.tables:
        table_id_map = py_utils.NestedMap()
        for key in table.input_keys:
          if ids_map.Get(key) is not None:
            table_id_map.Set(key, ids_map.GetItem(key))
        table_rets = table.CpuEmbLookup(table_id_map, p.partition_strategy)
        # Merge table_rets with rets
        for key in table.input_keys:
          if ids_map.Get(key) is not None:
            rets.Set(key, table_rets.GetItem(key))
      return rets

    if _IsTpuTraining(p):
      return self._TpuEmbLookup(ids_map)
    else:
      return CpuEmbLookup(ids_map)

  def EmbLookupSparse(self, ids_map: py_utils.NestedMap) -> py_utils.NestedMap:
    """Looks up embedding vectors for each entry in SparseTensor ids_map.

    Since the TPUEmbedding is monolothic, and consulted once per
    FProp/BProp, we must centralize the lookup. Thus, for multiple
    features, we contain them into a single-lookup rather than allowing
    the caller to call Lookup multiple times.

    Args:
      ids_map: A NestedMap of nested `input_key` string -> [batch, ...] int
      SparseTensor.

    Returns:
      Activations NestedMap of nested string ->
      For non-sequence embeddings:  [batch, 1, embedding_dim],
      For sequence embeddings: [batch, max_sequence_length, embedding_dim]
      float32 Tensor.
    """
    assert isinstance(ids_map, py_utils.NestedMap)
    p = self.params

    def CpuEmbLookupSparse(ids_map):
      """CPU evaluation embedding lookup."""
      rets = py_utils.NestedMap()
      for table in self.tables:
        table_id_map = py_utils.NestedMap()
        for key in table.input_keys:
          if ids_map.Get(key) is not None:
            table_id_map.Set(key, ids_map.GetItem(key))
        table_rets = table.CpuEmbLookupSparse(table_id_map,
                                              p.partition_strategy)
        # Merge table_rets with rets
        for key in table.input_keys:
          if ids_map.Get(key) is not None:
            rets.Set(key, table_rets.GetItem(key))
      return rets

    if _IsTpuTraining(p):
      return self._TpuEmbLookup(ids_map)
    else:
      return CpuEmbLookupSparse(ids_map)
