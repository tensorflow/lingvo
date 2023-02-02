# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Base class for all tasks.

A model solely consists of the network, while a task combines one or several
models with one or several learners/optimizers.
"""

import abc
import itertools
from typing import Sequence

import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import base_model
from lingvo.jax import learners as learners_lib
from lingvo.jax import optimizers
from lingvo.jax import py_utils
from lingvo.jax import train_states
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap
NestedJTensor = base_layer.NestedJTensor
InstantiableParams = py_utils.InstantiableParams
PartitionSpec = jax.sharding.PartitionSpec
TrainState = train_states.TrainState


class BaseTask(metaclass=abc.ABCMeta):
  """Abstract base class for all tasks."""

  def __init__(self, params: InstantiableParams) -> None:
    assert params.name, ('Task params for %s must have a "name"' %
                         self.__class__.__name__)
    self._params = params.Copy()

  @property
  def params(self) -> base_layer.BaseLayerParamsT:
    """Returns the params upon which this layer is built."""
    return self._params


class SingleTask(BaseTask):
  """A JAX task."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint:disable=invalid-name
    """Task parameters."""
    p = InstantiableParams(cls)
    p.Define('name', '',
             'Name of this task object, must be a valid identifier.')
    p.Define('model', None,
             'The underlying JAX model encapsulating all the layers.')
    p.Define('train', py_utils.Params(),
             'Params to control how this task should be trained.')
    p.Define('metrics', None, 'How metrics are computed.')
    tp = p.train
    tp.Define('learner', learners_lib.Learner.Params(),
              'One or a list of learners.')
    tp.Define('num_train_steps', 1e7,
              'Maximum number of training steps to run.')
    # TODO(bf-jax): Add an option to perform this wrt. a time duration.
    tp.Define(
        'save_interval_steps', 5000,
        'How frequently to save a model checkpoint in terms of the number of '
        'training steps.')
    tp.Define(
        'save_keep_interval_duration', '12h',
        'How frequently to keep a saved model checkpoint as a duration string '
        'such as `1h` for one hour or `90m` for 90 minutes. This is performed '
        'in addition to keeping the most recent `max_to_keep` checkpoint '
        'files.')
    tp.Define('save_max_to_keep', 10,
              'The maximum number of recent checkpoints to keep.')
    tp.Define(
        'summary_interval_steps', 100,
        'How frequently to generate summaries in terms of the number of '
        'training steps.')
    tp.Define(
        'norm_summary_interval_steps', 500,
        'How frequently to generate expensive summaries computing the norms '
        'of variables in terms of the number of training steps.')
    tp.Define(
        'eval_interval_steps', 100,
        'How frequently to evaluate the model on the evaluation splits in '
        'terms of the number of training steps.')
    tp.Define(
        'inputs_split_mapping', None, 'The PartitionSpec for inputs'
        'such as inputs, labels, targets, paddings, num words etc. This is only'
        'relevant for SPMD sharded models. By default it is None, which means'
        'all the inputs are replicated. For sharding inputs, this is a '
        '`NestedMap` with keys `map_1d`, `map_2d`, ..., etc.,'
        'which specifies how to shard the inputs of that dimension.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self._params

    assert p.train.learner is not None
    # TODO(yonghui): implement multiple learners.
    assert not isinstance(p.train.learner, (tuple, list))
    learner_params = [p.train.learner]
    learner_params = NestedMap.FromNestedDict(learner_params)
    uid = itertools.count()

    def instantiate(p: InstantiableParams) -> base_layer.BaseLayerT:
      p = p.Copy()
      p.name = 'learner_%d' % next(uid)
      return p.Instantiate()

    self._learners = NestedMap(sub=learner_params).Transform(instantiate).sub

    assert p.model is not None
    self._model = p.model.Instantiate()

  @property
  def learners(self) -> Sequence[learners_lib.Learner]:
    return self._learners

  @property
  def model(self) -> base_model.BaseModel:
    return self._model

  @property
  def has_ema_decay(self):
    return bool(self.learners[0].params.optimizer and
                self.learners[0].params.optimizer.ema_decay > 0)

  def create_train_state_unpadded_shapes(self,
                                         var_weight_params,
                                         discard_opt_states=False
                                        ) -> TrainState:
    """Creates shapes for all variables used in training without padding...

    due to uneven sharding.

    Args:
      var_weight_params: a nested map of variable params for all the forward
        variables.
      discard_opt_states: bool, When true, optimizer slot variables are skipped.


    Returns:
      A TrainState contains shapes for all the forward and backward variables.
    """
    def _get_shape(var_param):
      return tuple(var_param.repeat_prefix or ()) + tuple(var_param.shape)

    var_shapes = jax.tree_map(_get_shape, var_weight_params)
    if discard_opt_states:
      opt_shapes = {}
    else:
      grad_txs = [x.get_grad_tx(var_weight_params) for x in self.learners]
      opt_var_weight_params = []
      for grad_tx in grad_txs:
        assert isinstance(grad_tx, optimizers.ShardedGradientTransformation)
        opt_var_weight_params.append(
            grad_tx.init_partition_spec(var_weight_params))
      opt_shapes = jax.tree_map(_get_shape, opt_var_weight_params)
    step_shapes = ()
    return TrainState(
        step=step_shapes, mdl_vars=var_shapes, opt_states=opt_shapes)

  def create_train_state_partition_specs(self,
                                         var_weight_params,
                                         discard_opt_states=False):
    """Creates partition specs for all variables used in training.

    Args:
      var_weight_params: a nested map of variable params for all the forward
        variables.
      discard_opt_states: bool, When true, optimizer slot variables are skipped.

    Returns:
      A TrainState contains PartitionSpecs for all the forward and/or backward
        variables depending on the value of discard_opt_states, or None.
    """
    p = self.params
    device_mesh = p.model.device_mesh
    if device_mesh is None:
      return None
    step_partition_spec = PartitionSpec()
    var_partition_specs = base_layer.var_partition_specs(
        var_weight_params,
        device_mesh=device_mesh,
        device_axis_names=p.model.mesh_axis_names)
    if discard_opt_states:
      opt_var_partition_specs = {}
    else:
      grad_txs = [x.get_grad_tx(var_weight_params) for x in self.learners]
      opt_var_weight_params = []
      for grad_tx in grad_txs:
        assert isinstance(grad_tx, optimizers.ShardedGradientTransformation)
        opt_var_weight_params.append(
            grad_tx.init_partition_spec(var_weight_params))

      opt_var_partition_specs = base_layer.var_partition_specs(
          opt_var_weight_params,
          device_mesh=device_mesh,
          device_axis_names=p.model.mesh_axis_names)
    return TrainState(
        step=step_partition_spec,
        mdl_vars=var_partition_specs,
        opt_states=opt_var_partition_specs)

  def create_train_state(self,
                         mdl_vars: NestedJTensor,
                         var_weight_params: NestedJTensor,
                         discard_opt_states=False) -> TrainState:
    """Creates train states that holds all the forward/backward variables.

    Args:
      mdl_vars: A nested structure of model vars to create TrainState for.
        'mdl_vars' can be a sub-set of self.vars.
      var_weight_params: WeightParams for each of the variable in mdl_vars.
        var_weight_params must be of the same structure as mdl_vars. Each model
        weight variable is associated with some WeightParams which contains all
        the meta information about the weight variable.
      discard_opt_states: bool, When true, optimizer slot variables are skipped.


    Returns:
      a TrainState.
    """
    # Make a private copy of mdl_vars and var_weight_params structures that are
    # not shared with the caller.
    mdl_vars = tf.nest.map_structure(lambda x: x, mdl_vars)
    var_weight_params = tf.nest.map_structure(lambda x: x, var_weight_params)
    if discard_opt_states:
      opt_states = {}
    else:
      grad_txs = [x.get_grad_tx(var_weight_params) for x in self.learners]
      tf.nest.assert_same_structure(mdl_vars, var_weight_params)
      opt_states = [x.init(mdl_vars) for x in grad_txs]
    return TrainState(
        # The global step for the model.
        step=jnp.array(0, dtype=jnp.uint32),
        mdl_vars=mdl_vars,
        opt_states=opt_states)
