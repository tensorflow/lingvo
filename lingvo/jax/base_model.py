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
"""Base class for all models.

The model solely consists of the network, while the task combines one or several
models with one or several learners/optimizers.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
from jax import numpy as jnp
from lingvo.jax import base_input
from lingvo.jax import base_layer
from lingvo.jax import layers
from lingvo.jax import metric_utils
from lingvo.jax import py_utils
from lingvo.jax import train_states

NestedMap = py_utils.NestedMap
JTensor = base_layer.JTensor
InstantiableParams = py_utils.InstantiableParams
Predictions = Union[JTensor, NestedMap, Dict[str, Any]]
Metrics = Dict[str, Tuple[JTensor, JTensor]]
TrainState = train_states.TrainState


def _compute_xent_loss_helper(
    predictions: NestedMap, input_batch: NestedMap,
    return_predictions: bool) -> Tuple[Metrics, Dict[str, Any]]:
  """Helper for computing the xent loss for Language model and Sequence model.

  Args:
    predictions: A `.NestedMap` containing the keys `per_example_argmax`,
      `total_loss`, `avg_xent`, `aux_loss`, `total_weight` which corresponds to
      the output of the Softmax layer.
    input_batch: A `.NestedMap` object containing input tensors which contains
      the keys `labels` and `weights` which corresponds to the labels and the
      `weights` for each token in the sequence.
    return_predictions: Whether to return predictions, which can be more
      expensive.

  Returns:
    - A dict or NestedMap containing str keys and (metric, weight) pairs as
      values, where one of the entries is expected to correspond to the loss.
    - A dict containing arbitrary tensors describing something about each
      training example, where the first dimension of each tensor is the batch
      index. The base class just returns an empty dict.
  """
  if 'tgt' in input_batch:
    labels = input_batch.tgt.labels
    if 'paddings' in input_batch.tgt:
      weights = 1.0 - input_batch.tgt.paddings
    else:
      weights = jnp.not_equal(input_batch.tgt.segment_ids, 0)
    weights = weights.astype(labels.dtype)
  else:
    labels = input_batch.labels
    weights = input_batch.weights
  predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
  # To improve aggregation stability for large topologies we compute stats in
  # f32 as well. Since we already do f32 logits, this is little overhead.
  weights = weights.astype(jnp.float32)
  num_preds = predictions.total_weight.astype(jnp.float32)
  mean_acc = jnp.sum(
      (labels == predicted_labels).astype(jnp.float32) * weights) / jnp.maximum(
          num_preds, 1)
  metric_weight = jnp.array(num_preds, predictions.avg_xent.dtype)

  if hasattr(predictions, 'avg_xent_weight'):
    avg_xent_weight = predictions.avg_xent_weight
  else:
    avg_xent_weight = metric_weight

  assert mean_acc.dtype == jnp.float32, mean_acc.dtype
  total_loss = predictions.total_loss.astype(jnp.float32)
  avg_xent = predictions.avg_xent.astype(jnp.float32)
  aux_loss = predictions.aux_loss.astype(jnp.float32)
  metrics = NestedMap(
      total_loss=(total_loss, metric_weight),
      avg_xent=(avg_xent, avg_xent_weight),
      aux_loss=(aux_loss, jnp.array(1.0, aux_loss.dtype)),
      log_pplx=(predictions.avg_xent, avg_xent_weight),
      fraction_of_correct_next_step_preds=(mean_acc, metric_weight),
      num_predictions=(num_preds, jnp.array(1.0, num_preds.dtype)),
  )
  per_example_output = NestedMap()
  if return_predictions:
    per_example_output = predictions
  return metrics, per_example_output


def greedy_decode(extend_step_fn: Callable[[NestedMap, JTensor],
                                           Tuple[NestedMap, JTensor]],
                  decoder_state: NestedMap,
                  target_ids: JTensor,
                  target_paddings: JTensor,
                  seq_len: int,
                  max_decode_steps: Optional[int] = None,
                  prefix_lengths: Optional[JTensor] = None,
                  eos_id: Optional[int] = None) -> NestedMap:
  """Greedy decode the input batch.

  Args:
    extend_step_fn: A function that takes in `states` and the decoded sequence
      at the current time step (with shape [B] or [B, P] where B corresponds to
      the batch size and P corresponds to a possible prefix) and returns a tuple
      of (`NestedMap`, `JTensor`), where the first `NestedMap` corresponds to
      the `new_states` and the second `JTensor` corresponds to the logits of the
      next step.
    decoder_state: The initialized cache for autoregressive cached decoding.
    target_ids: The token ids that correspond to the target sequence.
    target_paddings: The paddings corresponding to the target sequence, with a 1
      denoting padding token and 0 denoting non-padding tokens.
    seq_len: The output sequence length to decode to.
    max_decode_steps: Python int or None, the max decode step to run after the
      prefix (if any). Since the prefixes might be of unequal lengths, this
      value is not equivalent with `seq_len` above. When None, decode steps is
      only limited by `seq_len` above.
    prefix_lengths: Optional argument supplying a prefix sizes to initialize the
      model to decode from a certain target prefix for each position in the
      batch. This can either be None or a JTensor of shape [batch] signifying
      the prefix length for each sequence in the batch.
    eos_id: Optional EOS id which to terminate the decoding early.

  Returns:
    A NestedMap with `.prefix_lengths` (indicating the lengths of prefixes for
    each target sequence), `.output_ids` (matrix of int ids with the
    decoded output), `.decode_lengths` (vector of ints indicating the lengths
    of non-padding tokens in `.output_ids`, which includes the prefix), and
    `.logprobs` (the log probability of selected tokens, including the prefix,
    where a positive value of 1.0 is used to indicate padded positions).
  """
  if seq_len <= 0:
    raise ValueError('The sequence length for decoding must be > 0, '
                     f'current value = {seq_len}.')
  max_decode_steps = max_decode_steps or seq_len
  batch_size = target_ids.shape[0]

  # If prefix length is not specified set it to 0.
  if prefix_lengths is None:
    prefix_lengths = jnp.zeros([batch_size], dtype=jnp.int32)

  output_ids = jnp.zeros(shape=(batch_size, seq_len), dtype=jnp.int32)
  output_ids = output_ids.at[:, 0].set(target_ids[:, 0])

  val = NestedMap()
  val.state = decoder_state
  val.step = 0
  val.output_ids = output_ids
  # Shape [batch_size], whether each row has terminated and should stop.
  val.done = jnp.zeros(shape=batch_size, dtype=jnp.bool_)
  val.decode_lengths = jnp.ones_like(prefix_lengths) * seq_len
  # We use a positive value of 1.0 to indicate blank or padded positions.
  val.logprobs = jnp.ones_like(output_ids, dtype=jnp.float32)

  def cond_func(val):
    """Whether the while loop should continue."""
    # We continue the greedy search iff both:
    #   (1) We have yet to exceed the max steps set by p.decoder.seqlen, AND;
    #   (2) At least one row in the batch has not terminated.
    length_ok = val.step < seq_len - 1
    all_rows_done = jnp.all(val.done)
    return jnp.logical_and(length_ok, jnp.logical_not(all_rows_done))

  def loop_body(val):
    """From ids at `step`, update output ids at `step + 1`."""
    step = val.step
    decoder_state, logits = extend_step_fn(val.state, val.output_ids[:, step])
    logprobs = jax.nn.log_softmax(logits.astype(jnp.float32))
    val.state = decoder_state
    # When step becomes prefix_length - 1, the new output has index beyond
    # the known prefix.
    # If prefix_length is 0, the condition is always False, so we take the
    # decoded output rather than the prefix.
    new_ids = jnp.where(step < prefix_lengths - 1, target_ids[:, step + 1],
                        jnp.argmax(logits, axis=1))
    prev_done = val.done
    new_ids = jnp.where(prev_done, jnp.zeros_like(new_ids), new_ids)
    if eos_id is not None:
      val.done = jnp.logical_or(prev_done, jnp.equal(new_ids, eos_id))
    max_decoding_steps_reached = (jnp.ones_like(prefix_lengths) * (step + 2) -
                                  prefix_lengths) >= max_decode_steps
    val.done = jnp.logical_or(val.done, max_decoding_steps_reached)
    done_at_this_step = jnp.logical_and(jnp.logical_not(prev_done), val.done)
    val.decode_lengths = jnp.where(
        done_at_this_step,
        jnp.ones_like(val.decode_lengths) * (step + 2), val.decode_lengths)
    val.output_ids = val.output_ids.at[:, step + 1].set(new_ids)
    logprobs_at_new_ids = logprobs.at[jnp.arange(batch_size), new_ids].get()
    logprobs_at_new_ids = jnp.where(prev_done,
                                    jnp.ones_like(logprobs_at_new_ids),
                                    logprobs_at_new_ids)
    val.logprobs = val.logprobs.at[:, step + 1].set(logprobs_at_new_ids)
    val.step += 1
    return val

  result = jax.lax.while_loop(cond_func, loop_body, val)
  result.prefix_lengths = prefix_lengths
  result.original_lengths = jnp.sum(
      1.0 - target_paddings, axis=1).astype(jnp.int32)

  prefix_ids = target_ids
  # We manually pad out the ids not belonging to the prefix because some
  # tokenizers tested do not always obey the lengths arg.
  indices = jnp.tile(jnp.arange(prefix_ids.shape[1]), (prefix_ids.shape[0], 1))
  prefix_lengths_2d = jnp.tile(prefix_lengths[:, None],
                               (1, prefix_ids.shape[1]))
  prefix_ids = jnp.where(indices < prefix_lengths_2d, prefix_ids,
                         jnp.zeros_like(prefix_ids))
  result.prefix_ids = prefix_ids

  del result.state, result.step, result.done
  return result


class BaseModel(base_layer.BaseLayer):
  """An API that every model should be derived from."""

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`.

    This method must be defined in a concrete derived class.

    The output can be in the form of probablistic distributions, e.g., softmax
    logits for discrete outputs, mixture of logistics for continuous values, or
    regression values.

    For training/evaluation, the output will be used for computing loss and
    gradient updates, including comparing predicted distributions between
    teacher and student for distillation. During inference the output can be
    used to compute final outputs, perhaps with sampling.

    Args:
      input_batch: A `.NestedMap` object containing input tensors.

    Returns:
      Predictions, either a single Tensor, a `.NestedMap`, or a namedtuple.
    """
    raise NotImplementedError('Abstract method')

  def compute_loss(self, predictions: Union[JTensor, NestedMap],
                   input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    This method must be defined in a concrete derived class.

    Args:
      predictions: The output of `compute_predictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (metric, weight) pairs as
        values, where one of the entries is expected to corresponds to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    raise NotImplementedError('Abstract method')

  def fprop(self, input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
    """Forward propagation through one tower of the model.

    Args:
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      (dict, dict):

      - A dict containing str keys and (metric, weight) pairs as values, where
        one of the keys is expected to be 'loss'.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    with py_utils.AuxLossContext():
      predictions = self.compute_predictions(input_batch)
      return self.compute_loss(predictions, input_batch)

  def decode(self, input_batch: NestedMap) -> Tuple[NestedMap, NestedMap]:
    """Decodes input_batch.

    Args:
      input_batch: The input batch. A `NestedMap` of tensors. Or, if input batch
        spiltting is used, a list of `NestedMap`, one for each split.

    Returns:
      - metrics, a NestedMap containing str keys and (metric, weight) pairs for
        the current batch (a tuple of two scalars).
      - results, a `.NestedMap` as decoder output.
    """
    raise NotImplementedError('Abstract method')

  def process_decode_out(
      self, input_obj: base_input.BaseInput,
      decode_out: NestedMap) -> Tuple[NestedMap, Sequence[Tuple[str, Any]]]:
    """Processes one batch of decoded outputs.

    Args:
      input_obj: The input object where a tokenizer is accessible.
      decode_out: The output from decode(). May have an extra leading axis.

    Returns:
      - metrics, a NestedMap containing str keys and (metric, weight) pairs for
        the current batch (a tuple of two scalars).
      - A list of tuples where each element corresponds to a row in the batch.
        Each tuple is a key value pair.
    """
    raise NotImplementedError('Abstract method')


class ClassificationMLPModel(BaseModel):
  """Language Model task with a simple MLP model."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()

    p.Define('mlp_tpl', layers.linears.MLPBlock.Params(),
             'MLP model parameters.')
    p.Define('softmax_tpl', layers.SingleShardSharedEmbeddingSoftmax.Params(),
             'Input softmax embedding lookup layer.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    self.create_children('mlp_layers', p.mlp_tpl.Copy())
    self.create_child('softmax', p.softmax_tpl.Copy())

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:

    input_emb = self.softmax.emb_lookup(input_batch.ids)

    output = self.mlp_layers.fprop(input_emb)
    predictions = self.softmax.fprop(
        inputs=output,
        class_weights=input_batch.weights[:, :, jnp.newaxis],
        class_ids=input_batch.ids[:, :, jnp.newaxis])
    return predictions

  def compute_loss(self, predictions: NestedMap,  # pytype: disable=signature-mismatch  # jax-ndarray
                   input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
    labels = input_batch.labels
    weights = input_batch.weights
    class_weights = weights[:, :, jnp.newaxis]
    num_preds = jnp.sum(class_weights)
    predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
    mean_acc = jnp.sum(
        (labels == predicted_labels) * weights) / jnp.maximum(num_preds, 1)
    metrics = NestedMap(total_loss=(mean_acc, mean_acc),)

    return metrics, NestedMap()


class LanguageModel(BaseModel):
  """Language Model base task."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('lm', layers.TransformerLm.Params(), 'LM layer.')
    p.Define(
        'return_predictions', False, 'Whether to return predictions during'
        'eval. Returning predictions is more expensive, but may be useful'
        'for debugging.')

    greedy_search_p = py_utils.Params()
    greedy_search_p.Define('seqlen', 0, 'Maximum output sequence length.')
    greedy_search_p.Define(
        'min_prefix_len', 5,
        'Minimum number of tokens picked to be used as decoding prefix.')
    greedy_search_p.Define(
        'eos_id', 2,
        'The id of EOS token indicating the termination of greedy search.')
    greedy_search_p.Define(
        'max_decode_steps', None,
        'If not None, the max decode steps for each example. If None, this '
        'is set to `seqlen`, which contains prefix.')
    p.Define('decoder', greedy_search_p, 'Decoder param.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    # Construct the model.
    lm_p = p.lm.Copy()
    self.create_child('lm', lm_p)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`."""
    p = self.params
    if 'tgt' in input_batch:
      input_batch = input_batch.tgt

    if 'paddings' in input_batch:
      paddings = input_batch.paddings
    else:
      paddings = jnp.equal(input_batch.segment_ids, 0).astype(self.fprop_dtype)

    if 'weights' in input_batch:
      weights = input_batch.weights
    else:
      weights = 1.0 - paddings
      weights = weights.astype(self.fprop_dtype)
      input_batch.weights = weights

    inputs = input_batch.ids
    labels = NestedMap(class_ids=input_batch.labels, class_weights=weights)
    if p.lm.packed_input:
      packed_input_kwargs = {
          'segment_ids': input_batch.segment_ids,
          'segment_pos': input_batch.segment_pos,
      }
    else:
      packed_input_kwargs = {}
    return self.lm.fprop(
        inputs=inputs,
        paddings=paddings,
        labels=labels,
        **packed_input_kwargs)

  def compute_loss(self, predictions: NestedMap,  # pytype: disable=signature-mismatch  # jax-ndarray
                   input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    Args:
      predictions: The output of `compute_predictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (metric, weight) pairs as
        values, where one of the entries is expected to corresponds to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    return _compute_xent_loss_helper(predictions, input_batch,
                                     self.params.return_predictions)

  def decode(self, input_batch: NestedMap) -> Tuple[NestedMap, NestedMap]:
    """Greedy decodes the input_batch.

    Args:
      input_batch: The input batch, with fields like `.ids`.

    Returns:
      - metrics, a NestedMap containing str keys and (metrics, weight) pairs.
      - A NestedMap like `input_batch`, with `.prefix_lengths` (vector of
        randomly generated ints indicating the lengths of prefixes for each
        row), and `.output_ids` (matrix of int ids with the decoded output).
    """
    p = self.params
    if p.decoder.seqlen <= 0:
      raise ValueError('Must set p.decoder.seqlen > 0, current value = '
                       f'{p.decoder.seqlen}')
    batch_size = input_batch.ids.shape[0]
    maxval = jnp.sum(1 - input_batch.paddings, axis=1).astype(jnp.int32)
    minval = jnp.minimum(maxval, p.decoder.min_prefix_len)
    prefix_lengths = jax.random.randint(base_layer.next_prng_key(),
                                        [batch_size], minval, maxval + 1,
                                        input_batch.ids.dtype)
    decoder_state = self.lm.init_states(
        target_batch_size=batch_size,
        target_max_length=p.decoder.seqlen)

    global_step = base_layer.cur_global_step()

    lm_theta = self.lm.local_theta()
    def extend_step_fn(states, ids):
      with base_layer.JaxContext.new_context(
          prng_key=base_layer.next_prng_key(),
          global_step=global_step) as jax_context:
        jax_context.bind(self.lm, self.lm.vars_to_flax_vars(lm_theta),
                         [base_layer.SCOPE_AUX_LOSS])
        new_states, xent = self.lm.extend_step(states, ids)
        return new_states, xent.logits

    result = greedy_decode(
        extend_step_fn,
        decoder_state,
        input_batch.ids,
        input_batch.paddings,
        p.decoder.seqlen,
        max_decode_steps=p.decoder.max_decode_steps,
        prefix_lengths=prefix_lengths,
        eos_id=p.decoder.eos_id)
    result.update(input_batch)

    metrics = NestedMap(
        num_decoded=(jnp.array(0.0, jnp.float32),
                     jnp.array(batch_size, jnp.float32)))
    return metrics, result

  def process_decode_out(
      self, input_obj: base_input.BaseInput,
      decode_out: NestedMap) -> Tuple[NestedMap, Sequence[Tuple[str, Any]]]:
    """Processes one batch of decoded outputs.

    Args:
      input_obj: The input object where a tokenizer is accessible.
      decode_out: The output from decode(). May have an extra leading axis.

    Returns:
      - metrics, a NestedMap containing str keys and (metric, weight) pairs for
        the current batch (a tuple of two scalars).
      - A list of dict where each entry corresponds to a row in the batch. The
        keys should be unique across the entire decode dataset.
    """
    decoded_strs = input_obj.ids_to_strings(decode_out.output_ids,
                                            decode_out.decode_lengths)
    original_strs = input_obj.ids_to_strings(decode_out.ids,
                                             decode_out.original_lengths)
    prefix_strs = input_obj.ids_to_strings(decode_out.prefix_ids,
                                           decode_out.prefix_lengths)
    ret = list()
    for idx, decoded_str in enumerate(decoded_strs):
      ret.append((prefix_strs[idx], {
          'prefix': prefix_strs[idx],
          'decoded': decoded_str,
          'original': original_strs[idx],
          'ids': decode_out.output_ids[idx],
          'logprobs': decode_out.logprobs[idx],
          'prefix_length': decode_out.prefix_lengths[idx],
          'decode_length': decode_out.decode_lengths[idx],
      }))
    decoded_lengths = jnp.average(decode_out.decode_lengths).astype(jnp.float32)
    metrics = NestedMap(
        decoded_length=(decoded_lengths, jnp.array(1.0, jnp.float32)))
    return metrics, ret


class SequenceModel(BaseModel):
  """Sequence Model base task."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('model', layers.TransformerEncoderDecoder.Params(),
             'Sequence model layer for this task.')
    p.Define(
        'return_predictions', False, 'Whether to return predictions during'
        'eval. Returning predictions is more expensive, but may be useful'
        'for debugging.')
    decoder_p = py_utils.Params()
    decoder_p.Define('seqlen', 0, 'Maximum output sequence length.')
    decoder_p.Define(
        'eos_id', 2,
        'The id of EOS token indicating the termination of decoding.')
    p.Define('decoder', decoder_p, 'Decoder params.')
    p.Define(
        'label_smoothing_prob', 0.0,
        'If > 0.0, smooth out one-hot prob by spreading this amount of'
        ' prob mass to all other tokens.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    # Construct the model.
    model_p = p.model.Copy()
    self.create_child('model', model_p)

  def compute_predictions(self, input_batch):
    """Computes predictions for `input_batch`."""
    p = self.params
    if p.model.packed_input:
      packed_input_kwargs = {
          'input_segment_ids': input_batch.src.segment_ids,
          'input_segment_pos': input_batch.src.segment_pos,
          'target_segment_ids': input_batch.tgt.segment_ids,
          'target_segment_pos': input_batch.tgt.segment_pos,
      }
    else:
      packed_input_kwargs = {}

    labels = NestedMap(
        class_ids=input_batch.tgt.labels, class_weights=input_batch.tgt.weights)
    if p.label_smoothing_prob > 0.0:
      vocab_size = p.model.softmax_tpl.num_classes
      class_probabilities = jax.nn.one_hot(labels.class_ids, vocab_size)
      fill_prob = p.label_smoothing_prob / (vocab_size - 1)
      class_probabilities = (
          (1.0 - p.label_smoothing_prob) * class_probabilities + fill_prob *
          (1.0 - class_probabilities)).astype(self.fprop_dtype)
      labels.class_probabilities = class_probabilities

    return self.model.fprop(
        inputs=input_batch.src.ids,
        input_paddings=input_batch.src.paddings,
        targets=input_batch.tgt.ids,
        target_paddings=input_batch.tgt.paddings,
        labels=labels,
        **packed_input_kwargs)

  def compute_loss(self, predictions, input_batch):
    """Computes the loss and other metrics for the given predictions.

    Args:
      predictions: The output of `ComputePredictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (metric, weight) pairs as
        values, where one of the entries is expected to corresponds to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    return _compute_xent_loss_helper(predictions, input_batch.tgt,
                                     self.params.return_predictions)

  def decode(self, input_batch: NestedMap) -> Tuple[NestedMap, NestedMap]:
    """Decodes input_batch.

    Args:
      input_batch: The input batch, with a field `.src` and `.tgt` corresponding
        to source and target, which itself contains the `.ids` and `.paddings.`

    Returns:
      - metrics, a nestedmap of metrics.
      - results, a NestedMap like `input_batch`, with `.output_ids` (matrix of
        int ids with the decoded output) as well as the decoded length.
    """
    p = self.params
    model_theta = self.model.local_theta()
    if p.decoder.seqlen <= 0:
      raise ValueError('Must set p.decoder.seqlen > 0, current value = '
                       f'{p.decoder.seqlen}')
    batch_size = input_batch.tgt.ids.shape[0]
    decoder_state = self.model.init_states(
        inputs=input_batch.src.ids,
        input_paddings=input_batch.src.paddings,
        target_batch_size=batch_size,
        target_max_length=p.decoder.seqlen)

    global_step = base_layer.cur_global_step()

    def extend_step_fn(states, ids):
      with base_layer.JaxContext.new_context(
          prng_key=base_layer.next_prng_key(),
          global_step=global_step) as jax_context:
        jax_context.bind(self.model, self.model.vars_to_flax_vars(model_theta),
                         [base_layer.SCOPE_AUX_LOSS])
        new_states, xent = self.model.extend_step(states, ids)
        return new_states, xent.logits

    result = greedy_decode(
        extend_step_fn,
        decoder_state,
        input_batch.tgt.ids,
        input_batch.tgt.paddings,
        p.decoder.seqlen,
        eos_id=p.decoder.eos_id)
    # Prefix lengths are not needed for sequence model decoding.
    del result.prefix_lengths
    result.update(input_batch)
    metrics = NestedMap(
        num_decoded=(jnp.array(0.0, jnp.float32),
                     jnp.array(batch_size, jnp.float32)))
    return metrics, result

  def process_decode_out(
      self, input_obj: base_input.BaseInput,
      decode_out: NestedMap) -> Tuple[NestedMap, Sequence[Tuple[str, Any]]]:
    """Processes one batch of decoded outputs.

    Args:
      input_obj: The input object where a tokenizer is accessible.
      decode_out: The output from decode(). May have an extra leading axis.

    Returns:
      - metrics, a NestedMap containing str keys and (metric, weight) pairs for
        the current batch (a tuple of two scalars).
      - A list of dict where each entry corresponds to a row in the batch. The
        keys should be unique across the entire decode dataset.
    """
    decoded_strs = input_obj.ids_to_strings(
        decode_out.output_ids, decode_out.decode_lengths, key='tgt')
    source_lengths = jnp.sum(
        1.0 - decode_out.src.paddings, axis=1).astype(jnp.int32)
    source_strs = input_obj.ids_to_strings(
        decode_out.src.ids, source_lengths, key='src')
    target_strs = input_obj.ids_to_strings(
        decode_out.tgt.ids, decode_out.original_lengths, key='tgt')
    ret = list()
    for idx, decoded_str in enumerate(decoded_strs):
      ret.append((source_strs[idx], {
          'source': source_strs[idx],
          'decoded': decoded_str,
          'target': target_strs[idx],
          'ids': decode_out.output_ids[idx],
          'logprobs': decode_out.logprobs[idx],
          'decode_length': decode_out.decode_lengths[idx],
      }))
    decode_lengths = jnp.average(decode_out.decode_lengths).astype(jnp.float32)
    metrics = NestedMap(
        decode_length=(decode_lengths, jnp.array(1.0, jnp.float32)))
    return metrics, ret


class ClassificationModel(BaseModel):
  """Classification task for images and video."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('network', layers.ResNet.Params(),
             'The classifier network, which is ResNet-50 by default.')
    p.Define('softmax', layers.SingleShardFullSoftmax.Params(),
             'The softmax layer used for the classification.')
    p.Define(
        'input_field', 'image',
        'The input field which contains the image or video features to'
        'pass to the classification network.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    self.create_child('network', p.network)
    self.create_child('softmax', p.softmax)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`.

    Args:
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A NestedMap containing str keys and features, softmax output and the
        class weights as values.
    """
    p = self.params
    inputs = input_batch.Get(p.input_field)
    features = self.network.fprop(inputs)
    batch_size = inputs.shape[0]
    example_weights = jnp.ones([batch_size])
    if 'weight' in input_batch:
      example_weights = input_batch.weight
      if example_weights.shape != (batch_size,):
        raise ValueError(
            f'Shape of example weights should be ({batch_size},), but instead'
            f'is {example_weights.shape}')
    # Softmax expects weights to be of shape [..., 1].
    softmax_output = self.softmax.fprop(
        inputs=features,
        class_weights=example_weights[:, jnp.newaxis],
        class_probabilities=input_batch.label_probs)
    return NestedMap(
        features=features,
        softmax_output=softmax_output,
        example_weights=example_weights)

  def compute_loss(self, predictions: NestedMap,  # pytype: disable=signature-mismatch  # jax-ndarray
                   input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    Args:
      predictions: The output of `compute_predictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (metric, weight) pairs as
        values, where one of the entries is expected to correspond to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index. The base class just returns an empty dict.
    """
    avg_xent = predictions.softmax_output.avg_xent
    total_weight = predictions.softmax_output.total_weight
    metrics = NestedMap(
        avg_xent=(avg_xent, total_weight),
        num_predictions=(total_weight, jnp.array(1.0, total_weight.dtype)))
    # Compute top-1 and top-5 accuracy and add summary.
    acc1 = metric_utils.top_k_accuracy(
        1,
        predictions.softmax_output.logits,
        label_probs=input_batch.label_probs,
        weights=predictions.example_weights)
    acc5 = metric_utils.top_k_accuracy(
        5,
        predictions.softmax_output.logits,
        label_probs=input_batch.label_probs,
        weights=predictions.example_weights)
    metrics.update(
        accuracy=(acc1, predictions.softmax_output.total_weight),
        acc5=(acc5, predictions.softmax_output.total_weight),
        error=(1.0 - acc1, predictions.softmax_output.total_weight),
        error5=(1.0 - acc5, predictions.softmax_output.total_weight))
    # Add top-1 and top-5 accuracies to summaries.
    base_layer.add_summary('acc1', acc1)
    base_layer.add_summary('acc5', acc5)
    return metrics, {}


class BertModel(BaseModel):
  """Bert Model base task."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('lm', layers.TransformerLm.Params(), 'Bert lm layer.')
    p.Define(
        'label_smoothing_prob', 0.0,
        'If > 0.0, smooth out one-hot prob by spreading this amount of'
        ' prob mass to all other tokens.')
    p.Define('mask_token_id', 0, 'Mask token id')

    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    assert p.lm.masked_lm
    assert p.lm.packed_input

    self.create_child('lm', p.lm)

    mlm_augment_p = layers.MaskedLmDataAugmenter.Params()
    mlm_augment_p.vocab_size = p.lm.vocab_size
    mlm_augment_p.mask_token_id = p.mask_token_id
    self.create_child('mlm_augmenter', mlm_augment_p)

  def compute_predictions(self, input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`."""
    p = self.params
    assert p.lm.packed_input
    segment_ids = input_batch.segment_ids
    segment_pos = input_batch.segment_pos
    paddings = input_batch.paddings
    # Note that internal BertTransformer uses input_batch.ids instead.
    labels = input_batch.labels
    if 'masked_ids' in input_batch:
      # Input data already has masking done.
      augmented_labels = input_batch.masked_ids
      augmented_pos = input_batch.masked_pos
    else:
      augmented_labels, augmented_pos = self.mlm_augmenter.fprop(
          labels, paddings)

    if p.label_smoothing_prob > 0.0:
      class_probabilities = jax.nn.one_hot(labels, p.lm.vocab_size)
      fill_prob = p.label_smoothing_prob / (p.lm.vocab_size - 1)
      class_probabilities = (
          (1.0 - p.label_smoothing_prob) * class_probabilities + fill_prob *
          (1.0 - class_probabilities)).astype(self.fprop_dtype)

      # Only compute loss on masked pos.
      labels = NestedMap(
          class_probabilities=class_probabilities, class_weights=augmented_pos)
    else:
      # Only compute loss on masked pos.
      labels = NestedMap(class_ids=labels, class_weights=augmented_pos)

    lm_out = self.lm.fprop(
        inputs=augmented_labels,
        paddings=paddings,
        labels=labels,
        segment_ids=segment_ids,
        segment_pos=segment_pos)
    lm_out.augmented_labels = augmented_labels
    lm_out.augmented_pos = augmented_pos
    return lm_out

  def compute_loss(self, predictions: NestedMap,  # pytype: disable=signature-mismatch  # jax-ndarray
                   input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    Args:
      predictions: The output of `compute_predictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (metric, weight) pairs as
        values, where one of the entries is expected to corresponds to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    labels = input_batch.labels
    num_tokens = jnp.sum(1.0 - input_batch.paddings.astype(jnp.float32))
    num_seqs = jnp.sum(
        jnp.amax(input_batch.segment_ids.astype(jnp.float32), axis=1))
    weights = predictions.augmented_pos.astype(jnp.float32)
    predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
    num_preds = predictions.total_weight.astype(jnp.float32)
    mean_acc = jnp.sum(
        (labels == predicted_labels) * weights) / jnp.maximum(num_preds, 1)
    metric_weight = jnp.array(num_preds, predictions.avg_xent.dtype)
    metrics = py_utils.NestedMap(
        total_loss=(predictions.total_loss, metric_weight),
        avg_xent=(predictions.avg_xent, metric_weight),
        aux_loss=(predictions.aux_loss, metric_weight),
        log_pplx=(predictions.avg_xent, metric_weight),
        fraction_of_correct_preds=(mean_acc, jnp.array(num_preds,
                                                       mean_acc.dtype)),
        num_predictions=(num_preds, jnp.array(1.0, num_preds.dtype)),
        num_tokens=(num_tokens, jnp.array(1.0, num_tokens.dtype)),
        num_seqs=(num_seqs, jnp.array(1.0, num_seqs.dtype)),
    )

    per_example_output = py_utils.NestedMap()
    return metrics, per_example_output
