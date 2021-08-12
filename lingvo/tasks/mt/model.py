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
"""MT models."""

import lingvo.compat as tf
from lingvo.core import base_model
from lingvo.core import insertion
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.tasks.mt import decoder
from lingvo.tasks.mt import encoder


class MTBaseModel(base_model.BaseTask):
  """Base Class for NMT models."""

  def _EncoderDevice(self):
    """Returns the device to run the encoder computation."""
    if self.params.device_mesh is not None:
      # We perform spmd based partitioning, in which case, we don't specifically
      # assign any operation to a particular device.
      return tf.device('')
    if py_utils.use_tpu():
      return tf.device(self.cluster.WorkerDeviceInModelSplit(0))
    else:
      return tf.device('')

  def _DecoderDevice(self):
    """Returns the device to run the decoder computation."""
    if self.params.device_mesh is not None:
      # We perform spmd based partitioning, in which case, we don't specifically
      # assign any operation to a particular device.
      return tf.device('')
    if py_utils.use_tpu():
      return tf.device(self.cluster.WorkerDeviceInModelSplit(1))
    else:
      return tf.device('')

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.encoder:
      self.CreateChild('enc', p.encoder)
    self.CreateChild('dec', p.decoder)

  def ComputePredictions(self, theta, batch):
    p = self.params

    with self._EncoderDevice():
      encoder_outputs = (
          self.enc.FProp(theta.enc, batch.src) if p.encoder else None)
    with self._DecoderDevice():
      predictions = self.dec.ComputePredictions(theta.dec, encoder_outputs,
                                                batch.tgt)
      if isinstance(predictions, py_utils.NestedMap):
        # Pass through encoder output as well for possible use as a FProp output
        # for various meta-MT modeling approaches, such as MT quality estimation
        # classification.
        predictions['encoder_outputs'] = encoder_outputs
      return predictions

  def ComputeLoss(self, theta, predictions, input_batch):
    with self._DecoderDevice():
      return self.dec.ComputeLoss(theta.dec, predictions, input_batch.tgt)

  def _GetTokenizerKeyToUse(self, key):
    """Returns a tokenizer key to use for the provided `key`."""
    if key in self.input_generator.tokenizer_dict:
      return key
    return None

  def _BeamSearchDecode(self, input_batch):
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      encoder_outputs = self.enc.FPropDefaultTheta(input_batch.src)
      encoder_outputs = self.dec.AddExtraDecodingInfo(encoder_outputs,
                                                      input_batch.tgt)
      decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

      topk_hyps = decoder_outs.topk_hyps
      topk_ids = decoder_outs.topk_ids
      topk_lens = decoder_outs.topk_lens
      topk_scores = decoder_outs.topk_scores

      slen = tf.cast(
          tf.round(tf.reduce_sum(1 - input_batch.src.paddings, 1) - 1),
          tf.int32)
      srcs = self.input_generator.IdsToStrings(
          input_batch.src.ids, slen, self._GetTokenizerKeyToUse('src'))
      topk_decoded = self.input_generator.IdsToStrings(
          topk_ids, topk_lens - 1, self._GetTokenizerKeyToUse('tgt'))
      topk_decoded = tf.reshape(topk_decoded, tf.shape(topk_hyps))
      topk_scores = tf.reshape(topk_scores, tf.shape(topk_hyps))

      refs = self.input_generator.IdsToStrings(
          input_batch.tgt.labels,
          tf.cast(
              tf.round(tf.reduce_sum(1.0 - input_batch.tgt.paddings, 1) - 1.0),
              tf.int32), self._GetTokenizerKeyToUse('tgt'))

      ret_dict = {
          'target_ids': input_batch.tgt.ids,
          'target_labels': input_batch.tgt.labels,
          'target_weights': input_batch.tgt.weights,
          'target_paddings': input_batch.tgt.paddings,
          'sources': srcs,
          'targets': refs,
          'topk_decoded': topk_decoded,
          'topk_lens': topk_lens,
          'topk_scores': topk_scores,
      }
      return ret_dict

  def _PostProcessBeamSearchDecodeOut(self, dec_out_dict, dec_metrics_dict):
    """Post processes the output from `_BeamSearchDecode`."""
    p = self.params
    topk_scores = dec_out_dict['topk_scores']
    topk_decoded = dec_out_dict['topk_decoded']
    targets = dec_out_dict['targets']
    sources = dec_out_dict['sources']
    unsegment = dec_metrics_dict['corpus_bleu'].unsegmenter

    num_samples = len(targets)
    assert num_samples == len(topk_decoded), (
        '%s vs %s' % (num_samples, len(topk_decoded)))
    assert num_samples == len(sources)
    dec_metrics_dict['num_samples_in_batch'].Update(num_samples)

    key_value_pairs = []
    for i in range(num_samples):
      src, tgt = sources[i], targets[i]
      src_unseg, tgt_unseg = unsegment(src), unsegment(tgt)
      tf.logging.info('source: %s', src_unseg)
      tf.logging.info('target: %s', tgt_unseg)
      hyps = topk_decoded[i]
      assert p.decoder.beam_search.num_hyps_per_beam == len(hyps)
      info_str = u'src: {} tgt: {} '.format(src_unseg, tgt_unseg)
      for n, (score, hyp_str) in enumerate(zip(topk_scores[i], hyps)):
        hyp_str_unseg = unsegment(hyp_str)
        tf.logging.info('  %f: %s', score, hyp_str_unseg)
        info_str += u' hyp{n}: {hyp} score{n}: {score}'.format(
            n=n, hyp=hyp_str_unseg, score=score)
        # Only aggregate scores of the top hypothesis.
        if n == 0:
          dec_metrics_dict['corpus_bleu'].Update(tgt, hyp_str)
      key_value_pairs.append((src_unseg, info_str))
    return key_value_pairs

  def CreateDecoderMetrics(self):
    decoder_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
        'corpus_bleu': metrics.CorpusBleuMetric(separator_type='wpm'),
    }
    return decoder_metrics

  def Decode(self, input_batch):
    """Constructs the decoding graph."""
    return self._BeamSearchDecode(input_batch)

  def PostProcessDecodeOut(self, dec_out, dec_metrics):
    return self._PostProcessBeamSearchDecodeOut(dec_out, dec_metrics)


class TransformerModel(MTBaseModel):
  """Transformer Model.

  Implements Attention is All You Need:
  https://arxiv.org/abs/1706.03762
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.encoder = encoder.TransformerEncoder.Params()
    p.decoder = decoder.TransformerDecoder.Params()
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.encoder.model_dim == p.decoder.source_dim


class RNMTModel(MTBaseModel):
  """RNMT+ Model.

  Implements RNMT Variants in The Best of Both Worlds paper:
  https://aclweb.org/anthology/P18-1008
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.encoder = encoder.MTEncoderBiRNN.Params()
    p.decoder = decoder.MTDecoderV1.Params()
    return p


class InsertionModel(MTBaseModel):
  """Insertion-based model.

  References:
    KERMIT: https://arxiv.org/pdf/1906.01604.pdf
    Insertion Transformer: https://arxiv.org/pdf/1902.03249.pdf
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.decoder = decoder.InsertionDecoder.Params()
    p.Define('insertion', insertion.SymbolInsertionLayer.Params(),
             'Insertion specifications (i.e., rollin and oracle policy).')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    self.CreateChild('insertion', p.insertion)

  def _SampleCanvasAndTargets(self, x, x_paddings):
    """Sample a canvas and its corresponding targets.

    Args:
      x: A Tensor representing the canvas.
      x_paddings: A Tensor representing the canvas paddings.

    Returns:
      A `NestedMap` capturing the new sampled canvas and its targets.
    """
    p = self.params

    # TODO(williamchan): Consider grabbing `eos_id` from `x` instead of `p`.
    eos_id = p.decoder.target_eos_id

    # Sample a canvas (and it's corresponding targets).
    return self.insertion.FProp(None, x, x_paddings, eos_id, True)

  def _CreateCanvasAndTargets(self, batch):
    # pyformat: disable
    """Create the canvas and targets.

    Args:
      batch: A `.NestedMap`.

        - src: A `.NestedMap`.
          - ids: The source ids, ends in <eos>.
          - paddings: The source paddings.

        - tgt: A `.NestedMap`.
          - ids: The target ids, ends in <eos>.
          - paddings: The target paddings.

    Returns:
      A `NestedMap`.
        - canvas: The canvas (based off of the `rollin_policy`) of shape
          [batch_size, c_dim].
        - canvas_paddings: The paddings of `canvas_indices`.
        - target_indices: The target indices (i.e., use these indices to
          tf.gather_nd the log-probs). Optional, only during training.
        - target_weights: The target weights. Optional, only during training.
    """
    # pyformat: enable
    p = self.params

    if not self.do_eval:
      # Sample our src and tgt canvas.
      src_descriptor = self._SampleCanvasAndTargets(batch.src.ids,
                                                    batch.src.paddings)
      tgt_descriptor = self._SampleCanvasAndTargets(batch.tgt.ids,
                                                    batch.tgt.paddings)

      # Offset the src ids (to unshare embeddings between src/tgt). Note, we
      # only offset the canvas ids, but we do not offset the vocab ids. This
      # will result in unshared embeddings, but shared softmax. This is due to
      # GPU/TPU memory limitations, empirically it is known that unsharing
      # everything results in better performance.
      vocab_size = p.decoder.softmax.num_classes
      src_descriptor.canvas = tf.where(
          tf.equal(src_descriptor.canvas_paddings, 0),
          src_descriptor.canvas + vocab_size, src_descriptor.canvas)

      # Offset the tgt indices (need shift according to src length).
      batch_size = py_utils.GetShape(batch.src.ids)[0]
      # `target_batch` is a [num_targets, batch_size] tensor where each row
      # identifies which batch the target belongs to. Note the observation that,
      # tf.reduce_sum(target_batch, 1) == 1 \forall rows.
      target_batch = tf.cast(
          tf.equal(
              tf.expand_dims(tf.range(batch_size), 0),
              tf.expand_dims(tgt_descriptor.target_indices[:, 0], 1)), tf.int32)
      src_lens = tf.cast(
          tf.reduce_sum(1 - src_descriptor.canvas_paddings, 1), tf.int32)
      # `tgt_offset` is shape [num_targets] where each entry corresponds to the
      # offset needed for that target (due to the source length).
      tgt_offset = tf.matmul(target_batch, tf.expand_dims(src_lens, 1))
      # We shift the tgt slot without touching the batch or vocab.
      tgt_descriptor.target_indices += tf.concat(
          [tf.zeros_like(tgt_offset), tgt_offset,
           tf.zeros_like(tgt_offset)], 1)

      # The canvas is simply the sequence-level concat of the src and tgt.
      canvas, canvas_paddings = insertion.SequenceConcat(
          src_descriptor.canvas, src_descriptor.canvas_paddings,
          tgt_descriptor.canvas, tgt_descriptor.canvas_paddings)
      target_indices = tf.concat(
          [src_descriptor.target_indices, tgt_descriptor.target_indices], 0)
      target_weights = tf.concat(
          [src_descriptor.target_weights, tgt_descriptor.target_weights], 0)

      return py_utils.NestedMap(
          canvas=canvas,
          canvas_paddings=canvas_paddings,
          target_indices=target_indices,
          target_weights=target_weights)

  def ComputePredictions(self, theta, batch):
    # pyformat: disable
    """Compute the model predictions.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      batch: A `.NestedMap`.

        - src: A `.NestedMap`.
          - ids: The source ids, ends in <eos>.
          - paddings: The source paddings.

        - tgt: A `.NestedMap`.
          - ids: The target ids, ends in <eos>.
          - paddings: The target paddings.

    Returns:
      A `.NestedMap`.
        - outputs: The contextualized output vectors of shape
          [batch_size, time_dim, model_dim].
        - tgt: A `.NestedMap` (optional, only during training).
          - ids: The canvas ids.
          - paddings: The canvas paddings.
          - target_indices: The target indices.
          - target_weights: The target weights.
    """
    # pyformat: enable
    p = self.params

    # TODO(williamchan): Currently, we only support KERMIT mode (i.e., no
    # encoder, unified architecture).
    assert not p.encoder

    # Sometimes src and tgt have different types. We reconcile here and use
    # int32.
    batch.src.ids = tf.cast(batch.src.ids, tf.int32)
    batch.tgt.ids = tf.cast(batch.tgt.ids, tf.int32)

    canvas_and_targets = self._CreateCanvasAndTargets(batch)
    batch = py_utils.NestedMap(
        tgt=py_utils.NestedMap(
            ids=canvas_and_targets.canvas,
            paddings=canvas_and_targets.canvas_paddings))

    predictions = super().ComputePredictions(theta, batch)

    if not self.do_eval:
      predictions.tgt = py_utils.NestedMap(
          ids=canvas_and_targets.canvas,
          paddings=canvas_and_targets.canvas_paddings,
          target_indices=canvas_and_targets.target_indices,
          target_weights=canvas_and_targets.target_weights)

    return predictions


class TransformerXEnDecModel(TransformerModel):
  """Implementation of XEnDec.

  Refer to https://arxiv.org/abs/2106.04060.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.encoder = encoder.TransformerXEncoder.Params()
    p.decoder = decoder.TransformerXDecoder.Params()
    p.Define('loss_mix_weight', 1.0, 'Weight for mix loss')
    p.Define('loss_clean_weight', 1.0, 'Weight for clean loss')
    p.Define('loss_mono_weight', 1.0, 'Weight for mono loss')
    p.Define('use_atten_drop', False, 'use attention dropout')
    p.Define('use_prob_cl', False, 'use prob cl')
    p.Define('use_prob_drop', False, 'prob drop out')
    p.Define('atten_drop', 0.0, 'attention drop')
    return p

  def _CreateTargetLambdas(self,
                           atten_probs,
                           source_lambdas_pair,
                           source_paddings_pair,
                           target_paddings_pair,
                           smooth=0):
    """Compute target interpolation ratios.

    Args:
      atten_probs: A list containing two attention matrics.
      source_lambdas_pair: A list containing two source interpolation ratios.
      source_paddings_pair: A list containing two source paddings.
      target_paddings_pair: A list containing two target paddings
      smooth: A real value to smooth target interpolation ratios before
        normalization.

    Returns:
      source_lambdas_pair: Source interpolation ratios.
      input_lambdas: Interpolation ratios for target input embeddings.
      label_lambdas: Interpolation ratios for target labels.
    """
    atten_probs_0 = tf.stop_gradient(atten_probs[0])
    atten_probs_1 = tf.stop_gradient(atten_probs[1])

    source_lambdas = source_lambdas_pair[0]
    other_source_lambdas = source_lambdas_pair[1]
    lambdas_0 = atten_probs_0 * tf.expand_dims(
        source_lambdas * (1.0 - source_paddings_pair[0]), 1)

    lambdas_0 = tf.reduce_sum(lambdas_0, -1)
    lambdas_0 = (lambdas_0 + smooth) * (1.0 - target_paddings_pair[0])
    lambdas_1 = atten_probs_1 * tf.expand_dims(
        other_source_lambdas * (1.0 - source_paddings_pair[1]), 1)
    lambdas_1 = tf.reduce_sum(lambdas_1, -1)
    lambdas_1 = (lambdas_1 + smooth) * (1.0 - target_paddings_pair[1])
    label_lambdas_0 = lambdas_0 / (lambdas_0 + lambdas_1 + 1e-9)

    label_lambdas = [
        label_lambdas_0 * (1. - target_paddings_pair[0]),
        (1.0 - label_lambdas_0) * (1. - target_paddings_pair[1])
    ]
    input_lambdas_0 = tf.pad(
        label_lambdas_0, [[0, 0], [1, 0]], constant_values=1.)[:, :-1]
    input_lambdas = [
        input_lambdas_0 * (1. - target_paddings_pair[0]),
        (1.0 - input_lambdas_0) * (1. - target_paddings_pair[1])
    ]

    return source_lambdas_pair, input_lambdas, label_lambdas

  def ComputePredictions(self,
                         theta,
                         batch,
                         other_batch=None,
                         source_lambdas=None,
                         target_lambdas=None):
    # pyformat: disable
    """Compute the model predictions.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      batch: A `.NestedMap`.

      - src: A `.NestedMap`.
        - ids: The source ids, ends in <eos>.
        - paddings: The source paddings.
        - embs: The source embeddings. It is none when other_batch == None.

      - tgt: A `.NestedMap`.
        - ids: The target ids, ends in <eos>.
        - paddings: The target paddings.
        - embs: The source embeddings. It is none when other_batch == None.

      other_batch: Same as batch.
      source_lambdas: Interpolation ratios for source embeddings.
      target_lambdas: Interpolation ratios for target embeddings.

    Returns:
      A `.NestedMap` containg:

      - encoder_outputs: The contextualized output vectors from encoder.
      - softmax_input: Tensor of shape [time, batch, params.softmax.input_dim].
      - attention: `.NestedMap` of attention distributions of shape
        [batch, target_length, source_length].
      - source_embs: Tensor of shape [batch, time, emb_dim].
      - target_embs: Tensor of shape [batch, time, emb_dim].
    """
    # pyformat: enable
    p = self.params

    with self._EncoderDevice():
      other_batch_src = None
      if other_batch is not None:
        other_batch_src = other_batch.src
      encoder_outputs = (
          self.enc.FProp(theta.enc, batch.src, other_batch_src, source_lambdas)
          if p.encoder else None)
    with self._DecoderDevice():
      other_batch_tgt = None
      if other_batch is not None:
        other_batch_tgt = other_batch.tgt
      predictions = self.dec.ComputePredictions(theta.dec, encoder_outputs,
                                                batch.tgt, other_batch_tgt,
                                                target_lambdas)
      if isinstance(predictions, py_utils.NestedMap):
        predictions['encoder_outputs'] = encoder_outputs
      return predictions

  def ComputeLoss(self, theta, predictions, input_batch):
    p = self.params
    # Computes the loss for input_batch.
    with self._DecoderDevice():
      result = self.dec.ComputeLoss(theta.dec, predictions, input_batch.tgt)
      if self.do_eval:
        return result

    probs = result[1]['reshape_probs']
    probs_hard = result[1]['target_hard_probs']
    atten_probs = predictions.attention.probs

    if 'other_src' in input_batch and 'other_tgt' in input_batch:
      other_batch = py_utils.NestedMap()
      other_batch.src = input_batch.other_src.DeepCopy()
      other_batch.tgt = input_batch.other_tgt.DeepCopy()
    else:
      other_batch = py_utils.NestedMap()
      other_batch.src = input_batch.src.DeepCopy()
      other_batch.tgt = input_batch.tgt.DeepCopy()
      other_batch = other_batch.Transform(lambda x: tf.roll(x, 1, 0))
      other_atten_probs = tf.roll(atten_probs, 1, 0)
      other_probs = tf.roll(probs, 1, 0)
      other_probs_hard = tf.roll(probs_hard, 1, 0)
      other_predictions = py_utils.NestedMap()
      other_predictions.source_embs = tf.roll(predictions.source_embs, 1, 0)
      other_predictions.target_embs = tf.roll(predictions.target_embs, 1, 0)

    # Computes the loss for other_batch.
    if p.loss_mono_weight > 0:
      other_predictions = self.ComputePredictions(theta, other_batch)
      with self._DecoderDevice():
        other_result = self.dec.ComputeLoss(theta.dec, other_predictions,
                                            other_batch.tgt)
        other_atten_probs = other_predictions.attention.probs
        other_probs = other_result[1]['reshape_probs']
        other_probs_hard = other_result[1]['target_hard_probs']

    if p.use_atten_drop:
      atten_probs = tf.nn.dropout(atten_probs, p.atten_drop)
      if other_atten_probs is not None:
        other_atten_probs = tf.nn.dropout(other_atten_probs, p.atten_drop)

    # Computes the xendec loss.
    mix_results = []
    if p.loss_mix_weight > 0:
      if other_atten_probs is not None:
        if p.use_prob_cl:
          cur_step = py_utils.GetGlobalStep()
          cur_ratio = tf.minimum(
              tf.cast(cur_step, py_utils.FPropDtype(p)) / 40000, 0.8)
          probs_hard = tf.cast(probs_hard, py_utils.FPropDtype(p))
          other_probs_hard = tf.cast(other_probs_hard, py_utils.FPropDtype(p))
          prob_ratio = tf.expand_dims(input_batch.tgt.weights, -1) * cur_ratio
          probs = probs_hard * (1.0 - prob_ratio) + probs * prob_ratio
          other_prob_ratio = tf.expand_dims(other_batch.tgt.weights,
                                            -1) * cur_ratio
          other_probs = other_probs_hard * (
              1.0 - other_prob_ratio) + other_probs * other_prob_ratio
        else:
          probs = tf.cast(probs_hard, py_utils.FPropDtype(p))
          other_probs = tf.cast(other_probs_hard, py_utils.FPropDtype(p))

      source_paddings_pair = [
          input_batch.src.paddings, other_batch.src.paddings
      ]
      target_paddings_pair = [
          input_batch.tgt.paddings, other_batch.tgt.paddings
      ]

      source_mask = input_batch.src.source_mask
      other_lambdas = source_mask * (1. - source_paddings_pair[1])
      source_lambdas = (1. - other_lambdas) * (1. - source_paddings_pair[0])
      source_lambdas = [source_lambdas, other_lambdas]

      source_lambdas, input_lambdas, label_lambdas = self._CreateTargetLambdas(
          [atten_probs, other_atten_probs],
          source_lambdas,
          source_paddings_pair,
          target_paddings_pair,
          smooth=0.001)

      mix_tgt = input_batch.tgt
      target_weights = input_batch.tgt.weights + other_batch.tgt.weights
      target_weights = tf.clip_by_value(target_weights, 0.0, 1.0)
      mix_tgt.weights = target_weights

      input_batch.src.embs = predictions.source_embs
      input_batch.tgt.embs = predictions.target_embs
      other_batch.src.embs = other_predictions.source_embs
      other_batch.tgt.embs = other_predictions.target_embs

      mix_predictions = self.ComputePredictions(theta, input_batch, other_batch,
                                                source_lambdas, input_lambdas)

      target_probs_0 = probs
      target_probs_1 = other_probs

      target_probs = target_probs_0 * tf.expand_dims(
          label_lambdas[0], -1) + target_probs_1 * tf.expand_dims(
              label_lambdas[1], -1)

      target_probs = target_probs + 1e-9
      target_probs = target_probs / tf.reduce_sum(
          target_probs, -1, keepdims=True)

      with self._DecoderDevice():
        mix_result = self.dec.ComputeLoss(theta.dec, mix_predictions, mix_tgt,
                                          target_probs)
      mix_results.append(mix_result)

    losses = []
    loss_names = []
    loss_weights = []
    new_metrics = {}

    if p.loss_clean_weight > 0:
      losses.append(result)
      loss_weights.append(p.loss_clean_weight)
      loss_names.append('clean_loss')

    if p.loss_mono_weight > 0:
      losses.append(other_result)
      loss_weights.append(p.loss_mono_weight)
      loss_names.append('other_loss')

    if p.loss_mix_weight > 0.0:
      for idx, mix_result in enumerate(mix_results):
        losses.append(mix_result)
        loss_weights.append(p.loss_mix_weight)
        loss_names.append('mix_loss_' + str(idx))

    loss_length = len(loss_names)
    assert loss_length > 0

    # Combines three losses.
    for i in range(len(loss_names)):
      new_metrics[loss_names[i]] = (losses[i][0]['loss'][0] * loss_weights[i],
                                    losses[i][0]['loss'][1])
    return new_metrics, losses[0][1]

  def _FPropResult(self, dec_metrics, per_example):
    # Adds stats about the input batch.
    p = self.params
    if p.input is not None:
      dec_metrics['num_samples_in_batch'] = (tf.convert_to_tensor(
          self.input_generator.GlobalBatchSize()), tf.constant(1.0))
    # Generates summaries.
    for name, (value, weight) in dec_metrics.items():
      self.AddEvalMetric(name, value, weight)
    per_example = self.FilterPerExampleTensors(per_example)
    for name, value in per_example.items():
      self.AddPerExampleTensor(name, value)
    # Loss.
    if self.do_eval:
      self._loss, self._num_predictions = dec_metrics['loss']
    else:
      self._loss = 0
      for key, value in dec_metrics.items():
        if 'loss' in key:
          self._loss = self._loss + value[0]
          self._num_predictions = value[1]
      if 'clean_loss' in dec_metrics:
        self._num_predictions = dec_metrics['clean_loss'][1]
      dec_metrics['loss'] = (self._loss, self._num_predictions)
      self.AddEvalMetric('loss', self._loss, self._num_predictions)
    self._loss = py_utils.CheckNumerics(self._loss)
    self._metrics = dec_metrics
