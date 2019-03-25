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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import cluster_factory
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.core import summary_utils
from lingvo.tasks.mt import decoder
from lingvo.tasks.mt import encoder


class MTBaseModel(base_model.BaseTask):
  """Base Class for NMT models."""

  def _EncoderDevice(self):
    """Returns the device to run the encoder computation."""
    if py_utils.use_tpu():
      return tf.device(self.cluster.WorkerDeviceInModelSplit(0))
    else:
      return tf.device('')

  def _DecoderDevice(self):
    """Returns the device to run the decoder computation."""
    if py_utils.use_tpu():
      return tf.device(self.cluster.WorkerDeviceInModelSplit(1))
    else:
      return tf.device('')

  @base_layer.initializer
  def __init__(self, params):
    super(MTBaseModel, self).__init__(params)
    p = self.params

    with tf.variable_scope(p.name):
      with self._EncoderDevice():
        self.CreateChild('enc', p.encoder)
      with self._DecoderDevice():
        self.CreateChild('dec', p.decoder)

  def ComputePredictions(self, theta, batch):
    with self._EncoderDevice():
      encoder_outputs = self.enc.FProp(theta.enc, batch.src)
    with self._DecoderDevice():
      return self.dec.ComputePredictions(theta.dec, encoder_outputs, batch.tgt)

  def ComputeLoss(self, theta, batch, predictions):
    with self._DecoderDevice():
      return self.dec.ComputeLoss(theta.dec, predictions, batch.tgt)

  def _GetTokenizerKeyToUse(self, key):
    """Returns a tokenizer key to use for the provided `key`."""
    if key in self.input_generator.tokenizer_dict:
      return key
    return None

  def _BeamSearchDecode(self, input_batch):
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      encoder_outputs = self.enc.FPropDefaultTheta(input_batch.src)
      decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

      topk_hyps = decoder_outs.topk_hyps
      topk_ids = decoder_outs.topk_ids
      topk_lens = decoder_outs.topk_lens
      topk_scores = decoder_outs.topk_scores

      slen = tf.to_int32(tf.reduce_sum(1 - input_batch.src.paddings, 1) - 1)
      srcs = self.input_generator.IdsToStrings(
          input_batch.src.ids, slen, self._GetTokenizerKeyToUse('src'))
      topk_decoded = self.input_generator.IdsToStrings(
          topk_ids, topk_lens - 1, self._GetTokenizerKeyToUse('tgt'))
      topk_decoded = tf.reshape(topk_decoded, tf.shape(topk_hyps))
      topk_scores = tf.reshape(topk_scores, tf.shape(topk_hyps))

      refs = self.input_generator.IdsToStrings(
          input_batch.tgt.labels,
          tf.to_int32(tf.reduce_sum(1.0 - input_batch.tgt.paddings, 1) - 1.0),
          self._GetTokenizerKeyToUse('tgt'))

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
    p = super(TransformerModel, cls).Params()
    p.encoder = encoder.TransformerEncoder.Params()
    p.decoder = decoder.TransformerDecoder.Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerModel, self).__init__(params)
    p = self.params
    assert p.encoder.model_dim == p.decoder.source_dim

  def BProp(self):
    super(TransformerModel, self).BProp()
    # Computes gradients' norm and adds their summaries.
    p = self.params
    vg = self._var_grads
    emb_vg = py_utils.NestedMap()
    emb_vg.child = [vg.enc.token_emb, vg.dec.token_emb]

    # Note that positional embedding layer has no trainable variable
    # if its trainable_scaling is false.
    if 'position_emb' in vg.enc:
      emb_vg.child += [vg.enc.position_emb]
    if 'position_emb' in vg.dec:
      emb_vg.child += [vg.dec.position_emb]
    summary_utils.AddNormSummary('emb', emb_vg)
    summary_utils.AddNormSummary('atten',
                                 [vg.enc.transformer_stack.trans, vg.dec.trans])
    summary_utils.AddNormSummary('softmax', vg.dec.softmax)


class RNMTModel(MTBaseModel):
  """RNMT+ Model.

  Implements RNMT Variants in The Best of Both Worlds paper:
  https://aclweb.org/anthology/P18-1008
  """

  @classmethod
  def Params(cls):
    p = super(RNMTModel, cls).Params()
    p.encoder = encoder.MTEncoderBiRNN.Params()
    p.decoder = decoder.MTDecoderV1.Params()
    return p

  def BProp(self):
    super(RNMTModel, self).BProp()

    if self.cluster.add_summary:
      vg = self._var_grads
      # Computes gradients' norm and adds their summaries.
      emb_grads = []
      rnn_grads = []
      atten_grads = []
      softmax_grads = []
      if 'enc' in vg:
        emb_grads += [vg.enc.emb] if 'emb' in vg.enc else []
        rnn_grads += [vg.enc.rnn] if 'rnn' in vg.enc else []
      if 'dec' in vg:
        emb_grads += [vg.dec.emb] if 'emb' in vg.dec else []
        rnn_grads += [vg.dec.frnn] if 'frnn' in vg.dec else []
        softmax_grads += [vg.dec.softmax] if 'softmax' in vg.dec else []
        if 'frnn_with_atten' in vg.dec:
          if 'cell' in vg.dec.frnn_with_atten:
            rnn_grads += [vg.dec.frnn_with_atten.cell]
          if 'atten' in vg.dec.frnn_with_atten:
            atten_grads += [vg.dec.frnn_with_atten.atten]

      if emb_grads:
        summary_utils.AddNormSummary('emb', emb_grads)
      if rnn_grads:
        summary_utils.AddNormSummary('lstm', rnn_grads)
      if atten_grads:
        summary_utils.AddNormSummary('atten', atten_grads)
      if softmax_grads:
        summary_utils.AddNormSummary('softmax', softmax_grads)
