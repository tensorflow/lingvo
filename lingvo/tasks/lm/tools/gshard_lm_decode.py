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
r"""Tool to decode GShard LM models.

Sample usage:

.. code-block:: bash

  $ bazel run -c opt lingvo/tasks/lm/tools:gshard_lm_decode -- \
      --model=lm.synthetic_packed_input.DenseLm8B2x2Decode \
      --checkpoint=<checkpoint path> \
      --input=<input file> --output=<output file> \
      --tpu=<tpu node name> --print_outputs=True --is_cloud_tpu_node=True

This binary does not include a tokenizer, so each line in the input file should
be space-separated integer strings, e.g.,
8 3 5 4 2
5 8 2 493 23 432
6 22 3 42 2

To include a tokenizer, override the following functions of GShardLMDecode:
init_vocab(), encode_string_to_ids(), and decode_ids_to_string().
"""
import concurrent.futures
import functools
import sys
import time

from lingvo import compat as tf
from lingvo import model_registry
from lingvo.core import gshard_decode
import numpy as np

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '', 'Registered model name')
tf.flags.DEFINE_string('checkpoint', '', 'Checkpoint path')
tf.flags.DEFINE_string('input', '', 'Input TSV file')
tf.flags.DEFINE_string('output', '/dev/stdout', 'Output TSV file')
tf.flags.DEFINE_boolean('output_score', False, 'Output score to TSV file')
tf.flags.DEFINE_boolean(
    'output_all_samples', False,
    'Output all samples, sampled or within beam, to TSV file')
tf.flags.DEFINE_string('tpu', '', 'TPU node address, if remote.')
tf.flags.DEFINE_boolean('is_cloud_tpu_node', True,
                        'Whether tpu is cloud TPU node.')
tf.flags.DEFINE_string('worker_job', 'worker', 'e.g. worker.')
tf.flags.DEFINE_integer('prefix_max_len', 1024, 'input length limit')
tf.flags.DEFINE_boolean('truncate', True, 'truncate inputs to max_len')
tf.flags.DEFINE_boolean('batch_delimited_mode', False,
                        'For batch decoding, whether to append EOS to input.')
tf.flags.DEFINE_boolean('print_outputs', False,
                        'if false, does not print to stdout')
tf.flags.DEFINE_boolean('heartbeat', False, 'Run heartbeat thread.')
tf.flags.DEFINE_boolean(
    'reshape_weights', True, 'reshape model weights when '
    'restoring from checkpoint as necessary')
tf.flags.DEFINE_boolean('disable_logging', False,
                        'disable all tf.logging calls below level '
                        'CRITICAL')

_daemon = gshard_decode.daemon


def override_flags():
  FLAGS.enable_asserts = False
  FLAGS.xla_device = 'tpu'


class GShardLMDecode(gshard_decode.GShardDecode):
  """GShard LM decoder class.

  This implementation does not include a tokenizer, and directly reads/writes
  integer token IDs. To include a tokenizer, override the following functions:
  init_vocab(), encode_string_to_ids(), and decode_ids_to_string().
  """

  def __init__(self):
    super().__init__(
        tpu=FLAGS.tpu,
        worker_job_name=FLAGS.worker_job,
        prefix_max_len=FLAGS.prefix_max_len,
        is_cloud_tpu_node=FLAGS.is_cloud_tpu_node)
    self.streamz_heartbeat_latency = None
    self.ckpt = FLAGS.checkpoint
    self._heartbeat = FLAGS.heartbeat
    self._saver_reshape = FLAGS.reshape_weights

  def init_vocab(self, model_params):
    self.bos_token_id = model_params.task.decoder_bos_id
    self.eos_token_id = model_params.task.decoder_eos_id

  def encode_string_to_ids(self, string):
    """Returns a list of IDs by parsing a string.

    Args:
      string: str.

    Returns:
      int32 vector.
    """
    return [int(s) for s in string.split(' ')]

  def decode_ids_to_string(self, ids):
    """Returns a string from token IDs.

    Args:
      ids: int32 vector.

    Returns:
      list of strings
    """
    return ' '.join([str(i) for i in ids])

  def ids_to_strings_packed(self, ids, segment_id, skip_empty=True):
    """Returns a list of strings from packed representation.

    Strings in the returned as a flat list,
    ordered first by batch then by segment id.

    Args:
      ids: int32 tensor of shape [batch_size, max_len]
      segment_id: int32 tensor of shape [batch_size, max_len]
      skip_empty: skip empty segments

    Returns:
      list of strings
    """

    assert ids.shape == segment_id.shape
    (batch_size, max_len) = ids.shape
    strs = []
    for b in range(batch_size):
      buf = []
      empty = True
      for i in range(max_len):
        if segment_id[b, i] > 0:
          empty = False
          buf += [int(ids[b, i])]
          if i + 1 == max_len or segment_id[b, i] != segment_id[b, i + 1]:
            strs += [self.decode_ids_to_string(buf)]
            del buf[:]
      if empty and not skip_empty:
        strs += ['']
    return strs

  def preload_lm_prompts(self,
                         tsv_files=None,
                         tsv_data=None,
                         batch_size=None,
                         max_len=None,
                         truncate=False,
                         append_extra_eos=False):
    """Preloads lm prompts from line-delimited files.

    Args:
      tsv_files: input files, list of strings
      tsv_data: TSV data returned by read_files_1_col(); overrides tsv_files
      batch_size: batch size
      max_len: max length
      truncate: truncate inputs to max_len
      append_extra_eos: whether to tack on EOS at the end of the input (for
        delimited_lm model)

    Returns:
      fileno: file number, -1 is padding; int32 tensor [num_batches, batch_size]
      lineno: line number, -1 is padding; int32 tensor [num_batches, batch_size]
      tgt_id: int32 tensor of shape [num_batches, batch_size, max_len]
      tgt_segment_id: int32 tensor of shape [num_batches, batch_size, max_len]
      tgt_segment_pos: int32 tensor of shape [num_batches, batch_size, max_len]
      tgt_labels: int32 tensor of shape [num_batches, batch_size, max_len]

    Raises:
      ValueError: if strings exceed max_len
    """

    if tsv_data is None:
      assert tsv_files
      if isinstance(tsv_files, str):
        tsv_files = [tsv_files]
      tsv_data = read_files_1_col(tsv_files)
    num_lines = sum(len(lines) for lines in tsv_data)
    num_lines_padded = (num_lines + (-num_lines) % batch_size)
    num_batches = num_lines_padded // batch_size
    assert num_lines <= num_batches * batch_size

    fileno = np.zeros([num_batches, batch_size], dtype=np.int32) - 1
    lineno = np.zeros([num_batches, batch_size], dtype=np.int32) - 1
    tgt_id = np.zeros([num_batches, batch_size, max_len], dtype=np.int32)
    tgt_labels = np.zeros([num_batches, batch_size, max_len], dtype=np.int32)
    tgt_segment_id = np.zeros([num_batches, batch_size, max_len],
                              dtype=np.int32)
    tgt_segment_pos = np.zeros([num_batches, batch_size, max_len],
                               dtype=np.int32)

    t = 0
    eos_id = self.eos_token_id
    bos_id = self.bos_token_id

    for file_index, lines in enumerate(tsv_data):
      for line_index, tgt in enumerate(lines):
        m, b, t = t // batch_size, t % batch_size, t + 1
        fileno[m, b] = file_index
        lineno[m, b] = line_index
        tgt_ids_bos = [bos_id] + self.encode_string_to_ids(tgt) + (
            [eos_id] if append_extra_eos else [])

        if truncate:
          del tgt_ids_bos[max_len:]

        if len(tgt_ids_bos) > max_len:
          raise ValueError(
              'tgt_ids size exceeds max_len (%d > %d) for line %d in file %r' %
              (len(tgt_ids_bos), max_len, line_index, tsv_files[file_index]))

        tgt_ids_eos = tgt_ids_bos[1:] + [eos_id]

        for i in range(len(tgt_ids_bos)):
          if i >= max_len:
            break
          tgt_id[m, b, i] = tgt_ids_bos[i]
          tgt_labels[m, b, i] = tgt_ids_eos[i]
          tgt_segment_id[m, b, i] = 1
          tgt_segment_pos[m, b, i] = i

    return (fileno, lineno, tgt_id, tgt_segment_id, tgt_segment_pos, tgt_labels)


def read_file_1_col(f):
  """Reads one TSV file. Returns list of [(src, tgt)]."""
  return [
      line.rstrip('\n')
      for line in tf.io.gfile.GFile(f, 'r').readlines()
      if line.strip()
  ]


def read_files_1_col(ff):
  """Reads multiple TSV files. Returns nested list of [[(src, tgt)]]."""
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(ff)) as ex:
    return list(ex.map(functools.partial(read_file_1_col), ff))


class GShardLMDecodeBatch(GShardLMDecode):
  """Subclass for LM batch decoding."""

  def __init__(self):
    super().__init__()
    # A list of numpy array of shape [num_batches, batch, seqlen].
    self.data = None
    self.num_batches = None
    self.tsv_files = None

  def preload_data(self, batch_size):
    """Preload data into memory."""
    tf.logging.info('Loading input data')
    t0 = time.time()

    tsv_files = FLAGS.input.split(',')
    # in case of TPU rescheduling, skip files already decoded to save time
    self.tsv_files = tsv_files

    tsv_data = read_files_1_col(tsv_files)
    tsv_length = [len(lines) for lines in tsv_data]
    assert 0 not in tsv_length, 'There input files are empty: {}'.format(
        ','.join([
            tsv_files[i] for i, n_line in enumerate(tsv_length) if n_line == 0
        ]))
    self.tsv_length = tsv_length
    data = self.preload_lm_prompts(
        tsv_files=tsv_files,
        tsv_data=tsv_data,
        batch_size=batch_size,
        max_len=self._prefix_max_len,
        truncate=FLAGS.truncate,
        append_extra_eos=FLAGS.batch_delimited_mode)
    # drop source fields
    (fileno, lineno, tgt_id, tgt_segment_id, tgt_segment_pos, tgt_labels) = data
    # add per-example temperature field
    tgt_sample_temperature = np.zeros(tgt_id.shape[:-1], np.float32)

    key = np.stack([fileno, lineno], axis=-1)
    data = (key, tgt_id, tgt_segment_id, tgt_segment_pos, tgt_labels,
            tgt_sample_temperature)

    assert len(data) == 6, (len(data), data)

    self.data = data
    self.num_batches = data[0].shape[0]
    t1 = time.time()
    tf.logging.info('dt=%0.2f', (t1 - t0))
    tf.logging.info('num_batches=%d', self.num_batches)

  def write_ith_sample_to_output(self, output, target, topk_decoded,
                                 topk_scores, i):
    """Write samples to output file."""
    if FLAGS.output_all_samples:
      for j in range(len(topk_decoded[i])):
        if FLAGS.output_score and topk_scores is not None:
          if len(topk_scores.shape) == 2:
            output.write('%s\t%s\t%0.2f\n' %
                         (target, topk_decoded[i][j], topk_scores[i, j]))
          else:
            output.write('%s\t%s\t%s\n' %
                         (target, topk_decoded[i][j], topk_scores[i, j]))
        else:
          output.write('%s\t%s\n' % (target, topk_decoded[i][j]))
    else:
      if FLAGS.output_score and topk_scores is not None:
        if len(topk_scores.shape) == 2:
          output.write('%s\t%s\t%0.2f\n' %
                       (target, topk_decoded[i][0], topk_scores[i, 0]))
        else:
          output.write('%s\t%s\t%s\n' %
                       (target, topk_decoded[i][0], topk_scores[i, 0]))
      else:
        output.write('%s\t%s\n' % (target, topk_decoded[i][0]))

  def batch_decode(self):
    """Starts batch decoding."""
    assert self.outfeed_op is not None
    assert self.outfeed is not None
    assert self.decode_loop is not None

    sess = self.get_session()

    num_batches = self.data[0].shape[0]
    batch_size = self.data[0].shape[1]
    tf.logging.info('num_batches: %s, batch_size: %s', num_batches, batch_size)

    def run_infeed_loop():

      try:
        for t in range(num_batches):
          assert len(self.infeed_args) == len(self.data)
          feeds = {p: x[t] for p, x in zip(self.infeed_args, self.data)}
          print('infeed loop %d' % t)
          sess.run(self.infeed_op, feeds)
          print('infeed loop %d' % t)
      except Exception as e:  # pylint: disable=broad-except
        tf.logging.warning('Exception in infeed loop: %s %r', e, e)

    infeed_loop_thread = _daemon(run_infeed_loop)

    def run_decode_loop():  # pylint: disable=missing-docstring
      try:
        t0 = time.time()
        sess.run([self.decode_loop])
        t1 = time.time()
        tf.logging.info('decode_loop dt=%0.2f lines/sec=%0.2f', (t1 - t0),
                        (num_batches * batch_size) / (t1 - t0))
        return
      except Exception as e:
        tf.logging.error('Exception in decode loop thread: %r %s', e, e)
        raise

    decode_loop_thread = _daemon(run_decode_loop)

    output = None
    if FLAGS.output or FLAGS.split_output:
      tf.logging.info('Start writing output to %s', FLAGS.output)
      output = tf.io.gfile.GFile(FLAGS.output, 'w')

    t0 = time.time()
    for b in range(num_batches):
      [flat_outfeed] = sess.run([self.outfeed_op])
      t1 = time.time()
      tf.logging.info('outfeed_op iter=%d dt=%0.2f', b, (t1 - t0))
      t1 = t0

      (key, tgt_ids, tgt_segment_id, topk_ids, topk_lens, topk_scores,
       dec_metrics) = tf.nest.pack_sequence_as(self.outfeed, flat_outfeed)
      output_len = int(topk_ids.shape[-1])
      topk_segment_id = (np.arange(output_len) < np.expand_dims(
          topk_lens, -1)).astype(np.int32)

      if dec_metrics and FLAGS.print_outputs:
        tf.logging.info('dec_metrics:')
        for k in dec_metrics:
          tf.logging.info('  %r\t%s', dec_metrics[k], k)

      # TODO(krikun): somehow there is -1 in tgt_ids
      tgt_ids = np.maximum(tgt_ids, 0)
      targets = self.ids_to_strings_packed(tgt_ids, tgt_segment_id)

      if topk_ids is not None:
        # TODO(krikun): somehow there is -1 in topk_ids
        topk_ids = np.maximum(topk_ids, 0)
        topk_decoded = self.ids_to_strings_packed(
            topk_ids, topk_segment_id, skip_empty=False)
        topk_decoded = np.reshape(topk_decoded, [batch_size, -1])
      else:
        topk_decoded = None

      if topk_scores is not None:
        topk_scores = np.reshape(topk_scores, [batch_size, -1])

      if topk_decoded is None or targets is None:
        continue

      fileno, lineno = key[:, 0], key[:, 1]
      for i, target in enumerate(targets):
        if fileno[i] == -1 or lineno[i] == -1:
          continue

        if FLAGS.print_outputs:
          sys.stderr.flush()
          print('prompt[%d]: %s' % (i, target))
          if topk_scores is None:
            for j, topk_target in enumerate(topk_decoded[i]):
              print('topk_decoded[%d][%d]: %s' % (i, j, topk_target))
          elif len(topk_scores.shape) == 2:
            for j, topk_target in enumerate(topk_decoded[i]):
              print('topk_decoded[%d][%d]: (%0.2f) %s' %
                    (i, j, topk_scores[i, j], topk_target))
          else:
            for j, topk_target in enumerate(topk_decoded[i]):
              with np.printoptions(precision=4):
                print('topk_decoded[%d][%d]: (%s) %s' %
                      (i, j, topk_scores[i, j], topk_target))

          sys.stdout.flush()
        if output:
          self.write_ith_sample_to_output(output, target, topk_decoded,
                                          topk_scores, i)

    tf.logging.info('Waiting for infeed thread')
    infeed_loop_thread.join()

    tf.logging.info('Waiting for TPU thread')
    decode_loop_thread.join()
    tf.logging.info('Done')


def main(unused_argv):
  override_flags()
  if FLAGS.disable_logging:
    tf.get_logger().setLevel('CRITICAL')

  model_params = model_registry.GetParams(FLAGS.model, None)
  tf.logging.info('Found model %s', FLAGS.model)

  batch_size = model_params.task.batch_size

  decoder = GShardLMDecodeBatch()
  decoder.init_vocab(model_params)
  decoder.preload_data(batch_size)

  decoder.reset_tpu_cluster()
  decoder.reset_session()

  try:
    decoder.init_graph(model_params)
    decoder.run_init_sequence()
    decoder.batch_decode()
  finally:
    decoder.reset_tpu_cluster()


if __name__ == '__main__':
  tf.app.run(main)
