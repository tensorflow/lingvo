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
"""Generates the words_to_ids file from a BPE encoded corpus and BPE vocab file.

Extracts all the words in the corpus with their corresponding list of ids. Each
subword in the vocab file is mapped to their line number as its id. The lines of
the output file are like:
...
TAKE 43,7,50,14
THAT 16,35
THE 26
THEIR 16,4,9,56
...
Which is compatible with the BPE tokenizer op in core/tokenizer.py.

Typical workflow:

  subword-nmt learn-bpe train_file code_file
  subword-nmt apply-bpe code_file train_file train_bpe_file
  subword-nmt get-vocab train_bpe_file vocab_file

  bpe_word_tokenizer train_bpe_file vocab_file words_to_ids_file
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.flags.DEFINE_string('encoded_filepath', '',
                       'Path to the BPE encoded corpus file.')
tf.flags.DEFINE_string('vocab_filepath', '', 'Path to the BPE vocab file.')
tf.flags.DEFINE_string('output_filepath', '',
                       'The output filepath (word_to_ids).')
FLAGS = tf.flags.FLAGS


def _GetVocabulary(vocab_filepath):
  """Maps the first word in each line of the given file to its line number."""
  vocab = {}
  with open(vocab_filepath, 'r') as vocab_file:
    for i, line in enumerate(vocab_file):
      word = line.strip('\r\n ').split(' ')[0]
      if word:
        vocab[word] = i
  return vocab


def _ExtractTokenization(encoded_filepath, vocab):
  """Maps the words in the encoded file to their list of token ids.

  Reads all the subwords in encoded file. Concatenates them while they have @@
  as their last two characters. The last token of a word is the subword without
  @@. Maps the full word to the list of corresponding vocab ids of the subwords
  from the vocab dictionary.

  Args:
    encoded_filepath: String, filepath of the BPE encoded file.
    vocab: Dictionary of subwords (string) to token ids (int).

  Returns:
    Dictionary of words (string) to list of token ids (list of int).
  """
  word_tokenization = {}
  with open(encoded_filepath, 'r') as encoded_file:
    for line in encoded_file:
      full_word = ''
      ids = []
      for word in line.strip('\r\n ').split(' '):
        ids.append(vocab[word])
        if word[-2:] == '@@':
          full_word += word[:-2]
        else:
          full_word += word
          word_tokenization[full_word] = ids
          full_word = ''
          ids = []
  return word_tokenization


def main(_):
  vocab = _GetVocabulary(FLAGS.vocab_filepath)
  word_tokenization = _ExtractTokenization(FLAGS.encoded_filepath, vocab)
  with open(FLAGS.output_filepath, 'w') as output:
    for word, ids in word_tokenization.iteritems():
      output.write(word + ' ')
      output.write(','.join(map(str, ids)))
      output.write('\r\n')


if __name__ == '__main__':
  tf.app.run(main)
