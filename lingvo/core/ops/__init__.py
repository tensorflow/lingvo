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
"""Operations."""

from lingvo import compat as tf

# Try static linking:
try:
  from lingvo.core.ops import gen_x_ops  # pylint: disable=g-import-not-at-top
except ImportError:
  gen_x_ops = tf.load_op_library(
      tf.resource_loader.get_path_to_datafile('x_ops.so'))

# Set gen_x_ops function module so sphinx generates documentation.
for v in gen_x_ops.__dict__.values():
  try:
    v.__module__ = 'lingvo.core.ops'
  except:  # pylint: disable=bare-except
    pass

assert_shape_match = gen_x_ops.assert_shape_match
assert_same_dim0 = gen_x_ops.assert_same_dim0
random_permutation_sequence = gen_x_ops.random_permutation_sequence

best_step = gen_x_ops.best_step

beam_search_step = gen_x_ops.beam_search_step
beam_search_step_v2 = gen_x_ops.beam_search_step_v2
top_k_terminated_hyps = gen_x_ops.top_k_terminated_hyps
unpack_hyp = gen_x_ops.unpack_hyp
hyps_from_beam_search_outs = gen_x_ops.hyps_from_beam_search_outs

cached_call = gen_x_ops.cached_call

vocab_token_to_id = gen_x_ops.vocab_token_to_id
vocab_id_to_token = gen_x_ops.vocab_id_to_token
token_in_vocab = gen_x_ops.token_in_vocab
ascii_to_token_id = gen_x_ops.ascii_to_token_id
str_to_vocab_tokens = gen_x_ops.str_to_vocab_tokens
id_to_ascii = gen_x_ops.id_to_ascii
ml_perf_subword_id_to_string = gen_x_ops.ml_perf_subword_id_to_string
ngram_id_to_token = gen_x_ops.ngram_id_to_token
bpe_ids_to_words = gen_x_ops.bpe_ids_to_words
bpe_words_to_ids = gen_x_ops.bpe_words_to_ids

static_map_string_int = gen_x_ops.static_map_string_int
static_map_int_string = gen_x_ops.static_map_int_string
static_map_int_int = gen_x_ops.static_map_int_int

get_preconditioners = gen_x_ops.get_preconditioners
compute_preconditioners = gen_x_ops.compute_preconditioners

pack_sequences = gen_x_ops.pack_sequences
pack_single_sequence = gen_x_ops.pack_single_sequence
apply_packing = gen_x_ops.apply_packing
mass = gen_x_ops.mass
