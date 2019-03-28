/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "lingvo/core/ops/x_ops_helper.h"

namespace tensorflow {
namespace {

REGISTER_OP("AssertShapeMatch")
    .Input("x: int32")
    .Input("y: int32")
    .Attr("msg: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return Status::OK();
    })
    .Doc(R"doc(
Asserts that shape vector x and y matches.

The i-th dimension matches iff x[i] == y[i] || x[i] == -1 || y[i] == -1.

x: A shape vector.
y: A shape vector.
msg: The error message generated when the assertion failed.
)doc");

REGISTER_OP("AssertSameDim0")
    .Input("x: types")
    .Attr("msg: string = ''")
    .Attr("types: list(type)")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return Status::OK();
    })
    .Doc(R"doc(
Asserts that all input tensors are non-scalar and have the same 0-th dim size.

x: The list of tensors.
msg: The error message generated when the assertion failed.
)doc");

REGISTER_OP("RandomPermutationSequence")
    .Attr("num: int = 1")
    .Attr("batch: int = 1")
    .Attr("repeat: bool = false")
    .Attr("seed: int = 0")
    .Output("out: int32")
    .SetIsStateful()
    .Doc(R"doc(
Generate random samples from [0..num-1] without replacements.

num: The number of ids.
batch: Each output is a vector of size up to batch. Right now,
    the last batch from one epoch is the only one that can be
    smaller than a full batch.
repeat: If true, this op keep generating random samples after one
    epoch. If false, this op errors with `tf.errors.OutOfRangeError` when an
    epoch finishes.
seed: The random seed.
out: Each output is a vector of size up to batch.
)doc");

REGISTER_OP("BestStep")
    .Output("best_step: int64")
    .Attr("hist_file: string")
    .Attr("tol: float = 0.0")
    .Attr("minimize: bool = true")
    .Attr("metric: string = \"\"")
    .SetIsStateful()
    .Doc(R"doc(

Determines the best global step from a history file.

best_step: Shape [2]. best_step[0] is scalar value for best global step.
  best_step[1] is scalar value for last global step.
hist_file: A text file containing 'step score' records, or a file pattern that
    matches tf event files in the format of /path_to_file/events.out.tfevents*.
tol: Difference between previous best score and current score must be greater
than this amount to trigger update.
minimize: If the metric is being minimized. Recorded in hist_file, smaller
    scores are better if True, and bigger scores are better if False.
metric: The name of the metric being tracked.
)doc");

REGISTER_OP("BeamSearchStep")
    .Input("scores: float32")
    .Input("atten_probs: float32")
    .Input("best_scores: float32")
    .Input("cumulative_scores: float32")
    .Input("in_scores: float32")
    .Input("in_hyps: int32")
    .Input("in_prev_hyps: int32")
    .Input("in_done_hyps: string")
    .Input("in_atten_probs: float32")
    .Input("is_last_chunk: bool")
    .Input("cur_step: int32")
    .Output("out_best_scores: float32")
    .Output("out_cumulative_scores: float32")
    .Output("out_scores: float32")
    .Output("out_hyps: int32")
    .Output("out_prev_hyps: int32")
    .Output("out_done_hyps: string")
    .Output("out_atten_probs: float32")
    .Output("all_done: bool")
    .Attr("eoc_id: int = -1")
    .Attr("eos_id: int")
    .Attr("beam_size: float")
    .Attr("num_hyps_per_beam: int")
    .Attr("valid_eos_max_logit_delta: float = 5.0")
    .Attr("merge_paths: bool = false")
    .Attr("allow_empty_terminated_hyp: bool = true")
    .Attr("ensure_full_beam: bool = false")
    .Attr("force_eos_in_last_step: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      c->set_output(1, c->input(3));
      c->set_output(2, c->input(4));
      c->set_output(3, c->input(5));
      c->set_output(4, c->input(6));
      c->set_output(5, c->input(7));
      c->set_output(6, c->input(8));
      c->set_output(7, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(

Move forward one step in beam search.

Let "b" be the number of beams, "k" be the number hyps in each beam, "t" be the
maximum decoding steps.

The following data structures are allocated before the first decoding step and
are passed along from cur step to the next step:

in_scores
    A tensor of shape [t, b * k]. in_scores[i, j] is the local
    score of the j-th hyp at the i-th decoding step.
in_hyps
    A tensor of shape [t, b * k]. in_hyps[i, j] is the token id of the
    j-th hyp at the i-th decoding step.
in_prev_hyps
    A tensor of shape [t, b * k]. in_prev_hyps[i, j] stores a
    pointer of the j-th hyp at time step i to the hyp at previous timestep
    (i - 1). 0 <= in_prev_hyps[i, j] < b * k.
in_done_hyps
    A tensor of shape [t, b * k]. in_done_hyps[i, j] can be either an
    empty string, or a serialized Hypothesis proto. Terminated hyps are removed
    from the beam and are moved to the corresponding in_done_hyps slot.
in_atten_probs
    A tensor of shape [t, b * k, s_len]. in_atten_probs[i, j, ...]
    is the attention probs over the source words for the j-th hyp at the i-th
    timestep.

Those tensors are modified (with content for the cur_step timestep being filled
in) within this op invocation and are passed to the corresponding output
tensors.

is_last_chunk: A tensor of shape [b * k]. Used by neural transducer, determine
    whether the current hypothesis reaches the last chunk and should treat the
    next end-of-chunk symbol as end-of-sentence.
scores: A matrix of shape [b * k, vocab_size], where b is the number of
    active beams, and k is the number of hyps in each beam. Local scores for the
    current timestep.
atten_probs: A matrix of shape [b * k, source_len]. Attention probabilities
    for the current timestep.
best_scores: A vector of size [b], best scores of terminated hyps so far in
    each of the beams.
cumulative_scores: A vector of size [b * k]. The cumulative score of each
    active hyp before the current step.
in_scores: As explained above.
in_hyps: As explained above.
in_prev_hyps: As explained above.
in_done_hyps: As explained above.
in_atten_probs: As explained above.
cur_step: Current step id.
out_best_scores:
    Updated best scores for each of the beams.
out_cumulative_scores:
    A vector of size [b * k]. The cumulative score of the new hyps after the
    current decoding step.
out_scores:
    As explained above.
out_hyps:
    As explained above.
out_prev_hyps:
    As explained above.
out_done_hyps:
    As explained above.
out_atten_probs:
    As explained above.
all_done:
    A scalar, whether decoding should terminate for all beams.
eoc_id: Token id of the special end of chunk token.
eos_id: Token id of the special end of sequence token.
beam_size: Search terminates if the delta between the scores of the active hyps
    in a beam and the best scores exceeds this threashold.
num_hyps_per_beam: Number of hyps in a beam.
valid_eos_max_logit_delta: We allow </s> to terminate a hyp only if its logit
    is no more than `valid_eos_max_logit_delta` away from the logit of the best
    candidate.
merge_paths: If true, hyps which are identical when epsilons are removed will
    be combined into a single hyp.  The probability for that combined hyp will
    be the sum of the probabilities of the component hyps.  This can only be
    applied for epsilon-emitting models (RNN-T and NT).
allow_empty_terminated_hyp: Whether it is okay to consider a hyp that consists
    only of epsilons as terminated.  By default this is true, as an
    utterance may consist of silence.  It should be set to false when EMBR
    training epsilon-emitting models (e.g., RNN-T), which are prone to emit
    all-epsilon hyps even in the absence of silence.  Note that a hyp that
    terminates in EOS is not considered empty, so this flag has no effect for
    non-epsilon-emitting models.
ensure_full_beam: If True, we will not set the all_done output to True until we
     have found 'num_hyps_per_beam' terminated hyps AND no active hyps have a
     score within 'beam_size' of the best terminated hyp.  If False, only the
     second condition must be satisfied.  Generally this should be False unless
     beam search is being run as part of minimum word error rate training.
force_eos_in_last_step: If true, then if decode does not terminate even after
    (max - 1) steps, eos symbol is injected into the result and partial
    hypotheses (with a valid eos symbol in the end) are returned. all_done
    is set to true for these partials. If false, which is the default behavior,
    empty hypothesis are returned and all_done is set to false at termination.
)doc");

REGISTER_OP("TopKTerminatedHyps")
    .Input("in_done_hyps: string")
    .Input("src_seq_lengths: int32")
    .Output("out_topk_hyps: string")
    .Attr("k: int")
    .Attr("num_hyps_per_beam: int")
    .Attr("length_normalization: float")
    .Attr("coverage_penalty: float")
    .Attr("target_seq_length_ratio: float=1.0")
    .Attr("eoc_id: int=-1")
    .Attr("merge_paths: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto batch_size = c->Dim(c->input(1), 0);
      int k;
      TF_RETURN_IF_ERROR(c->GetAttr("k", &k));
      shape_inference::DimensionOrConstant k_dim = c->UnknownDim();
      if (k > 0) {
        k_dim = k;
      }
      c->set_output(0, c->Matrix(batch_size, k_dim));
      return Status::OK();
    })
    .Doc(R"doc(

Compute the top k terminated hyps based on normalized score for each beam.

Let "b" be the number of beams, "h" be the number hyps in each beam, "t" be the
maximum decoding steps.

in_done_hyps: A tensor of shape [t, h * b]. Each string in in_done_hyps can be
    either an empty string, or a serialized Hypothesis proto. If not empty,
    in_done_hyps[t, i * num_beams + j] represents the i-th hypothesis for beam
    j that terminates at step t.
src_seq_lengths: A tensor of shape [b] of the src sequence lengths.
out_topk_hyps:
    A string tensor of shape [b, k]. topk_hyps[i: ] contains
    top k terminated hyp for beam 'i', each hyp could be either an empty string
    or a serialized `Hypothesis` proto.
k: number of highest scoring hyps to be returned for each beam.
num_hyps_per_beam: Number of hyps per beam in the input `in_done_hyps`.
length_normalization: The length normalization ratio.
coverage_penalty: The alpha value for coverage penalty.
target_seq_length_ratio: Ratio of the average target sequence length
    over the average source sequence length.
eoc_id: Token id of the special end of chunk or blank (epsilon) token. -1 means
    this model does not use epsilon.
merge_paths: If true, hyps which are identical when epsilons are removed will
    be combined into a single hyp. The probability for that combined hyp will
    be the sum of the probabilities of the component hyps. This can only be
    applied for epsilon-emitting models (RNN-T and NT).
)doc");

REGISTER_OP("UnpackHyp")
    .Input("in_hyps: string")
    .Output("out_ids: int32")
    .Output("out_seq_lens: int32")
    .Output("out_scores: float32")
    .Attr("max_seq_length: int = 0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto batch_size = c->NumElements(c->input(0));
      int k;
      TF_RETURN_IF_ERROR(c->GetAttr("max_seq_length", &k));
      shape_inference::DimensionOrConstant k_dim = c->UnknownDim();
      if (k > 0) {
        k_dim = k;
      }
      c->set_output(0, c->Matrix(batch_size, k_dim));
      c->set_output(1, c->Vector(batch_size));
      c->set_output(2, c->Vector(batch_size));
      return Status::OK();
    })
    .Doc(R"doc(
Unpacks hyps into tensors of ids, seq_len and scores.

in_hyps: A vector of serialized `Hypothesis` protos.
out_ids:
    Output sequences, a matrix of shape (batch_size, max_seq_length).
    Sequences shorter than max_seq_length are padded with 0s. If max_seq_length is 0, derive it from the longest sequence in input_hyps.
out_seq_lens:
    Length of each of the output sequence, a vector of size `batch_size`.
out_scores:
    Scores for each of the output sequence, a vector of `batch_size`.
)doc");

REGISTER_OP("HypsFromBeamSearchOuts")
    .Input("hyps: int32")
    .Input("prev_hyps: int32")
    .Input("done_hyps: bool")
    .Input("scores: T")
    .Input("atten_probs: T")
    .Input("eos_scores: T")
    .Input("eos_atten_probs: T")
    .Output("out_hyps: string")
    .Attr("T: {float, bfloat16} = DT_FLOAT")
    .Attr("eos_id: int")
    .Attr("num_hyps_per_beam: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

Generates `Hypothesis` protos from output of a beam search step.

hyps: A tensor of shape [t, b * k] with ids of the token selected.
prev_hyps: A tensor of shape [t, b * k] with index to the previous hyps which
    was selected.
done_hyps: A boolean tensor of shape [t, b * k] where value indicates if hyps
    was terminated.
scores: A tensor of shape [t, b * k]. in_scores[i, j] is the local score of
    the j-th hyp at the i-th decoding step.
atten_probs:  A tensor of shape [t, b * k, s_len]. atten_probs[i, j, ...]
    is the attention probs over the source words for the j-th hyp at the i-th
    timestep.
eos_scores: A tensor of shape [t, b * k]. eos_scores[i, j] is the local
    score of the EOS token at the j-th hyp at the i-th decoding step.
eos_atten_probs: A tensor of shape [t, b * k, s_len].
    eos_atten_probs[i, j, ...] is the attention probs over the source words
    for the j-th terminated hyp at the i-th timestep.
out_hyps: A tensor of shape [t, b * k] with terminated hyps.
eos_id: Token id of the special end of sequence token.
num_hyps_per_beam: Number of hyps per beam.
)doc");

REGISTER_OP("CachedCall")
    .Output("output: T")
    .Attr("f: func")
    .Attr("T: list(type) >= 0")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Invokes function f once and memorize its output.

output: A list of output tensors whose types are T.
f: A function that returns a list of tensors (T).
)doc");

REGISTER_OP("VocabTokenToId")
    .Input("token: string")
    .Output("id: int32")
    .Attr("vocab: list(string)")
    .Attr("load_token_ids_from_vocab: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Looks up the token in the vocab and return its id.

token: A scalar or list of strings.
id: A scalar or list of ints.
vocab: A list of strings.
load_token_ids_from_vocab: Whether token ids are present in vocab (i.e. vocab
    contains two colums, one for IDs and one for words).  If false, line numbers
    are used.
)doc");

REGISTER_OP("VocabIdToToken")
    .Input("id: int32")
    .Output("token: string")
    .Attr("vocab: list(string)")
    .Attr("load_token_ids_from_vocab: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Looks up the token at the given id from a vocab.

id: A scalar or list of ints.
token: A scalar or list of strings.
vocab: A list of strings.
load_token_ids_from_vocab: Whether token ids are present in vocab (i.e. vocab
    contains two colums, one for IDs and one for words).  If false, line numbers
    are used.
)doc");

REGISTER_OP("TokenInVocab")
    .Input("token: string")
    .Output("result: bool")
    .Attr("vocab: list(string)")
    .Attr("load_token_ids_from_vocab: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Checks whether the provided token is in the vocab.

token: A scalar or list of strings.
result: A scalar or list of bools.
vocab: A list of strings.
load_token_ids_from_vocab: Whether token ids are present in vocab (i.e. vocab
    contains two colums, one for IDs and one for words).  If false, line numbers
    are used.
)doc");

REGISTER_OP("AsciiToTokenId")
    .Input("labels: string")
    .Output("token_ids: int32")
    .Output("target_ids: int32")
    .Output("paddings: float")
    .Attr("append_eos: bool = true")
    .Attr("maxlen: int = 300")
    .Attr("pad_to_maxlen: bool = true")
    .Doc(R"doc(
Converts ASCII label strings into token ids.

labels: A vector of shape [batch].
token_ids: A matrix of shape [batch, maxlen].
    token_ids[i, j] is the i-th sample's j-th token id.
    token_ids[i, 0] is always <s>.
target_ids: A matrix of shape [batch, maxlen].
    target_ids[i, j] is the i-th sample's j-th prediction label id.
paddings: A matrix of shape [batch, maxlen].
    paddings[i, j] == 1.0 indicates that i-th training example'
    j-th target token is padded and should be ignored.
append_eos: Whether to append </s> at the end and treat it as a non-padded
    label.
maxlen: an integer, sequence length of the output tensors.
pad_to_maxlen: Whether to pad the output to maxlen.
)doc");

REGISTER_OP("StrToVocabTokens")
    .Input("labels: string")
    .Output("token_ids: int32")
    .Output("target_ids: int32")
    .Output("paddings: float")
    .Attr("append_eos: bool = true")
    .Attr("maxlen: int = 300")
    .Attr("pad_to_maxlen: bool = true")
    .Attr("vocab_filepath: string")
    .Attr("load_token_ids_from_vocab: bool = true")
    .Attr("delimiter: string = ' '")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto batch_size = c->Dim(c->input(0), 0);
      int maxlen;
      TF_RETURN_IF_ERROR(c->GetAttr("maxlen", &maxlen));
      c->set_output(0, c->Matrix(batch_size, maxlen));
      c->set_output(1, c->Matrix(batch_size, maxlen));
      c->set_output(2, c->Matrix(batch_size, maxlen));
      return Status::OK();
    })
    .Doc(R"doc(
Tokenizes string into white space separated tokens according to a vocab file.

labels: A vector of shape [batch].
token_ids: A matrix of shape [batch, maxlen].
    token_ids[i, j] is the i-th sample's j-th token id.
    token_ids[i, 0] is always <s>.
target_ids: A matrix of shape [batch, maxlen].
    target_ids[i, j] is the i-th sample's j-th prediction label id.
paddings: A matrix of shape [batch, maxlen].
    paddings[i, j] == 1.0 indicates that i-th training example's
    j-th target token is padded and should be ignored.
append_eos: Whether to append </s> at the end and treat it as a non-padded
    label.
maxlen: an integer, sequence length of the output tensors.
pad_to_maxlen: Whether to pad the output to maxlen.
vocab_filepath: a string, filepath to the vocab file.
load_token_ids_from_vocab: Whether token ids are present in vocab (i.e. vocab
    contains two colums, one for IDs and one for words).  If false, line numbers
    are used.
delimiter: The delimiter to split the labels to tokens by.
)doc");

REGISTER_OP("IdToAscii")
    .Input("token_ids: int32")
    .Input("seq_lengths: int32")
    .Output("sequence: string")
    .Doc(R"doc(
Converts sequences from token ids to actual ASCII tokens.

token_ids: A matrix of shape [batch, seq_len].
seq_lengths: A vector of shape [batch]. seq_lengths[i] is the length of the
    i-th sequence. Only the first seq_lengths[i] tokens in token_ids[i] are
    valid tokens for the i-th sequence.
sequence: A vector of shape [batch]. The converted string sequence.
)doc");

REGISTER_OP("NgramIdToToken")
    .Input("token_ids: int32")
    .Input("seq_lengths: int32")
    .Output("sequences: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Attr("ngram_vocab_filepath: string")
    .Attr("ngram_separator: string = \"\"")
    .Doc(R"doc(
Converts sequences from token ids to actual tokens.

token_ids: A matrix of shape [batch, seq_len].
seq_lengths: A vector of shape [batch]. seq_lengths[i] is the length of the
    i-th sequence. Only the first seq_lengths[i] tokens in token_ids[i] are
    valid tokens for the i-th sequence.
sequences: A vector of shape [batch]. The converted string sequence.
ngram_vocab_filepath: filepath to the ngram vocab file.
ngram_separator: separator to use when joining ngrams into string.
)doc");

REGISTER_OP("BpeWordsToIds")
    .Input("labels: string")
    .Output("token_ids: int32")
    .Output("target_ids: int32")
    .Output("paddings: float")
    .Attr("append_eos: bool = true")
    .Attr("maxlen: int = 300")
    .Attr("sos_id: int = 1")
    .Attr("eos_id: int = 2")
    .Attr("tokenization_filepath: string")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      const auto batch_size = ctx->Dim(ctx->input(0), 0);
      int maxlen;
      TF_RETURN_IF_ERROR(ctx->GetAttr("maxlen", &maxlen));
      ctx->set_output(0, ctx->Matrix(batch_size, maxlen));
      ctx->set_output(1, ctx->Matrix(batch_size, maxlen));
      ctx->set_output(2, ctx->Matrix(batch_size, maxlen));
      return Status::OK();
    })
    .Doc(R"doc(
A tokenizer to convert string to BPE ids.

This op is especially convenient for mapping string text to a sequenec of word BPE
ids. The `labels` strings are tokenized via BPE. This op is typically used in conjunction with BpeIdsToWords.
As the vocabulary file it receives the mapping from each word to the series of BPE ids.
An example of lines in the vocabulary file:
...
AARON 10005,16,29
AARON'S 10005,16,3466
AARONSON 10005,16,17,447
...

The output tensor `token_ids` is a sequence of integer ids with <s> prepended.
The output tensor `target_ids` is a sequence of integer ids with </s> appended.

labels: The batch of tf.String tensors. Expected shape is [batch_size].
token_ids: The ids with <s>. The shape is [batch_size, maxlen].
target_ids: The ids with </s>. The shape is [batch_size, maxlen].
paddings: The paddings. The shape is [batch_size, maxlen].
maxlen: Maximum length of token_ids/target_ids/paddings.
tokenization_filepath: A path to a text file where each line is a word separated with space form a list of ids which are separated by ','.
)doc");

REGISTER_OP("BpeIdsToWords")
    .Input("token_ids: int32")
    .Input("seq_lengths: int32")
    .Output("sequences: string")
    .Attr("vocab_filepath: string")
    .SetShapeFn([](shape_inference::InferenceContext* ctx) {
      const auto batch_size = ctx->Dim(ctx->input(0), 0);
      ctx->set_output(0, ctx->Vector(batch_size));
      return Status::OK();
    })
    .Doc(R"doc(
A tokenizer to map BPE ids to strings.

This op is to map a sequence of integer ids to a string. The op is typically
used in conjunction with BpeWordsToIds.

The op will consume `seq_lengths` of tokens from `token_ids` and convert it to
string `sequences`. A space character will be interested inbetween tokens. We do
not filter any tokens (i.e., <s> and </s> are not treated specially).

token_ids: The ids (can include paddings; length is determined by seq_lengths). The shape is [batch_size, maxlen].
seq_lengths: The length of the ids. The shape is [batch_size].
sequences: The string sequences. The shape is [batch_size].
vocab_filepath: A path to a text file where each line is a BPE string token.
)doc");



REGISTER_OP("GenericInput")
    .Output("out: out_types")
    .INPUT_ATTRS  // Common input attributes.
    .Attr("out_types: list(type)")
    .Attr("processor: func")
    .Attr("dynamic_padding_dimensions: list(int) = []")
    .Attr("dynamic_padding_constants: list(int) = []")
    .Doc(R"doc(
Produces examples from processed from records.

out: The 1st dimension of every tensor is the batch dimension.
)doc" INPUT_DOCS
         R"doc(
out_types: A list of tensor types.
processor: A function that processes a string (one record) and returns
    a list of tensors. The last tensor must be a int32 scalar, which is
    used in conjunction with `bucket_upper_bound` to bucket all the
    samples.  The other tensors belongs to one sample. They have the
    respective `out_types`.  These tensors' first dimension are _not_ the
    batch dimension. Instead, when multiple samples are merged into a
    batch, GenericInput's implementation expand the batch dimension (dim
    0) and concatenate the corresponding tensors into one tensor.
dynamic_padding_dimensions: If not empty, must be the same length as out.
    Specifies the 0-indexed dimension to pad dynamically for each output.
    The output is padded to the longest tensor in the batch along the dimension.
    The first (0-th) dimension is _not_ the batch dimension.
dynamic_padding_constants: Must be set if `dynamic_padding_dimension` is
    provided. The constant value to use for padding.
)doc");

}  // namespace
}  // namespace tensorflow
