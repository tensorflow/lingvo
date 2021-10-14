/* Copyright 2018 The TensorFlow Authors. All Rights Reserved

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
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "x_ops_helper.h"

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
    .Input("scores: float32")                  // 0
    .Input("atten_probs: float32")             // 1
    .Input("best_scores: float32")             // 2
    .Input("cumulative_scores: float32")       // 3
    .Input("in_scores: float32")               // 4
    .Input("in_hyps: int32")                   // 5
    .Input("in_prev_hyps: int32")              // 6
    .Input("in_done_hyps: string")             // 7
    .Input("in_atten_probs: float32")          // 8
    .Input("is_last_chunk: bool")              // 9
    .Input("cur_step: int32")                  // 10
    .Output("out_best_scores: float32")        // 0
    .Output("out_cumulative_scores: float32")  // 1
    .Output("out_scores: float32")             // 2
    .Output("out_hyps: int32")                 // 3
    .Output("out_prev_hyps: int32")            // 4
    .Output("out_done_hyps: string")           // 5
    .Output("out_atten_probs: float32")        // 6
    .Output("all_done: bool")                  // 7
    .Attr("eoc_id: int = -1")
    .Attr("eos_id: int")
    .Attr("beam_size: float")
    .Attr("num_hyps_per_beam: int")
    .Attr("valid_eos_max_logit_delta: float = 5.0")
    .Attr("local_eos_threshold: float = -100.0")
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
See BeamSearchStepV2 below. This op is identical except that it does not support
`beam_independence`.

This exists to support backward-compatibility of exported graphs only. Please
use BeamSearchStepV2 below.
)doc");

REGISTER_OP("BeamSearchStepV2")
    .Input("scores: float32")                  // 0
    .Input("atten_probs: float32")             // 1
    .Input("best_scores: float32")             // 2
    .Input("cumulative_scores: float32")       // 3
    .Input("in_scores: float32")               // 4
    .Input("in_hyps: int32")                   // 5
    .Input("in_prev_hyps: int32")              // 6
    .Input("in_done_hyps: string")             // 7
    .Input("in_atten_probs: float32")          // 8
    .Input("in_beam_done: bool")               // 9
    .Input("is_last_chunk: bool")              // 10
    .Input("cur_step: int32")                  // 11
    .Output("out_best_scores: float32")        // 0
    .Output("out_cumulative_scores: float32")  // 1
    .Output("out_scores: float32")             // 2
    .Output("out_hyps: int32")                 // 3
    .Output("out_prev_hyps: int32")            // 4
    .Output("out_done_hyps: string")           // 5
    .Output("out_atten_probs: float32")        // 6
    .Output("out_beam_done: bool")             // 7
    .Output("all_done: bool")                  // 8
    .Attr("eoc_id: int = -1")
    .Attr("eos_id: int")
    .Attr("beam_size: float")
    .Attr("num_hyps_per_beam: int")
    .Attr("valid_eos_max_logit_delta: float = 5.0")
    .Attr("local_eos_threshold: float = -100.0")
    .Attr("merge_paths: bool = false")
    .Attr("allow_empty_terminated_hyp: bool = true")
    .Attr("ensure_full_beam: bool = false")
    .Attr("force_eos_in_last_step: bool = false")
    .Attr("force_eos_in_top_k: bool = false")
    .Attr("force_last_chunk_eoc_in_top_k: bool = false")
    .Attr("merged_topk_buffer_size_factor: int = 2")
    .Attr("beam_independence: bool = false")
    .Attr("atten_vecs_in_hypothesis_protos: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      c->set_output(1, c->input(3));
      c->set_output(2, c->input(4));
      c->set_output(3, c->input(5));
      c->set_output(4, c->input(6));
      c->set_output(5, c->input(7));
      c->set_output(6, c->input(8));
      c->set_output(7, c->input(9));
      c->set_output(8, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Move forward one step in beam search.

Let "b" be the number of beams, "k" be the number hyps in each beam, "t" be the
maximum decoding steps.

The following data structures are allocated before the first decoding step and
are passed along from cur step to the next step:

in_scores
    A tensor of shape [t, k * b]. in_scores[i, j] is the local
    score of the j-th hyp at the i-th decoding step.
in_hyps
    A tensor of shape [t, k * b]. in_hyps[i, j] is the token id of the
    j-th hyp at the i-th decoding step.
in_prev_hyps
    A tensor of shape [t, k * b]. in_prev_hyps[i, j] stores a
    pointer of the j-th hyp at time step i to the hyp at previous timestep
    (i - 1). 0 <= in_prev_hyps[i, j] < k * b.
in_done_hyps
    A tensor of shape [t, k * b]. in_done_hyps[i, j] can be either an
    empty string, or a serialized Hypothesis proto. Terminated hyps are removed
    from the beam and are moved to the corresponding in_done_hyps slot.
in_atten_probs
    A tensor of shape [t, k * b, s_len]. in_atten_probs[i, j, ...]
    is the attention probs over the source words for the j-th hyp at the i-th
    timestep.
in_beam_done
    A tensor of shape [b] of bools, whether each individual beam is done.
    See attr `beam_independence`.

Those tensors are modified (with content for the cur_step timestep being filled
in) within this op invocation and are passed to the corresponding output
tensors.

is_last_chunk: A tensor of shape [k * b]. Used by neural transducer, determine
    whether the current hypothesis reaches the last chunk and should treat the
    next end-of-chunk symbol as end-of-sentence.
scores: A matrix of shape [k * b, vocab_size], where b is the number of
    active beams, and k is the number of hyps in each beam. Local scores for the
    current timestep.
atten_probs: A matrix of shape [k * b, source_len]. Attention probabilities
    for the current timestep.
best_scores: A vector of size [b], best scores of terminated hyps so far in
    each of the beams.
cumulative_scores: A vector of size [k * b]. The cumulative score of each
    active hyp before the current step.
in_scores: As explained above.
in_hyps: As explained above.
in_prev_hyps: As explained above.
in_done_hyps: As explained above.
in_atten_probs: As explained above.
in_beam_done: A vector of [b], whether each beam was previously done. See
    attr `beam_independence`.
cur_step: Current step id.
out_best_scores:
    Updated best scores for each of the beams.
out_cumulative_scores:
    A vector of size [k * b]. The cumulative score of the new hyps after the
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
out_beam_done:
    A vector of size [b], whether each beam is done after this step is taken.
    `all_done` below is the logical AND of out_beam_done over all beams.
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
local_eos_threshold: We allow </s> to terminate a hyp if the local score for
    </s> is greater than local_eos_threshold.
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
force_eos_in_top_k: Whether to always consider the eos token to be among the top
    k tokens for every step. When False, hyps can only terminate if the eos
    token is part of the top k. Note that valid_eos_max_logit_delta and
    local_eos_threshold always apply regardless of this.
force_last_chunk_eoc_in_top_k: Whether to always consider the last chunk eoc
    token to be among the top k tokens. This is effective only when decoding
    has reached the last frame of input. When True, hyps can terminate at the
    last frame by eoc even if the eoc score is not high enough to enter the
    top k. Note that p.valid_eos_max_logit_delta and p.local_eos_threshold
    always apply regardless of this.
merged_topk_buffer_size_factor: The buffer size factor when pruning the per
    hyp top-k extensions to form the per beam top-k extensions. If this factor
    is set to greater than or equal num_hyps_per_beam + 2 when eoc_id >= 0,
    there will be no pruning before all possible path mergings are performed
    (if merge_paths=True). To be memory efficient (i.e., to maintain less hyps
    during pruning), a reasonable value is 2.
beam_independence: When enabled, this step will become a no-op for beam_id if
    and only if in_beam_done[beam_id] == True.
atten_vecs_in_hypothesis_protos: Whether to populate the atten_vecs fields in
    the returned Hypothesis protos.
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
    .Attr("eoc_id: int=-1")  // unused and deprecated
    .Attr("merge_paths: bool = false")  // unused and deprecated
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
out_topk_hyps: A string tensor of shape [b, k]. topk_hyps[i: ] contains
    top k terminated hyp for beam 'i', each hyp could be either an empty string
    or a serialized `Hypothesis` proto.
k: number of highest scoring hyps to be returned for each beam.
num_hyps_per_beam: Number of hyps per beam in the input `in_done_hyps`.
length_normalization: The length normalization ratio.
coverage_penalty: The alpha value for coverage penalty.
target_seq_length_ratio: Ratio of the average target sequence length
    over the average source sequence length.
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
    .Attr("fix_hyp_atten_vecs: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

Generates `Hypothesis` protos from output of a beam search step.

hyps: A tensor of shape [t, k * b] with ids of the token selected.
prev_hyps: A tensor of shape [t, k * b] with index to the previous hyps which
    was selected.
done_hyps: A boolean tensor of shape [t, k * b] where value indicates if hyps
    was terminated.
scores: A tensor of shape [t, k * b]. in_scores[i, j] is the local score of
    the j-th hyp at the i-th decoding step.
atten_probs:  A tensor of shape [t, k * b, s_len]. atten_probs[i, j, ...]
    is the attention probs over the source words for the j-th hyp at the i-th
    timestep.
eos_scores: A tensor of shape [t, k * b]. eos_scores[i, j] is the local
    score of the EOS token at the j-th hyp at the i-th decoding step.
eos_atten_probs: A tensor of shape [t, k * b, s_len].
    eos_atten_probs[i, j, ...] is the attention probs over the source words
    for the j-th terminated hyp at the i-th timestep.
out_hyps: A tensor of shape [t, k * b] with terminated hyps.
eos_id: Token id of the special end of sequence token.
num_hyps_per_beam: Number of hyps per beam.
fix_hyp_atten_vecs: Obsolete and unused.
)doc");

REGISTER_OP("TopKFromBeamSearchOuts")
    .Input("hyps: int32")                    // 0
    .Input("prev_hyps: int32")               // 1
    .Input("done_hyps: bool")                // 2
    .Input("cumulative_scores: T")           // 3
    .Input("eos_scores: T")                  // 4
    .Input("scores: T")                      // 5
    .Input("atten_probs: T")                 // 6
    .Input("eos_atten_probs: T")             // 7
    .Input("cumulative_atten_probs: T")      // 8
    .Input("length_normalization: float32")  // 9
    .Input("coverage_penalty: float32")      // 10
    .Output("out_ids: int32")                // 0
    .Output("out_seq_lens: int32")           // 1
    .Output("out_scores: float32")           // 2
    .Output("topk_hyps: string")             // 3
    .Attr("T: {float, bfloat16} = DT_FLOAT")
    .Attr("num_hyps_per_beam: int")
    .Attr("max_seq_length: int")
    .Attr("eos_id: int = 2")
    .Attr("target_seq_length_ratio: float = 1.0")
    .Attr("populate_topk_hyps: bool = false")
    .Doc(R"doc(
Compute tensors of ids, seq_len and scores from outputs of beam search steps.

This op is able to combine the work of 3 ops that are typically called
consecutively together (HypsFromBeamSearchOuts, TopKTerminatedHyps,
UnpackHyp) into one.

When we have a dimension of size b * k (of all the hyps), there are two ways to
order it. 'div' means for beam n, hyp j will have index (n * k + j). Conversely,
j = index % k, n = index // k. 'mod' means for beam n, hyp j will have index
(j * b + n). Conversely, j = index // b, n = index % b. All inputs on the
(k * b) sized dimension is mod ordered. The outputs have different ordering as
indicated below.

Note that the inputs `scores`, `atten_probs`, and `eos_atten_probs` are only
used to assemble the Hypothesis protos. So if `populate_topk_hyps` is false,
these 3 inputs are unused and ignored.

hyps: A tensor of shape [t, k * b] with ids of the token selected.
prev_hyps: A tensor of shape [t, k * b] with index to the previous hyps which
    was selected. prev_hyps[i, j] should be in the range [0, k * b) and we
    should have prev_hyps[i, j] % b == j % b (i.e. they belong to the same
    beam).
done_hyps: A boolean tensor of shape [t, k * b] where value indicates if hyps
    was terminated.
cumulative_scores: A tensor of shape [t, k * b]. cumulative_scores[i, j] is the
    cumulative score of the j-th hyp at the i-th decoding step. Note that EOS
    tokens are tracked separately, hence cumulative_scores cover only
    non-terminated hyps, i.e. the i-th step is never an EOS token.
eos_scores: A tensor of shape [t, k * b]. eos_scores[i, j] is the local
    score of the EOS token at the j-th hyp at the i-th decoding step.
scores: A tensor of shape [t, k * b]. scores[i, j] is the local score of
    the j-th hyp at the i-th decoding step.
atten_probs:  A tensor of shape [t, k * b, s_len]. atten_probs[i, j, ...]
    is the attention probs over the source words for the j-th hyp at the i-th
    timestep. Only used to assemble `done_hyps` and `topk_hyps`.
eos_atten_probs: A tensor of shape [t, k * b, s_len].
    eos_atten_probs[i, j, ...] is the attention probs over the source words
    for the j-th terminated hyp at the i-th timestep. Only used to assemble
    `done_hyps` and `topk_hyps`.
cumulative_atten_probs: A tensor of shape [t, k * b, s_len].
    cumulative_atten_probs[i, j, ...] is the cumulative attention probs (
    summation along the t dimension) over the source words
    for the j-th terminated hyp at the i-th timestep. Only used when
    `coverage_penalty` is strictly positive.
length_normalization: The length normalization factor.
coverage_penalty: The coverage penalty coefficient.
out_ids:
    Output sequences, a matrix of shape (b * k, max_seq_length).
    Sequences shorter than max_seq_length are padded with 0s. div ordered.
out_seq_lens:
    Length of each of the output sequence, a vector of size (b * k).
    div ordered.
out_scores:
    Scores for each of the output sequence, a vector of (b * k). div ordered.
topk_hyps:
    A string tensor of shape [b, k]. topk_hyps[i,] contains top k terminated
    hyp for beam 'i', each hyp could be either an empty string or a serialized
    `Hypothesis` proto. When `populate_topk_hyps` is False, all strings are
    empty.
num_hyps_per_beam: Number of hyps per beam, i.e. the value of k. Required.
max_seq_length: Max output sequence length. Required.
eos_id: Token id of the special end of sequence token.
target_seq_length_ratio: ratio used when computing coverage penalty.
populate_topk_hyps: whether to populate `topk_hyps` with serialized protos. When
    False, the output `topk_hyps` is just empty string with the shape [b, k].
)doc")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Validate input tensor shapes.
      if (c->Rank(c->input(0)) != 2) {
        return errors::InvalidArgument(
            "input tensor `hyps` must have rank 2, got shape: ",
            c->DebugString(c->input(0)));
      }
      const auto t = c->Value(c->Dim(c->input(0), 0));
      const auto b_times_k = c->Value(c->Dim(c->input(0), 1));
      for (int i = 0; i < 5; ++i) {
        if (c->Rank(c->input(i)) != 2 ||
            c->Value(c->Dim(c->input(i), 0)) != t ||
            c->Value(c->Dim(c->input(i), 1)) != b_times_k) {
          return errors::InvalidArgument(
              "input[", i, "] must have shape [", t, ", ", b_times_k,
              "], got shape: ", c->DebugString(c->input(i)));
        }
      }
      bool populate_topk;
      TF_RETURN_IF_ERROR(c->GetAttr("populate_topk_hyps", &populate_topk));
      if (populate_topk) {
        if (c->Rank(c->input(5)) != 2 ||
            c->Value(c->Dim(c->input(5), 0)) != t ||
            c->Value(c->Dim(c->input(5), 1)) != b_times_k) {
          return errors::InvalidArgument(
              "input[5] `scores` must have shape [", t, ", ", b_times_k,
              "], got shape: ", c->DebugString(c->input(5)));
        }
        for (int i = 6; i < 8; ++i) {
          if (c->Rank(c->input(i)) != 3 ||
              c->Value(c->Dim(c->input(i), 0)) != t ||
              c->Value(c->Dim(c->input(i), 1)) != b_times_k) {
            return errors::InvalidArgument(
                "input[", i, "] must have shape [", t, ", ", b_times_k,
                ", ?], got shape: ", c->DebugString(c->input(i)));
          }
        }
        if (c->Value(c->Dim(c->input(6), 2)) !=
            c->Value(c->Dim(c->input(7), 2))) {
          return errors::InvalidArgument(
              "input tensors `atten_probs` and `eos_atten_probs` must have the "
              "same shape, got shapes: atten_probs.shape=",
              c->DebugString(c->input(6)),
              ", eos_atten_probs.shape=", c->DebugString(c->input(7)));
        }
      }
      for (int i : {9, 10}) {
        if (c->Rank(c->input(i)) != 0) {
          return errors::InvalidArgument(
              "input tensor ", i,
              " must have rank 0, got shape: ", c->DebugString(c->input(i)));
        }
      }

      // Infer output tensor shapes.
      int32 k;
      TF_RETURN_IF_ERROR(c->GetAttr("num_hyps_per_beam", &k));
      if (k <= 0) {
        return errors::InvalidArgument("Requires num_hyps_per_beam > 0, got: ",
                                       k);
      }
      int32 max_length;
      TF_RETURN_IF_ERROR(c->GetAttr("max_seq_length", &max_length));
      if (max_length <= 0) {
        return errors::InvalidArgument("Requires max_seq_length > 0, got: ",
                                       max_length);
      }
      shape_inference::DimensionOrConstant b = c->UnknownDim();
      if (b_times_k > 0) {
        b = b_times_k / k;
        if (b_times_k % k) {
          return errors::InvalidArgument(
              "num_hyps (b * k) does not divide num_hyps_per_beam (k):  "
              "num_hyps=",
              b_times_k, ", num_hyps_per_beam=", k);
        }
      }
      c->set_output(0, c->Matrix(b_times_k, max_length));
      c->set_output(1, c->Vector(b_times_k));
      c->set_output(2, c->Vector(b_times_k));
      c->set_output(3, c->Matrix(b, k));
      return ::tensorflow::Status::OK();
    });

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
token_ids:
    A matrix of shape [batch, maxlen].
    token_ids[i, j] is the i-th sample's j-th token id.
    token_ids[i, 0] is always <s>.
target_ids:
    A matrix of shape [batch, maxlen].
    target_ids[i, j] is the i-th sample's j-th prediction label id.
paddings:
    A matrix of shape [batch, maxlen].
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
token_ids:
    A matrix of shape [batch, maxlen].
    token_ids[i, j] is the i-th sample's j-th token id.
    token_ids[i, 0] is always <s>.
target_ids:
    A matrix of shape [batch, maxlen].
    target_ids[i, j] is the i-th sample's j-th prediction label id.
paddings:
    A matrix of shape [batch, maxlen].
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
token_ids:
    The ids with <s>. The shape is [batch_size, maxlen].
target_ids:
    The ids with </s>. The shape is [batch_size, maxlen].
paddings:
    The paddings. The shape is [batch_size, maxlen].
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
    The first (0-th) dimension is _not_ the batch dimension. A value of -1
    indicates the specified output should not be padded, eg. if the output is a
    scalar rather than a sequence.
dynamic_padding_constants: Must be set if `dynamic_padding_dimension` is
    provided. The constant value to use for padding.
)doc");

REGISTER_OP("StaticMapStringInt")
    .Input("x: string")
    .Output("y: int32")
    .Attr("keys: list(string)")
    .Attr("vals: list(int) = []")
    .Attr("unk: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Maps every element of x according a static mapping.

x: A Tensor of type string.
y: A Tensor of type int32. Same shape of x.
keys: The list of keys.
vals: The list of values. If empty, defaults to [0 .. len(keys)).
unk: The value when the key is not found.
)doc");

REGISTER_OP("StaticMapIntString")
    .Input("x: int32")
    .Output("y: string")
    .Attr("keys: list(int) = []")
    .Attr("vals: list(string)")
    .Attr("unk: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Maps every element of x according a static mapping.

x: A Tensor of type int32.
y: A Tensor of type string. Same shape of x.
keys: The list of keys. If empty, defaults to [0 .. len(keys)).
vals: The list of values.
unk: The value when the key is not found.
)doc");

REGISTER_OP("StaticMapIntInt")
    .Input("x: int32")
    .Output("y: int32")
    .Attr("keys: list(int) = []")
    .Attr("vals: list(int)")
    .Attr("unk: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Maps every element of x according a static mapping.

x: A Tensor of type int32.
y: A Tensor of type int32. Same shape of x.
keys: The list of keys. If empty, defaults to [0 .. len(keys)).
vals: The list of values.
unk: The value when the key is not found.
)doc");

REGISTER_OP("ComputePreconditioners")
    .Input("inputs: num_tensors * float32")
    .Input("exponents: num_tensors * float32")
    .Input("global_step: int32")
    .Attr("preconditioner_compute_graphdef: string")
    .Attr("keys: list(string)")
    .Attr("sync: bool = false")
    .Attr("num_tensors: int >= 1")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return Status::OK();
    })
    .Doc(R"doc(
Compute preconditioners for Shampoo optimizer.

inputs: A list of Tensors of type float32, of statistic matrices.
exponents: A list of scalar Tensors of type float32, exponent for matrix power.
global_step: A scalar Tensor of type int32 which indicates the global step.
preconditioner_compute_graphdef: A graphdef which indicates the function to run.
keys: A list of keys indicating the name of preconditioners.
sync: Boolean indicating whether to run preconditioning in synchronous mode.
num_tensors: Number of tensor inputs.
)doc");

REGISTER_OP("GetPreconditioners")
    .Input("shapes: num_tensors * Tshape")
    .Output("outputs: num_tensors * float32")
    .Output("statuses: num_tensors * bool")
    .Attr("preconditioner_compute_graphdef: string")
    .Attr("keys: list(string)")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .Attr("num_tensors: int >= 1")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<shape_inference::ShapeHandle> shapes;
      if (c->input("shapes", &shapes).ok()) {
        for (int i = 0; i < shapes.size(); ++i) {
          shape_inference::ShapeHandle out;
          TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(i, &out));
          c->set_output(i, out);
          c->set_output(shapes.size() + i, c->Scalar());
        }
      }
      return Status::OK();
    })
    .Doc(R"doc(
Get preconditioners for Shampoo optimizer.

shapes: A list of Tensors of type Tshape indicating the size of preconditioner.
outputs: A list of Tensors of type float32 which are the preconditioners.
statuses: A list of Tensors of type bool which are the preconditioner status.
preconditioner_compute_graphdef: A graphdef which indicates the function to run.
keys: A list of keys indicating the name of preconditioners.
Tshape: The data-type to use for shape.
num_tensors: Number of tensor inputs.
)doc");

REGISTER_OP("MlPerfSubwordIdToString")
    .Input("token_ids: int32")
    .Input("seq_lengths: int32")
    .Output("sequences: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Attr("vocab_filepath: string")
    .Doc(R"doc(
Converts sequences from subword token ids to strings

token_ids: A matrix of shape [batch, seq_len].
seq_lengths: A vector of shape [batch]. seq_lengths[i] is the length of the
    i-th sequence. Only the first seq_lengths[i] tokens in token_ids[i] are
    valid tokens for the i-th sequence.
sequences: A vector of shape [batch]. The converted string sequence.
vocab_filepath: filepath to the MLPerf subword vocab file.
)doc");

REGISTER_OP("PackSequences")
    .Input("src_actual_seq_len: int32")
    .Input("tgt_actual_seq_len: int32")
    .Attr("packed_batch_size: int")
    .Attr("packed_src_seq_len: int")
    .Attr("packed_tgt_seq_len: int")
    .SetIsStateful()  // TODO(navari): disable when packed_batch_size==0?
    .Output("src_segment_ids: int32")
    .Output("src_segment_pos: int32")
    .Output("src_indices_in_input: int32")
    .Output("tgt_segment_ids: int32")
    .Output("tgt_segment_pos: int32")
    .Output("tgt_indices_in_input: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int packed_batch_size, packed_src_seq_len, packed_tgt_seq_len;
      TF_RETURN_IF_ERROR(c->GetAttr("packed_batch_size", &packed_batch_size));
      shape_inference::DimensionOrConstant batch_dim = c->UnknownDim();
      if (packed_batch_size > 0) {
        batch_dim = packed_batch_size;
      }
      TF_RETURN_IF_ERROR(c->GetAttr("packed_src_seq_len", &packed_src_seq_len));
      TF_RETURN_IF_ERROR(c->GetAttr("packed_tgt_seq_len", &packed_tgt_seq_len));
      c->set_output(0, c->Matrix(batch_dim, packed_src_seq_len));
      c->set_output(1, c->Matrix(batch_dim, packed_src_seq_len));
      c->set_output(2, c->Matrix(batch_dim, packed_src_seq_len));
      c->set_output(3, c->Matrix(batch_dim, packed_tgt_seq_len));
      c->set_output(4, c->Matrix(batch_dim, packed_tgt_seq_len));
      c->set_output(5, c->Matrix(batch_dim, packed_tgt_seq_len));
      return Status::OK();
    })
    .Attr("seed: int = 0")
    .Doc(R"doc(
Produces a packing pattern for the (src, tgt) input pair with the provided
lengths, according to the given packed shape.

If only a single input sequence is used and a fixed output batch size is not
required, consider using the simpler PackSingleSequence op.

For example, the following input::

  src_actual_seq_len = [3, 2, 1]
  tgt_actual_seq_len = [4, 1, 5]
  packed_batch_size = 2
  packed_src_seq_len = 5
  packed_tgt_seq_len = 5

will result in::

  src_segment_ids = [ [1, 1, 1, 2, 2], [1, 0, 0, 0, 0] ]
  src_segment_pos = [ [0, 1, 2, 0, 1], [0, 0, 0, 0, 0] ]
  src_indices_in_input = [ [0, 0, 0, 1, 1], [2, 0, 0, 0, 0] ]
  tgt_segment_ids = [ [1, 1, 1, 1, 2], [1, 1, 1, 1, 1] ]
  tgt_segment_pos = [ [0, 1, 2, 3, 0], [0, 1, 2, 3, 4] ]
  tgt_indices_in_input = [ [0, 0, 0, 0, 1], [2, 2, 2, 2, 2] ]

The packed sequence length can be different between src and tgt. For example,
the following input::

  src_actual_seq_len = [3, 2, 1]
  tgt_actual_seq_len = [4, 1, 5]
  packed_batch_size = 2
  packed_src_seq_len = 4
  packed_tgt_seq_len = 6

will result in::

  src_segment_ids = [ [1, 1, 1, 0], [1, 1, 2, 0] ]
  src_segment_pos = [ [0, 1, 2, 0], [0, 1, 0, 0] ]
  src_indices_in_input = [ [0, 0, 0, 0], [1, 1, 2, 0] ]
  tgt_segment_ids = [ [1, 1, 1, 1, 0, 0], [1, 2, 2, 2, 2, 2] ]
  tgt_segment_pos = [ [0, 1, 2, 3, 0, 0], [0, 0, 1, 2, 3, 4] ]
  tgt_indices_in_input = [ [0, 0, 0, 0, 0, 0], [1, 2, 2, 2, 2, 2] ]

If packed_batch_size is set to 0, output will be of variable batch
size, determined by the number of row needed to pack all given inputs.

If there are too few input sequences to pack into `output_shape`, the op pads
the remaining elements in the output.

If there are too many input sequences to pack into `output_shape`, the op drops
input sequences. The dropping is done randomly uniformly on the input sequences
to not bias the distribution of sequence lengths in the packed output.

src_actual_seq_len: A tensor of shape [N], where N is the input batch size.
  This tensor contains the actual lengths for the src sequence.
tgt_actual_seq_len: A tensor of shape [N], where N is the input batch size.
  This tensor contains the actual lengths for the tgt sequence.
packed_batch_size: A scalar. The output batch size. The packed output will
  be of shape [packed_batch_size, packed_{src,tgt}_seq_len] for src and tgt,
  respectively. if this value is set to 0, output will be of variable batch
  size, determined by the number of row needed to pack all given inputs.
packed_src_seq_len: A scalar. The output sequence length for src. A src input
  with shape [N, src_input_seq_len] will be packed into an output with shape
  [packed_batch_size, packed_src_seq_len].
packed_tgt_seq_len: A scalar. The output sequence length for tgt. A tgt input
  with shape [N, tgt_input_seq_len] will be packed into an output with shape
  [packed_batch_size, packed_tgt_seq_len].
src_segment_ids:
  A tensor of shape [packed_batch_size, packed_src_seq_len]. Incrementing from 1
  to indicate each segment in the packed output for src. Zero is reserved for
  indicating padding at the end of each row.
tgt_segment_ids:
  A tensor of shape [packed_batch_size, packed_tgt_seq_len]. Incrementing from 1
  to indicate each segment in the packed output for tgt. Zero is reserved for
  indicating padding at the end of each row.
src_segment_pos:
  A tensor of shape [packed_batch_size, packed_src_seq_len]. Zero-based index to
  indicate relative position within each segment for src. Zero is also used to
  indicate padding. When needed, use `src_segment_ids` to disambiguate.
tgt_segment_pos:
  A tensor of shape [packed_batch_size, packed_tgt_seq_len]. Zero-based index to
  indicate relative position within each segment for tgt. Zero is also used to
  indicate padding. When needed, use `tgt_segment_ids` to disambiguate.
src_indices_in_input:
  A tensor of shape [packed_batch_size, packed_src_seq_len]. For each segment in
  the packed output, it contains the original (zero-based) row index of each
  segment found in `src_actual_seq_len`. Zero is also used to indicate padding.
  When needed, use `src_segment_ids` to disambiguate.
tgt_indices_in_input:
  A tensor of shape [packed_batch_size, packed_tgt_seq_len]. For each segment in
  the packed output, it contains the original (zero-based) row index of each
  segment found in `tgt_actual_seq_len`. Zero is also used to indicate padding.
  When needed, use `tgt_segment_ids` to disambiguate.
seed: Seed for random number generator, which is used when we need to drop
  excessive input sequences. If seed is zero, use completely random seed.
)doc");

REGISTER_OP("PackSingleSequence")
    .Input("input_lengths: int32")
    .Attr("max_packed_length: int")
    .Attr("require_sequential_order: bool = false")
    .Output("segment_ids: int32")
    .Output("indices_in_input: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::DimensionHandle batch_dim = c->UnknownDim();
      int max_packed_length;
      TF_RETURN_IF_ERROR(c->GetAttr("max_packed_length", &max_packed_length));
      c->set_output(0, c->Matrix(batch_dim, max_packed_length));
      c->set_output(1, c->Matrix(batch_dim, max_packed_length));
      return Status::OK();
    })
    .Doc(R"doc(
Produces a packing pattern with the provided `input_lengths`.

Examples are packed into sequences not exceeding max_packed_length. The number
of sequences in the output is dynamic. No examples are dropped.

If x and y are packed together, and if x comes before y in `input_lengths`, it
is guaranteed that x will come before y in the packed sequence. That is,
it is guaranteed that each row of `indices_in_input` is non-descending.

input_lengths: A tensor of shape [batch_size], containing actual lengths for the
  input examples. All input lengths must be no larger than `max_packed_length`.
max_packed_length: A scalar. The maximum length of a packed sequence. The output
  will have the length dimension padded to this value.
require_sequential_order: A boolean. If true, the input will be packed in order
  (fill the first output before moving to the next one). If false, the input
  examples can be reordered for better packing.
segment_ids:
  A tensor of shape [packed_batch_size, max_packed_length]. Incrementing from 1
  to indicate segments in the packed output. Zero indicates padding at the end.
indices_in_input:
  A tensor of shape [packed_batch_size, max_packed_length]. For each segment in
  the packed output, it contains the original (zero-based) index of each input
  segment.
)doc");

REGISTER_OP("ApplyPacking")
    .Input("input: T")
    .Input("padding: T")
    .Input("segment_ids: int32")
    .Input("indices_in_input: int32")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &dtype));

      if (c->Rank(c->input(1)) != 0) {
        return errors::InvalidArgument(
            "padding must be a scalar, got padding shape: ",
            c->DebugString(c->input(1)));
      }

      if (c->Rank(c->input(2)) != 2 || c->Rank(c->input(3)) != 2 ||
          c->Value(c->Dim(c->input(2), 0)) !=
              c->Value(c->Dim(c->input(3), 0)) ||
          c->Value(c->Dim(c->input(2), 1)) !=
              c->Value(c->Dim(c->input(3), 1))) {
        return errors::InvalidArgument(
            "segment_ids and indices_in_input must be "
            "matrices of the same shape, got: ",
            c->DebugString(c->input(2)), " vs. ", c->DebugString(c->input(3)));
      }

      const auto batch_size = c->Dim(c->input(2), 0);
      const auto output_length = c->Dim(c->input(2), 1);
      if (c->Rank(c->input(0)) == 1 || dtype == DT_STRING) {
        c->set_output(0, c->Vector(batch_size));
      } else {
        const shape_inference::ShapeHandle& input_shape = c->input(0);
        shape_inference::ShapeHandle output_shape;
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(input_shape, 0, batch_size, &output_shape));
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(output_shape, 1, output_length, &output_shape));
        c->set_output(0, output_shape);
      }
      return Status::OK();
    })
    .Attr("T: type")
    .Doc(R"doc(
Applies a packing pattern on the input to obtain a packed output.

The input can be either a matrix or higher dimension, in which case the output
is shorter and the elements are rearranged according to the packing pattern;
or the input can be a vector, in which case the output is a shorter vector,
where each position is the sum of the elements packed on that same row.

When T is tf.string type, only the vector input is supported and the output
joins the strings that are packed on the same row, separated by `padding`.

The inputs `segment_ids` and `indices_in_input` can be obtained from the outputs
of an `PackSequence` op (though only the src or the tgt tensors are needed).

Note that ApplyPacking is done on a per column basis (either on the src or on
the tgt), as opposed to in PackSequences, when both src and tgt columns must be
processed together within the same op.

input: A tensor of shape [N, seq_len, ...] or [N]. The input to apply the
  packing to. For tf.string typed input only a vector of shape [N] is supported.
padding: A scalar to indicate the padding value. This is typically the zero
  value of T, but may not always be the case, e.g. when the input is a paddings
  tensor, in which case caller should set padding=1.
  For tf.string typed input, padding is used as a separator to join all the
  strings on the same row in the output.
segment_ids: A rank 2 tensor of shape [M, packed_seq_len].
indices_in_input: A rank 2 tensor of shape [M, packed_seq_len].

output:
  A tensor of shape [M, packed_seq_len, ...], where the later dimensions match
  those of `input`. For tf.string typed input, the output is a vector of strings
  of shape [M].
)doc");

REGISTER_OP("Mass")
    .Input("ids: int32")
    .Input("weights: float32")
    .Input("actual_seq_len: int32")
    .Attr("mask_id: int")
    .Attr("mask_ratio: float = 0.5")
    .Attr("mask_minlen: int = 0")
    .Attr("span_len: int = 100000")
    .Attr("random_start_prob: float = 0.6")
    .Attr("keep_prob: float = 0.1")
    .Attr("rand_prob: float = 0.1")
    .Attr("mask_prob: float = 0.8")
    // TODO(alisonlui): This flag is rarely used; remove after verification.
    .Attr("mask_target: bool = True")
    .Attr("vocab_size: int")
    .Attr("first_unreserved_id: int = 4")
    .Output("src_ids: int32")
    .Output("tgt_ids: int32")
    .Output("tgt_labels: int32")
    .Output("tgt_weights: float32")
    .Doc(R"doc(
Applies masking to implement MASS.

ids: Tensor of shape [batch_size, max_seq_len] containing the token ids.
  Should include EOS token </s>.
weights: Tensor of shape [batch_size, max_seq_len].
actual_seq_len: Tensor of shape [batch_size].

mask_id: The id to use for the mask token.
mask_ratio: Proportion of src to mask.
mask_minlen: Skip sentences too short to mask at least this many tokens.
span_len: Split mask_len into segments of this size and randomly distribute
those across the src.
random_start_prob: The probability that the placement of masked segments will be
  entirely random. The remaining cases are split evenly between masking at the
  beginning and at the end of the src.
keep_prob: The probability that a token to be masked will be unchanged.
  `keep_prob + rand_prob + mask_prob` must sum to 1.
rand_prob: The probability that a token to be masked will be replaced with a
  random token in the vocab. `keep_prob + rand_prob + mask_prob` must sum to 1.
mask_prob: The probability that a token to be masked will be replaced with the
  mask_id. `keep_prob + rand_prob + mask_prob` must sum to 1.
mask_target: whether to mask the target (the mask will be the inverse of that of
  the src).
vocab_size: Vocab size used when selecting a random token to replace a masked
  token.
first_unreserved_id: Tokens greater than or equal to this may be selected at
  random to replace a masked token.

src_ids:
  Masked ids. E.g. `s1 s2 s3 m m </s>`
tgt_ids:
  Right-shifted ids with BOS token added, where the mask is the
  positional inverse of that of the source unless mask_target=False.
  E.g. `m m m s3 s4 m`
tgt_labels:
  E.g. `s1 s2 s3 s4 s5 </s>`
tgt_weights:
  weights are zeroed wherever the target is masked.
)doc");

}  // namespace
}  // namespace tensorflow
