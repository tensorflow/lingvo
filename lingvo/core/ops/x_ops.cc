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

i-th dimension matches if and only if
  x[i] == y[i] || x[i] == -1 || y[i] == -1.

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
  epoch. If false, this op errors with tf.errors.OutOfRangeError when an epoch
  finishes.
seed: The random seed.
out: Each output is a vector of size up to batch.
)doc");

REGISTER_OP("BestStep")
    .Output("best_step: int64")
    .Attr("hist_file: string")
    .Attr("tol: float = 0.0")
    .SetIsStateful()
    .Doc(R"doc(

Determines the best global step from a history file.

best_step: Scalar value for best global step.
hist_file: Text file containing 'step score' records; lower scores are better.
tol: Difference between previous best score and current score must be greater
than this amount to trigger update.
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
    .Input("lm_log_probs: float32")
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
    .Attr("lm_weight: float = 0.0")
    .Attr("merge_paths: bool = false")
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

in_scores: A tensor of shape [t, b * k]. in_scores[i, j] is the local
    score of the j-th hyp at the i-th decoding step.
in_hyps: A tensor of shape [t, b * k]. "in_hyps[i, j]" is the token id of the
    j-th hyp at the i-th decoding step.
in_prev_hyps: A tensor of shape [t, b * k]. "in_prev_hyps[i, j]" stores a
    pointer of the j-th hyp at time step i to the hyp at previous timestep
    (i - 1).  0 <= in_prev_hyps[i, j] < b * k.
in_done_hyps: A tensor of shape [t, b * k]. in_done_hyps[i, j] can be either an
    empty string, or a serialized Hypothesis proto. Terminated hyps are removed
    from the beam and are moved to the corresponding in_done_hyps slot.
in_atten_probs: A tensor of shape [t, b * k, s_len]. "in_atten_probs[i, j, ...]
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
lm_log_probs: A matrix of shape [b * k, vocab_size], where b is the number of
    active beams, and k is the number of hyps in each beam. Local scores for the
    current timestep according to a language model.  These scores will be used
    to adjust scores of the top k hyps if lm_weight is nonzero (see
    'lm_weight' below).
out_best_scores: Updated best scores for each of the beams.
out_cumulative_scores: A vector of size [b * k]. The cumulative score of the new
    hyps after the current decoding step.
out_scores: As explained above.
out_hyps: As explained above.
out_prev_hyps: As explained above.
out_done_hyps: As explained above.
out_atten_probs: As explained above.
all_done: A scalar, whether decoding should terminate for all beams.
eoc_id: Token id of the special end of chunk token.
eos_id: Token id of the special end of sequence token.
beam_size: Search terminates if the delta between the scores of the active hyps
    in a beam and the best scores exceeds this threashold.
num_hyps_per_beam: Number of hyps in a beam.
valid_eos_max_logit_delta: We allow </s> to terminate a hyp only if its logit
    is no more than 'valid_eos_max_logit_delta' away from the logit of the best
    candidate.
lm_weight: A scalar specifying how much weight to place on 'lm_log_probs' when
    determining the scores of the top k hyps.  If lm_weight is zero, the local
    score of each hyp is the score of the chosen word according to 'scores'.
    Otherwise, the local score is a linear combination of the chosen word
    according to 'scores' and 'lm_log_probs', with:
    effective score = 'scores' + ('lm_log_probs' * lm_ weight).
    Note that this rescoring is done only after the top k hyps have been chosen
    using 'scores' alone, such that 'lm_log_probs' do not actually change what
    the top k hyps are, in a given step.  Global score remains the sum of local
    scores.
merge_paths: If true, hyps which are identical when epsilons are removed will
    be combined into a single hyp.  The probability for that combined hyp will
    be the sum of the probabilities of the component hyps.  This can only be
    applied for epsilon-emitting models (RNN-T and NT).
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

in_done_hyps: A tensor of shape [t, b * h]. in_done_hyps[i, j] can be either an
    empty string, or a serialized Hypothesis proto. The non-empty hyps in
    in_done_hyps are terminated hyps.
src_seq_lengths: A tensor of shape [b] of the src sequence lengths.
out_topk_hyps: A string tensor of shape [b, k]. topk_hyps[i: ] contains top k
    terminated hyp for beam 'i', each hyp could be either an empty string or
    a serialized Hypothesis proto.
k: number of highest scoring hyps to be returned for each beam.
num_hyps_per_beam: Number of hyps per beam in the input 'in_done_hyps'.
length_normalization: The length normalization ratio.
coverage_penalty: The alpha value for coverage penalty.
target_seq_length_ratio: Ratio of the average target sequence length
    over the average source sequence length.
eoc_id: Token id of the special end of chunk or blank (epsilon) token.  -1 means
    this model does not use epsilon.
merge_paths: If true, hyps which are identical when epsilons are removed will
    be combined into a single hyp.  The probability for that combined hyp will
    be the sum of the probabilities of the component hyps.  This can only be
    applied for epsilon-emitting models (RNN-T and NT).
)doc");

REGISTER_OP("UnpackHyp")
    .Input("in_hyps: string")
    .Output("out_ids: int32")
    .Output("out_seq_lens: int32")
    .Output("out_scores: float32")
    .Attr("max_seq_length: int")
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

in_hyps: A vector of serialized Hypothesis protos.
out_ids: Output sequences, a matrix of shape (batch_size, max_seq_length).
    Sequences shorter than max_seq_length are padded with 0s.
out_seq_lens: Length of each of the output sequence, a vector of size
    batch_size.
out_scores: Scores for each of the output sequence, a vector of batch_size.
)doc");

REGISTER_OP("HypsFromBeamSearchOuts")
    .Input("hyps: int32")
    .Input("prev_hyps: int32")
    .Input("done_hyps: bool")
    .Input("scores: float")
    .Input("atten_probs: float")
    .Input("eos_scores: float")
    .Input("eos_atten_probs: float")
    .Output("out_hyps: string")
    .Attr("eos_id: int")
    .Attr("num_hyps_per_beam: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(

Generates Hypothesis protos from output of a beam search step.

hyps: A tensor of shape [t, b * k] with ids of the token selected.
prev_hyps: A tensor of shape [t, b * k] with index to the previous hyps which
  was selected.
done_hyps: A boolean tensor of shape [t, b * k] where value indicates if hyps
  was terminated.
scores: A tensor of shape [t, b * k]. in_scores[i, j] is the local score of the
  j-th hyp at the i-th decoding step.
atten_probs:  A tensor of shape [t, b * k, s_len]. "atten_probs[i, j, ...]
  is the attention probs over the source words for the j-th hyp at the i-th
  timestep.
eos_scores: A tensor of shape [t, b * k]. in_scores[i, j] is the local score of
  the EOS token at the j-th hyp at the i-th decoding step.
eos_atten_probs:  A tensor of shape [t, b * k, s_len].
  "eos_atten_probs[i, j, ...] is the attention probs over the source words for
  the j-th terminated hyp at the i-th timestep.
out_hyps: A tensor of shape [t, b * k] with terminated Hyps.
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

REGISTER_OP("GenericInput")
    .Output("out: out_types")
    .Attr("out_types: list(type)")
    .Attr("processor: func")
    .INPUT_ATTRS  // Common input attributes.
    .Doc(R"doc(
Produces examples from processed from records.

This op does not do any paddings. If the samples need to be padded,
the 'processor' needs to pad the tensor before handing off to
GenericInput.

This op also ignores bucketing scheme and assumes there is only 1
bucket.

out: A list of tensors of the given types. The 1st dimension
  of every tensor is the batch dimension.
)doc"             // Common input attributes
         INPUT_DOCS
         R"doc(
out_types: A list of tensor types.
processor: A function that processes a string (one record) and returns
  a list of tensors. The last tensor must be a int32 scalar, which is
  used in conjunction with bucket_upper_bound to bucket all the
  samples.  The other tensors belongs to one sample. They have the
  respective out_types.  These tensors' first dimension are _not_ the
  batch dimension. Instead, when multiple samples are merged into a
  batch, GenericInput's implementation expand the batch dimension (dim
  0) and concatenate the corresponding tensors into one tensor.
)doc");

}  // namespace
}  // namespace tensorflow
