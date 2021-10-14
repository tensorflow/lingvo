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

#include "lingvo/core/ops/beam_search_step_op_kernels.h"

#include <cmath>

#include "absl/strings/string_view.h"
#include "lingvo/core/ops/hyps.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace lingvo {

namespace {
constexpr int kNumWorkers = 8;
}  // namespace

bool IdsMatchUpToIndex(const std::vector<int>& cur_hyp_ids,
                       const std::vector<int>& other_hyp_ids, const int index) {
  DCHECK_LE(index, cur_hyp_ids.size());
  DCHECK_LE(index, other_hyp_ids.size());
  return std::equal(cur_hyp_ids.begin(), cur_hyp_ids.begin() + index,
                    other_hyp_ids.begin());
}

bool IsDuplicateHyp(const Hyp& cur_hyp, const Hyp& other_hyp,
                    const int epsilon_id) {
  const std::vector<int>& cur_hyp_ids = cur_hyp.prev_labels;
  const std::vector<int>& other_hyp_ids = other_hyp.prev_labels;
  // Note word_id refers to id of current label, which could be grapheme,
  // phoneneme, wordpiece, etc.
  if (cur_hyp.word_id == other_hyp.word_id) {
    // If the cur step is the same (epsilon or otherwise), just need to compare
    // prev ids which already has epsilons stripped.
    return (
        cur_hyp_ids.size() == other_hyp_ids.size() &&
        IdsMatchUpToIndex(cur_hyp_ids, other_hyp_ids, other_hyp_ids.size()));
  } else if (cur_hyp.word_id == epsilon_id) {
    // If exactly one of the hyps has a cur step of epsilon, then need to
    // compare that hyp's final prev id to other hyp's current step id,
    // then compare the rest of the prev ids.
    return (
        cur_hyp_ids.size() == other_hyp_ids.size() + 1 &&
        cur_hyp_ids[cur_hyp_ids.size() - 1] == other_hyp.word_id &&
        IdsMatchUpToIndex(cur_hyp_ids, other_hyp_ids, other_hyp_ids.size()));
  } else if (other_hyp.word_id == epsilon_id) {
    return (other_hyp_ids.size() == cur_hyp_ids.size() + 1 &&
            other_hyp_ids[other_hyp_ids.size() - 1] == cur_hyp.word_id &&
            IdsMatchUpToIndex(other_hyp_ids, cur_hyp_ids, cur_hyp_ids.size()));
  } else {
    // If the cur step is not the same for the two hyps and neither is an
    // epsilon then this cannot be a duplicate hyp.
    return false;
  }
}

float LogSumExp(float a, float b) {
  const float m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

#ifdef __AVX__
// AVX version all_less_than.
bool all_less_than(const float* p, float threshold) {
  __m256 kth_logp = _mm256_set1_ps(threshold);
  __m256 logp_vec = _mm256_loadu_ps(p);
  // Skip this 8 elements if all of them are worst than 'kth_logp'.
  // 'OQ' in '_CMP_LE_OQ' means comparison against NaN fails
  // quietly (no crash).
  __m256 mask = _mm256_cmp_ps(logp_vec, kth_logp, _CMP_LT_OQ);
  return _mm256_movemask_ps(mask) == 0xFF;
}
#endif

// Given the current partial hypothesis in 'hyps' for all beams in a batch and
// the predicted next step scores 'scores', return the best scored 'k'
// hypotheses where the first 'k' hypotheses are used for search in the next
// step. 'eos_id' is the end of beam id of the target language.
//
// `skip_beam`: size of `num_beams`. Caller should set to true at beam_index
// to make this computation to ignore beam_index. This is used for beam
// independence mode when input_beam_done[beam_index] is true. Note that this
// means for the returned `top_k`, the Hyp instances corresponding to these
// beam ids are empty (i.e. default initialized).
//
// eos_in_topk is filled with true/false to indicate whether or not the eos
// symbol is among the topk candidate for a hyp.
// terminal_symbols stores the terminal token id (eos or eoc).
void ComputeTopK(int step, const std::vector<Hyp>& hyps, const Tensor& scores,
                 const int32 k, const int32 eos_id, const int32 eoc_id,
                 const int32 num_beams, const float valid_eos_max_logit_delta,
                 const float local_eos_threshold, bool is_first_step,
                 bool is_last_decoder_step, const Tensor& is_last_chunk,
                 bool merge_paths, bool allow_empty_terminated_hyp,
                 bool force_eos_in_top_k, bool force_last_chunk_eoc_in_top_k,
                 int merged_topk_buffer_size_factor,
                 const std::vector<bool>& skip_beam,
                 // Note that this is functionally a bool, however
                 // vector<bool> is not safe to parallel write into
                 // since it's underlying storage is at the byte-level.
                 std::vector<char>* eos_in_topk, std::vector<Hyp>* top_k,
                 std::vector<Hyp>* eos_hyps,
                 std::vector<int32>* terminal_symbols) {
  VLOG(1) << "Topk clear, num_beams: " << num_beams;
  DCHECK_EQ(hyps.size(), num_beams * k);
  DCHECK(eos_in_topk && top_k && eos_hyps && terminal_symbols);
  DCHECK_EQ(hyps.size(), scores.dim_size(0));
  DCHECK_LT(eos_id, scores.dim_size(1));
  int hyps_size = hyps.size();
  eos_in_topk->clear();
  top_k->clear();
  top_k->resize(hyps_size);
  eos_in_topk->resize(hyps_size);
  eos_hyps->resize(hyps_size);
  terminal_symbols->resize(hyps_size);
  static thread::ThreadPool* workers =
      new thread::ThreadPool(Env::Default(), "topk", kNumWorkers);
  const int num_ids = scores.dim_size(1);
  const auto scores_matrix = scores.matrix<float>();
  const int epsilon_id_for_path_merging = merge_paths ? eoc_id : -1;
  std::vector<
      TopK<Hyp, HigherScore, ExtractGlobalScore, InsertHypWithEpsilonDedupe>>
      merged_topk_vec(num_beams, TopK<Hyp, HigherScore, ExtractGlobalScore,
                                      InsertHypWithEpsilonDedupe>(
                                     k, epsilon_id_for_path_merging,
                                     merged_topk_buffer_size_factor));
  // Each mutex is used to protect corresponding merged_topk_vec.
  std::vector<mutex> mu_vec(num_beams);
  // The thread sharding is along the hyps_size.
  Shard(
      kNumWorkers, workers, hyps_size, num_ids, [&](int64 start, int64 limit) {
        for (int32 hyp_id = start; hyp_id < limit; ++hyp_id) {
          if (is_first_step && hyp_id >= num_beams) {
            // For first step, we only consider the first hyp of each beam, as
            // otherwise we will be continuing k identical hyps along the way.
            continue;
          }
          if (skip_beam[hyp_id % num_beams]) {
            continue;
          }
          // +1 to make sure that at least top-k hypotheses survive even with
          // the special treatment for eos.  +2 if we are also using eoc.
          const int topk_size = k + 1 + static_cast<int>(eoc_id >= 0);
          TopK<Hyp, HigherScoreWithEos, ExtractGlobalScore,
               InsertHypWithEpsilonDedupe>
              topk(topk_size, epsilon_id_for_path_merging, eos_id,
                   is_last_decoder_step);
          float bottom_of_topk = -INFINITY;
          int32 id = 0;
          const float current_global_score = hyps[hyp_id].global_score;
      // TODO(xbing): Try AVX512 if it is supported by machine.
#ifdef __AVX__
          const int STRIDE =
              sizeof(__m256) /
              sizeof(std::result_of<decltype(scores_matrix)(int, int)>::type);
          // We read STRIDE float values at a single iteration and compare
          // them with this k-th best value. STRIDE - 1 not to read outside
          // the row.
          for (; id + STRIDE - 1 < num_ids; id += STRIDE) {
            if (!all_less_than(&scores_matrix(hyp_id, id),
                               bottom_of_topk - current_global_score)) {
              for (int i = 0; i < STRIDE; ++i) {
                const float score = scores_matrix(hyp_id, id + i);
                const float global_score =
                    current_global_score + score;
                if (global_score >= bottom_of_topk) {
                  bottom_of_topk =
                      topk.Add({hyps[hyp_id].beam_id, hyp_id, id + i, score,
                                global_score, hyps[hyp_id].prev_labels});
                }
              }
            }
          }
      // Non-AVX code below handles the remaining elements.
#endif
          for (; id != num_ids; ++id) {
            const float score = scores_matrix(hyp_id, id);
            const float global_score = current_global_score + score;
            if (global_score >= bottom_of_topk) {
              bottom_of_topk =
                  topk.Add({hyps[hyp_id].beam_id, hyp_id, id, score,
                            global_score, hyps[hyp_id].prev_labels});
            }
          }

          std::vector<Hyp> entries = topk.Get();
          DCHECK(!entries.empty())
              << "No entries in TopK. This typically "
              << "happens if your model is producing NaNs in the output.";
          std::sort(entries.begin(), entries.end(), HigherScore());
          if (force_eos_in_top_k) {
            if (std::find_if(entries.begin(), entries.end(),
                             [=](const Hyp& hyp) {
                               return hyp.word_id == eos_id;
                             }) == entries.end()) {
              entries.pop_back();
              const float eos_score = scores_matrix(hyp_id, eos_id);
              entries.push_back({hyps[hyp_id].beam_id, hyp_id, eos_id,
                                 eos_score, current_global_score + eos_score,
                                 hyps[hyp_id].prev_labels});
            }
          }
          if (force_last_chunk_eoc_in_top_k && eoc_id >= 0 &&
              is_last_chunk.vec<bool>()(hyp_id) &&
              (std::find_if(entries.begin(), entries.end(),
                            [=](const Hyp& hyp) {return hyp.word_id == eoc_id;})
               == entries.end())) {
            Hyp last_hyp = Hyp(entries.back());
            entries.pop_back();
            entries.pop_back();
            const float eoc_score = scores_matrix(hyp_id, eoc_id);
            // Forced last chunk eoc is located in the second last position.
            // We choose to overwrite the second last position instead of the
            // very last one as the latter may already have been overwritten
            // due to force_eos_in_top_k.
            // Also note when eoc_id >= 0, we have reserved two additional
            // positions with topk_size, one for eos and one for eoc. So we
            // can afford to overwrite a different position for eoc than eos.
            entries.push_back({hyps[hyp_id].beam_id, hyp_id, eoc_id,
                               eoc_score, current_global_score + eoc_score,
                               hyps[hyp_id].prev_labels});
            entries.push_back(last_hyp);
          }
          const float eos_score_threshold =
              entries[0].global_score - valid_eos_max_logit_delta;
          VLOG(3) << "Best_score=" << entries[0].global_score
                  << " eos_score_threshold=" << eos_score_threshold;
          {
            const int beam_id = hyps[hyp_id].beam_id;
            mutex_lock l(mu_vec[beam_id]);
            for (const auto& e : entries) {
              VLOG(3) << "Extension for beam_id=" << beam_id
                      << ", hyp_id=" << hyp_id
                      << ": global_score=" << e.global_score
                      << ", local_score=" << e.local_score
                      << ", toks=[" << str_util::Join(e.prev_labels, " ")
                      << "], proposing token " << e.word_id;
              if (e.word_id == eos_id) {
                VLOG(3) << "EOS hyp: global_score=" << e.global_score
                        << ", local_score=" << e.local_score
                        << ", toks=[" << str_util::Join(e.prev_labels, " ")
                        << "]";
                // We move terminated hyps off of the beam.
                if (is_last_decoder_step ||
                    (e.global_score > eos_score_threshold &&
                    e.local_score > local_eos_threshold)) {
                  (*eos_in_topk)[hyp_id] = true;
                  (*eos_hyps)[hyp_id] = e;
                  (*terminal_symbols)[hyp_id] = eos_id;
                }
              } else if (eoc_id >= 0 && is_last_chunk.vec<bool>()(hyp_id) &&
                         e.word_id == eoc_id) {
                // At the last chunk and output <epsilon>. We terminate the
                // hypothesis, even though <eos> was not predicted, and
                // indicate that the final symbol for the hypothesis is
                // <epsilon>, not <eos>.
                if (e.global_score > eos_score_threshold &&
                    e.local_score > local_eos_threshold &&
                    // Only allow an empty hyp (all <epsilon>s) to be
                    // considered terminated, if explicitly permitted.
                    // 'prev_labels' contains only non-epsilons.
                    (allow_empty_terminated_hyp || !e.prev_labels.empty())) {
                    VLOG(3) << "Last chunk EOC hyp: global_score="
                        << e.global_score
                        << ", local_score=" << e.local_score
                        << ", toks=[" << str_util::Join(e.prev_labels, " ")
                        << "]";
                  (*eos_in_topk)[hyp_id] = true;
                  (*eos_hyps)[hyp_id] = e;
                  (*terminal_symbols)[hyp_id] = eoc_id;
                }
              } else {
                merged_topk_vec[beam_id].Add(e);
              }
            }
          }
        }
      });

  const int hyps_per_beam = k;
  for (int i = 0; i < num_beams; ++i) {
    if (skip_beam[i]) {
      continue;
    }
    auto ith_topk = merged_topk_vec[i].Get();
    std::sort(ith_topk.begin(), ith_topk.end(), HigherScore());
    const int num_hyps =
        std::min(static_cast<int>(ith_topk.size()), hyps_per_beam);
    VLOG(3) << "Active hyps for beam_id=" << i;
    for (int j = 0; j < num_hyps; ++j) {
      (*top_k)[j * num_beams + i] = ith_topk[j];
      VLOG(3) << "Active hyp " << j
              << ", global_score=" << ith_topk[j].global_score
              << ", local score=" << ith_topk[j].local_score
              << ", toks=[" << str_util::Join(ith_topk[j].prev_labels, " ")
              << "]";
    }
  }
  VLOG(1) << "Topk done";
}

// Symbols:
// B: num_beams.
// K: num_hyps_per_beam.
// N: num_hyps = K * B.
// S: source sequence length.
// T: target sequence length.
// V: vocab size.
template <int op_version>
class BeamSearchStepOp : public OpKernel {
 public:
  explicit BeamSearchStepOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    DCHECK_EQ(ctx->def().op(), OpName());

    OP_REQUIRES_OK(ctx, ctx->GetAttr("eos_id", &eos_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eoc_id", &eoc_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_size", &beam_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_hyps_per_beam", &num_hyps_per_beam_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("valid_eos_max_logit_delta",
                                     &valid_eos_max_logit_delta_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("local_eos_threshold",
                                     &local_eos_threshold_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_paths", &merge_paths_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("allow_empty_terminated_hyp",
                                     &allow_empty_terminated_hyp_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ensure_full_beam", &ensure_full_beam_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("force_eos_in_last_step", &force_eos_in_last_step_));
    if (op_version == 2) {
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("beam_independence", &beam_independence_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("atten_vecs_in_hypothesis_protos",
                                       &atten_vecs_in_hypothesis_protos_));
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("force_eos_in_top_k", &force_eos_in_top_k_));
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("force_last_chunk_eoc_in_top_k",
                                  &force_last_chunk_eoc_in_top_k_));
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("merged_topk_buffer_size_factor",
                                  &merged_topk_buffer_size_factor_));
    }

    DCHECK_GE(eos_id_, 0);
    DCHECK_GT(beam_size_, 0.0);
    DCHECK_GT(num_hyps_per_beam_, 0);
    DCHECK_GT(merged_topk_buffer_size_factor_, 1);

    if (merge_paths_) {
      OP_REQUIRES(
          ctx, eoc_id_ >= 0,
          errors::InvalidArgument(
              "Paths can only be merged for an epsilon-emitting model (RNN-T "
              "or NT).  Epsilon id must be non-negative, but got: ",
              eoc_id_));
    }
  }

 private:
  static constexpr absl::string_view OpName();

  Status ForwardOrCopyInputToOutput(OpKernelContext* ctx, int input_idx,
                                    int output_idx, Tensor** output) {
    const Tensor& input = ctx->input(input_idx);
    auto status = ctx->forward_input_or_allocate_output({input_idx}, output_idx,
                                                        input.shape(), output);
    if (status.ok()) {
      if (!(*output)->SharesBufferWith(input)) {
        // Copy the input data if we were unable to forward the underlying
        // buffer.
        if (DataTypeCanUseMemcpy(input.dtype())) {
          if (input.NumElements() > 0) {
            StringPiece input_data = input.tensor_data();
            StringPiece output_data = (*output)->tensor_data();
            memcpy(const_cast<char*>(output_data.data()), input_data.data(),
                   input_data.size());
          }
        } else if (input.dtype() == DT_STRING) {
          (*output)->flat<tstring>() = input.flat<tstring>();
        }
      }
    }
    return status;
  }

  string AssembleDoneHypProto(const Hyp& hyp, const int32 terminal_sym,
                              const TTypes<int32>::Matrix& t_out_prev_hyps,
                              const TTypes<int32>::Matrix& t_out_hyps,
                              const TTypes<float>::Matrix& t_out_scores,
                              const TTypes<float, 3>::Tensor t_out_atten_probs,
                              const Tensor& atten_probs, int t) const {
    std::vector<int> hyp_ids(t);
    int hyp_id = hyp.hyp_id;
    for (int i = t - 1; i >= 0; --i) {
      hyp_ids[i] = hyp_id;
      hyp_id = t_out_prev_hyps(i, hyp_id);
    }
    Hypothesis hypothesis;
    hypothesis.set_beam_id(hyp.beam_id);
    // Add one to account for t-th step (terminal sym).
    const float average_step_score = hyp.global_score / (t + 1);
    for (int i = 0; i < t; i++) {
      const int hyp_id = hyp_ids[i];
      hypothesis.add_ids(t_out_hyps(i, hyp_id));
      // If this is a model with epsilons (RNN-T or NT), then this hyp
      // may represent many possible paths that have been merged.  The
      // recorded per-step scores only are valid for one of these paths, so
      // they are not meaningful for the merged path.  For the merged path
      // we simply take an average per-step score.
      const float score_this_step =
          (merge_paths_ ? average_step_score : t_out_scores(i, hyp_id));
      hypothesis.add_scores(score_this_step);
      if (atten_vecs_in_hypothesis_protos_) {
        auto* att_vec = hypothesis.add_atten_vecs();
        for (int j = 0; j < atten_probs.dim_size(1); ++j) {
          att_vec->add_prob(t_out_atten_probs(i, hyp_id, j));
        }
      }
    }
    // Now add the terminal symbol.
    hypothesis.add_ids(terminal_sym);
    // As above, use the average per-step score for RNN-T and NT.
    const float score_this_step =
        merge_paths_ ? average_step_score : hyp.local_score;
    hypothesis.add_scores(score_this_step);
    if (atten_vecs_in_hypothesis_protos_) {
      auto* att_vec = hypothesis.add_atten_vecs();
      auto t_atten_probs = atten_probs.matrix<float>();
      for (int j = 0; j < atten_probs.dim_size(1); ++j) {
        att_vec->add_prob(t_atten_probs(hyp.hyp_id, j));
      }
    }
    return hypothesis.SerializeAsString();
  }

  void SanityCheckInputs(OpKernelContext* ctx) {
    const Tensor& scores = ctx->input(0);
    const Tensor& atten_probs = ctx->input(1);
    const Tensor& best_scores = ctx->input(2);
    const Tensor& cumulative_scores = ctx->input(3);
    const Tensor& in_scores = ctx->input(4);
    const Tensor& in_hyps = ctx->input(5);
    const Tensor& in_prev_hyps = ctx->input(6);
    const Tensor& in_done_hyps = ctx->input(7);
    const Tensor& in_atten_probs = ctx->input(8);
    const Tensor& cur_step = ctx->input(op_version == 2 ? 11 : 10);

    OP_REQUIRES(
        ctx, scores.dims() == 2,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. scores.dims() == 2. Got ",
            scores.dims()));
    OP_REQUIRES(
        ctx, atten_probs.dims() == 2,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. atten_probs.dims() == 2. Got ",
            atten_probs.dims()));
    OP_REQUIRES(
        ctx, best_scores.dims() == 1,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. best_scores.dims() == 1. Got ",
            best_scores.dims()));
    OP_REQUIRES(ctx, cumulative_scores.dims() == 1,
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "cumulative_scores.dims() == 1. Got ",
                                        cumulative_scores.dims()));
    OP_REQUIRES(
        ctx, in_scores.dims() == 2,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. in_scores.dims() == 2. Got ",
            in_scores.dims()));
    OP_REQUIRES(ctx, in_hyps.dims() == 2,
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "in_hyps.dims() == 2. Got ",
                                        in_hyps.dims()));

    OP_REQUIRES(
        ctx, in_prev_hyps.dims() == 2,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. in_prev_hyps.dims() == 2. Got ",
            in_prev_hyps.dims()));
    OP_REQUIRES(
        ctx, in_done_hyps.dims() == 2,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. in_done_hyps.dims() == 2. Got ",
            in_done_hyps.dims()));
    OP_REQUIRES(ctx, in_atten_probs.dims() == 3,
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "in_atten_probs.dims() == 3. Got ",
                                        in_atten_probs.dims()));
    OP_REQUIRES(
        ctx, cur_step.dims() == 0,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. cur_step.dims() == 0. Got ",
            cur_step.dims()));
    OP_REQUIRES(ctx, scores.dim_size(0) == atten_probs.dim_size(0),
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "scores.dim_size(0) == "
                                        "atten_probs.dim_size(0). Got ",
                                        scores.dim_size(0), " and ",
                                        atten_probs.dim_size(0)));
    OP_REQUIRES(ctx, scores.dim_size(0) == cumulative_scores.dim_size(0),
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "scores.dim_size(0) == "
                                        "cumulative_scores.dim_size(0). Got ",
                                        scores.dim_size(0), " and ",
                                        cumulative_scores.dim_size(0)));
    OP_REQUIRES(ctx, scores.dim_size(0) % best_scores.dim_size(0) == 0,
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "scores.dim_size(0) % "
                                        "best_scores.dim_size(0) == 0. Got ",
                                        scores.dim_size(0), " and ",
                                        best_scores.dim_size(0)));
    OP_REQUIRES(
        ctx, scores.dim_size(0) / best_scores.dim_size(0) == num_hyps_per_beam_,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. "
            "scores.dim_size(0) / best_scores.dim_size(0) "
            "== num_hyps_per_beam_. Got ",
            scores.dim_size(0), " and ", best_scores.dim_size(0),
            " where num_hyps_per_beam_ = ", num_hyps_per_beam_));
    OP_REQUIRES(ctx, scores.dim_size(0) == in_hyps.dim_size(1),
                errors::InvalidArgument(
                    "Failed tensor shape sanity check. "
                    "scores.dim_size(0) == in_hyps.dim_size(1). Got ",
                    scores.dim_size(0), " and ", in_hyps.dim_size(1)));
    OP_REQUIRES(ctx, in_hyps.dim_size(0) == in_scores.dim_size(0),
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "in_hyps.dim_size(0) == "
                                        "in_scores.dim_size(0). Got ",
                                        in_hyps.dim_size(0), " and ",
                                        in_scores.dim_size(0)));
    OP_REQUIRES(ctx, in_hyps.dim_size(0) == in_prev_hyps.dim_size(0),
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "in_hyps.dim_size(0) == "
                                        "in_prev_hyps.dim_size(0). Got ",
                                        in_hyps.dim_size(0), " and ",
                                        in_prev_hyps.dim_size(0)));
    OP_REQUIRES(ctx, in_hyps.dim_size(0) == in_done_hyps.dim_size(0),
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "in_hyps.dim_size(0) == "
                                        "in_done_hyps.dim_size(0). Got ",
                                        in_hyps.dim_size(0), " and ",
                                        in_done_hyps.dim_size(0)));
    OP_REQUIRES(ctx, in_hyps.dim_size(0) == in_atten_probs.dim_size(0),
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "in_hyps.dim_size(0) == "
                                        "in_atten_probs.dim_size(0). Got ",
                                        in_hyps.dim_size(0), " and ",
                                        in_atten_probs.dim_size(0)));
    OP_REQUIRES(ctx, in_hyps.dim_size(1) == in_scores.dim_size(1),
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "in_hyps.dim_size(1) == "
                                        "in_scores.dim_size(1). Got ",
                                        in_hyps.dim_size(1), " and ",
                                        in_scores.dim_size(1)));
    OP_REQUIRES(ctx, in_hyps.dim_size(1) == in_prev_hyps.dim_size(1),
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "in_hyps.dim_size(1) == "
                                        "in_prev_hyps.dim_size(1). Got ",
                                        in_hyps.dim_size(1), " and ",
                                        in_prev_hyps.dim_size(1)));
    OP_REQUIRES(ctx, in_hyps.dim_size(1) == in_done_hyps.dim_size(1),
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "in_hyps.dim_size(1) == "
                                        "in_done_hyps.dim_size(1). Got ",
                                        in_hyps.dim_size(1), " and ",
                                        in_done_hyps.dim_size(1)));
    OP_REQUIRES(ctx, in_hyps.dim_size(1) == in_atten_probs.dim_size(1),
                errors::InvalidArgument("Failed tensor shape sanity check. "
                                        "in_hyps.dim_size(1) == "
                                        "in_atten_probs.dim_size(1). Got ",
                                        in_hyps.dim_size(1), " and ",
                                        in_atten_probs.dim_size(1)));
    OP_REQUIRES(
        ctx, atten_probs.dim_size(1) == in_atten_probs.dim_size(2),
        errors::InvalidArgument(
            "Failed tensor shape sanity check. "
            "atten_probs.dim_size(1) == in_atten_probs.dim_size(2). Got ",
            atten_probs.dim_size(1), " and ", in_atten_probs.dim_size(2)));

    if (op_version == 2) {
      const Tensor& in_beam_done = ctx->input(9);
      OP_REQUIRES(ctx, in_beam_done.dtype() == DT_BOOL,
                  errors::InvalidArgument("Failed tensor type sanity check. "
                                          "in_beam_done is tf.bool. Got ",
                                          in_beam_done.dtype()));
      OP_REQUIRES(ctx, in_beam_done.dims() == 1,
                  errors::InvalidArgument("Failed tensor shape sanity check. "
                                          "in_beam_done.dims() == 1. Got ",
                                          in_beam_done.dims()));
      OP_REQUIRES(ctx, in_beam_done.dim_size(0) == best_scores.dim_size(0),
                  errors::InvalidArgument("Failed tensor shape sanity check. "
                                          "in_beam_done.dim_size(0) == "
                                          "best_scores.dim_size(0). Got ",
                                          in_beam_done.dim_size(0), " and ",
                                          best_scores.dim_size(0)));
    }
  }

  // Assembles hyps from input tensors.
  // in_hyps: [T, N]
  // in_prev_hyps: [T, N]
  // cumulative_scores: [N]
  void AssembleHyps(Tensor in_hyps, Tensor in_prev_hyps,
                    Tensor cumulative_scores, const int num_beams, const int t,
                    std::vector<Hyp>* hyps) {
    const int num_hyps = cumulative_scores.dim_size(0);
    auto t_cumulative_scores = cumulative_scores.vec<float>();
    for (int i = 0; i < num_hyps; ++i) {
      hyps->at(i).beam_id = i % num_beams;
      hyps->at(i).hyp_id = i;
      hyps->at(i).global_score = t_cumulative_scores(i);
      // Determines the sequence of prev ids that this hypothesis represents.
      std::vector<int> hyp_id_at_step(t);
      int hyp_id = i;
      for (int j = t - 1; j >= 0; --j) {
        hyp_id_at_step[j] = hyp_id;
        hyp_id = in_prev_hyps.matrix<int>()(j, hyp_id);
      }
      for (int j = 0; j < t; ++j) {
        const int prev_id = in_hyps.matrix<int>()(j, hyp_id_at_step[j]);
        if (prev_id != eoc_id_) {
          hyps->at(i).prev_labels.push_back(prev_id);
        }
      }
      VLOG(3) << "Step " << t << " input hyp " << i
              << ": global_score=" << hyps->at(i).global_score
              << ", toks=[" << str_util::Join(hyps->at(i).prev_labels, " ")
              << "]";
    }
  }

 public:
  void Compute(OpKernelContext* ctx) override {
    // [N, V]
    const Tensor& scores = ctx->input(0);
    // [N, S]
    const Tensor& atten_probs = ctx->input(1);
    // [B]
    const Tensor& best_scores = ctx->input(2);
    // [N]
    const Tensor& cumulative_scores = ctx->input(3);
    // [T, N]
    const Tensor& in_hyps = ctx->input(5);
    // [T, N]
    const Tensor& in_prev_hyps = ctx->input(6);
    // [N]
    const Tensor& is_last_chunk = ctx->input(op_version == 2 ? 10 : 9);
    // []
    const Tensor& cur_step = ctx->input(op_version == 2 ? 11 : 10);
    // [B], only when op_version == 2.
    const Tensor& input_beam_done = ctx->input(9);

    SanityCheckInputs(ctx);
    if (!ctx->status().ok()) {
      return;
    }

    const int num_beams = best_scores.dim_size(0);
    const int num_hyps = cumulative_scores.dim_size(0);
    const int t = cur_step.scalar<int>()();
    DCHECK_EQ(num_hyps_per_beam_, num_hyps / num_beams);
    DCHECK_LT(t, in_hyps.dim_size(0));
    VLOG(2) << "BeamSearchStepOp(" << num_hyps_per_beam_ << ") Step=" << t;

    // Assembles hyps from inputs.
    std::vector<Hyp> hyps(num_hyps);
    AssembleHyps(in_hyps, in_prev_hyps, cumulative_scores, num_beams, t, &hyps);

    std::vector<Hyp> top_k_hyps;
    std::vector<Hyp> eos_hyps;
    std::vector<char> eos_in_topk;
    std::vector<int32> terminal_symbols;
    const bool is_last_decoder_step =
        (t == (in_hyps.dim_size(0) - 1)) && force_eos_in_last_step_;
    std::vector<bool> skip_beam(num_beams, false);
    if (op_version == 2 && beam_independence_) {
      for (int i = 0; i < num_beams; ++i) {
        skip_beam[i] = input_beam_done.vec<bool>()(i);
      }
    }
    ComputeTopK(t, hyps, scores, /*k=*/num_hyps_per_beam_,
                /*eos_id=*/eos_id_, /*eoc_id=*/eoc_id_, num_beams,
                valid_eos_max_logit_delta_, local_eos_threshold_,
                /*is_first_step=*/t == 0, is_last_decoder_step, is_last_chunk,
                merge_paths_, allow_empty_terminated_hyp_, force_eos_in_top_k_,
                force_last_chunk_eoc_in_top_k_, merged_topk_buffer_size_factor_,
                skip_beam, &eos_in_topk, &top_k_hyps, &eos_hyps,
                &terminal_symbols);

    Tensor* out_done_hyps = nullptr;
    OP_REQUIRES_OK(ctx, ForwardOrCopyInputToOutput(ctx, 7, 5, &out_done_hyps));

    Tensor* out_best_scores = nullptr;
    Tensor* out_cumulative_scores = nullptr;
    Tensor* all_done = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, best_scores.shape(), &out_best_scores));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, cumulative_scores.shape(),
                                             &out_cumulative_scores));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(op_version == 2 ? 8 : 7,
                                             TensorShape({}), &all_done));

    // [B]
    auto t_out_best_scores = out_best_scores->vec<float>();
    // [N]
    auto t_out_cumulative_scores = out_cumulative_scores->vec<float>();
    // [T, N]
    Tensor* out_scores = nullptr;
    OP_REQUIRES_OK(ctx, ForwardOrCopyInputToOutput(ctx, 4, 2, &out_scores));
    auto t_out_scores = out_scores->matrix<float>();
    // [T, N]
    Tensor* out_hyps = nullptr;
    OP_REQUIRES_OK(ctx, ForwardOrCopyInputToOutput(ctx, 5, 3, &out_hyps));
    auto t_out_hyps = out_hyps->matrix<int>();
    // [T, N]
    Tensor* out_prev_hyps = nullptr;
    OP_REQUIRES_OK(ctx, ForwardOrCopyInputToOutput(ctx, 6, 4, &out_prev_hyps));
    auto t_out_prev_hyps = out_prev_hyps->matrix<int>();
    // [T, N]
    auto t_out_done_hyps = out_done_hyps->matrix<tstring>();
    // [T, N, S]
    Tensor* out_atten_probs = nullptr;
    OP_REQUIRES_OK(ctx,
                   ForwardOrCopyInputToOutput(ctx, 8, 6, &out_atten_probs));
    auto t_out_atten_probs = out_atten_probs->tensor<float, 3>();

    // To initialize the two vectors.
    t_out_best_scores = best_scores.vec<float>();
    t_out_cumulative_scores = cumulative_scores.vec<float>();

    // Fill in all the output tensors.
    for (int i = 0; i < num_hyps; ++i) {
      const int beam_id = i % num_beams;
      // Make this a no-op by skipping writing to output if beam_done[beam_id]
      // under op_version == 2 and beam_independence mode.
      if (op_version == 2 && skip_beam[beam_id]) {
        continue;
      }

      const Hyp& hyp = top_k_hyps[i];
      t_out_scores(t, i) = hyp.local_score;
      t_out_cumulative_scores(i) = hyp.global_score;
      t_out_hyps(t, i) = hyp.word_id;
      t_out_prev_hyps(t, i) = hyp.hyp_id;
      t_out_atten_probs.chip(t, 0).chip(i, 0) =
          atten_probs.matrix<float>().chip(hyp.hyp_id, 0);
      if (eos_in_topk[i]) {
        // We have a good terminated hyp.
        DCHECK_EQ(beam_id, eos_hyps[i].beam_id);
        VLOG(2) << "Terminated hyp @step " << t
                << ", global_score=" << eos_hyps[i].global_score
                << ", local_score=" << eos_hyps[i].local_score
                << ", terminal_symbol=" << terminal_symbols[i]
                << ", toks=[" << str_util::Join(eos_hyps[i].prev_labels, " ")
                << " " << terminal_symbols[i] << "]";
        // Update the best scores.
        if (eos_hyps[i].global_score > t_out_best_scores(beam_id)) {
          t_out_best_scores(beam_id) = eos_hyps[i].global_score;
        }
        string done_hyp = AssembleDoneHypProto(
            eos_hyps[i], terminal_symbols[i], t_out_prev_hyps, t_out_hyps,
            t_out_scores, t_out_atten_probs, atten_probs, t);
        t_out_done_hyps(t, i) = done_hyp;
      }
    }

    Tensor* out_beam_done = nullptr;
    // Only used when op_version == 1.
    Tensor temp_beam_done;
    if (op_version == 2) {
      OP_REQUIRES_OK(ctx,
                     ForwardOrCopyInputToOutput(ctx, 9, 7, &out_beam_done));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_BOOL, best_scores.shape(),
                                             &temp_beam_done));
      temp_beam_done.vec<bool>().setConstant(false);
      out_beam_done = &temp_beam_done;
    }
    // Update all_done (scalar) output.
    UpdateAllDone(top_k_hyps, num_beams, num_hyps, t, t_out_done_hyps,
                  t_out_best_scores, out_beam_done, all_done);
  }

 private:
  // Under op_version == 1, we may return early without fully updating
  // 'beam_done' for all beams.
  //
  // top_k_hyps: [K]
  // out_done_hyps: [T, N]
  // out_best_scores: [B]
  void UpdateAllDone(const std::vector<Hyp>& top_k_hyps, const int num_beams,
                     const int num_hyps, const int t,
                     TTypes<tstring>::Matrix t_out_done_hyps,
                     TTypes<float>::Vec t_out_best_scores, Tensor* beam_done,
                     Tensor* all_done) {
    // [B]
    auto t_beam_done = beam_done->vec<bool>();
    auto t_all_done = all_done->scalar<bool>();

    // For each beam with beam_id, beam_done[beam_id] is true if and only if:
    //   - beam_done[i] was previously already true, OR;
    //   - the following condition is met:
    //     - (if ensure_full_beam_) we have num_hyps_per_beam EOS hyps, AND;
    //     - all hyps outside of 'beam_size' of best score.
    //
    // all_done is logical AND over elements of beam_done.
    for (int beam_id = 0; beam_id < num_beams; ++beam_id) {
      if (op_version == 2 && t_beam_done(beam_id)) {
        continue;
      }
      if (ensure_full_beam_) {
        // First check how many EOS hyps we have.  If we have
        // num_hyps_per_beam for this beam, this beam is done.
        int num_done_hyps = 0;
        for (int hyp_id = 0; hyp_id < num_hyps_per_beam_; ++hyp_id) {
          for (int time_step = 0; time_step <= t; ++time_step) {
            int index = hyp_id * num_beams + beam_id;
            if (!t_out_done_hyps(time_step, index).empty()) {
              ++num_done_hyps;
            }
          }
        }
        if (num_done_hyps < num_hyps_per_beam_) {
          if (op_version == 1) {
            t_all_done() = false;
            return;
          }
          // If we are not done for this beam_id, we can move on to update next
          // beam_id without checking 'beam_size' based test below,
          t_beam_done(beam_id) = false;
          continue;
        }
      }
      // Now check for hyp quality.  If for all hyps are below best score -
      // 'beam_size', this beam is done.
      bool all_below_beam_size = true;
      VLOG(3) << "Check hyp quality for beam_id=" << beam_id
              << ": best score=" << t_out_best_scores(beam_id)
              << ", beam_size=" << beam_size_;
      for (int hyp_id = 0; hyp_id < num_hyps_per_beam_; ++hyp_id) {
        int i = hyp_id * num_beams + beam_id;
        const Hyp& hyp = top_k_hyps[i];
        DCHECK_EQ(beam_id, hyp.beam_id);
        VLOG(3) << "Hyp=[" << hyp.DebugString() << "]";
        if (hyp.global_score > t_out_best_scores(beam_id) - beam_size_) {
          all_below_beam_size = false;
          break;
        }
      }
      t_beam_done(beam_id) = t_beam_done(beam_id) || all_below_beam_size;

      if (op_version == 1 && !t_beam_done(beam_id)) {
        t_all_done() = false;
        return;
      }
    }

    t_all_done() = true;
    if (op_version == 1) {
      return;
    }

    // all_done is logical AND over elements of beam_done.
    for (int beam_id = 0; beam_id < num_beams; ++beam_id) {
      t_all_done() = t_all_done() && t_beam_done(beam_id);
    }
  }

 private:
  int eos_id_ = 0;
  int eoc_id_ = -1;
  float beam_size_ = 0.0;
  int num_hyps_per_beam_ = 0;
  float valid_eos_max_logit_delta_ = 0.0;
  float local_eos_threshold_ = 0.0;
  bool merge_paths_ = false;
  bool allow_empty_terminated_hyp_ = true;
  bool ensure_full_beam_ = false;
  bool force_eos_in_last_step_ = false;
  bool force_eos_in_top_k_ = false;
  bool force_last_chunk_eoc_in_top_k_ = false;
  int merged_topk_buffer_size_factor_ = 2;

  // Whether each beam terminates independently. Only supported when op_version
  // is 2.
  bool beam_independence_ = false;
  bool atten_vecs_in_hypothesis_protos_ = true;
};

template <>
constexpr absl::string_view BeamSearchStepOp<1>::OpName() {
  return "BeamSearchStep";
}

template <>
constexpr absl::string_view BeamSearchStepOp<2>::OpName() {
  return "BeamSearchStepV2";
}

REGISTER_KERNEL_BUILDER(Name("BeamSearchStep").Device(DEVICE_CPU),
                        BeamSearchStepOp<1>);
REGISTER_KERNEL_BUILDER(Name("BeamSearchStepV2").Device(DEVICE_CPU),
                        BeamSearchStepOp<2>);

class TopKTerminatedHypsOp : public OpKernel {
 public:
  explicit TopKTerminatedHypsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("k", &k_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_hyps_per_beam", &num_hyps_per_beam_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("length_normalization", &length_normalization_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("coverage_penalty", &coverage_penalty_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("target_seq_length_ratio",
                                     &target_seq_length_ratio_));
    DCHECK_GE(length_normalization_, 0.0);
    DCHECK_GE(target_seq_length_ratio_, 0.0);
    DCHECK_GE(coverage_penalty_, 0);
    DCHECK_GT(num_hyps_per_beam_, 0);
    DCHECK_GT(k_, 0);
  }

  void ComputeTopK(const Tensor& in_done_hyps,
                   const std::vector<int32> src_seq_lengths, const int32 k,
                   const int32 num_beams, Tensor* topk_hyps) {
    VLOG(1) << "Topk clear, num_beams: " << num_beams;
    int hyps_size = in_done_hyps.dim_size(1);
    int num_steps = in_done_hyps.dim_size(0);
    static thread::ThreadPool* workers = new thread::ThreadPool(
        Env::Default(), "topk_terminated_hyps", kNumWorkers);
    // No Insert struct is provided, so we use DefaultInsert, which inserts
    // without deduping.  No deduping is necessary here because we dedupe
    // partial hyps at each step of beam search.
    std::vector<TopK<Hypothesis, BetterTerminatedHyp, ExtractNormalizedScore>>
        topk_vec(num_beams,
                 TopK<Hypothesis, BetterTerminatedHyp, ExtractNormalizedScore>(
                     k, /* unused epsilon id */ -1));
    // Each mutex is used to protect corresponding topk_vec.
    std::vector<mutex> mu_vec(num_beams);
    auto t_done_hyps = in_done_hyps.matrix<tstring>();
    // The thread sharding is along hyps_size.
    Shard(kNumWorkers, workers, hyps_size, 1000 * num_steps,
          [&](int64 start, int64 limit) {
            Hypothesis hypothesis;
            for (int32 hyp_id = start; hyp_id < limit; ++hyp_id) {
              // The topk for this beam.
              TopK<Hypothesis, BetterTerminatedHyp, ExtractNormalizedScore>*
                  topk = &topk_vec[hyp_id % num_beams];
              for (int32 step_id = 0; step_id < num_steps; ++step_id) {
                const string& str_hyps = t_done_hyps(step_id, hyp_id);
                if (!str_hyps.empty()) {
                  hypothesis.ParseFromString(str_hyps);
                  if (hypothesis.has_beam_id()) {
                    // This hypothesis is a real terminated hyps.
                    int src_size = src_seq_lengths[hypothesis.beam_id()];
                    float normalized_score =
                        NormalizedScore(hypothesis, src_size);
                    hypothesis.set_normalized_score(normalized_score);
                    VLOG(2)
                        << "Add to terminated top-k:"
                        << " score=" << hypothesis.normalized_score()
                        << ", toks=[" << str_util::Join(hypothesis.ids(), " ")
                        << "]";
                    // TODO(xbing): avoid acquiring a mutex for each record.
                    mutex_lock l(mu_vec[hyp_id % num_beams]);
                    topk->Add(hypothesis);
                  }
                }
              }
            }
          });

    auto t_topk_hyps = topk_hyps->matrix<tstring>();
    for (int i = 0; i < num_beams; ++i) {
      auto ith_topk = topk_vec[i].Get();
      DCHECK_LE(ith_topk.size(), k);
      std::sort(ith_topk.begin(), ith_topk.end(), BetterTerminatedHyp());
      for (int j = 0; j < ith_topk.size(); ++j) {
        t_topk_hyps(i, j) = ith_topk[j].SerializeAsString();
        VLOG(2) << "TopK(" << i << ", " << j
                << ") ids = [" << str_util::Join(ith_topk[j].ids(), " ")
                << "], scores = [" << str_util::Join(ith_topk[j].scores(), ", ")
                << "]";
      }
    }
  }

  float NormalizedScore(const Hypothesis& hypothesis,
                        const int src_size) const {
    int length = hypothesis.scores_size();
    Tensor cumulative_atten_prob(DT_FLOAT, {src_size});
    auto cumulative_atten_prob_vec = cumulative_atten_prob.vec<float>();
    cumulative_atten_prob_vec.setZero();
    for (int step = 0; step < hypothesis.atten_vecs_size(); ++step) {
      const int hyp_prob_size = hypothesis.atten_vecs(step).prob_size();
      for (int src_id = 0; src_id < src_size; ++src_id) {
        if (src_id < hyp_prob_size) {
          cumulative_atten_prob_vec(src_id) +=
              hypothesis.atten_vecs(step).prob(src_id);
        } else {
          // This can happen e.g. for RNNT model. Here we simply assume
          // atten_prob for those source positions are 0.0
          VLOG(5) << "Missing atten_prob for source position " << src_id
                  << ". Total available positions are " << hyp_prob_size << ".";
          cumulative_atten_prob_vec(src_id) += 0.0;
        }
      }
    }

    // Coverage is capped at 0.5 so that so long as a word is
    // reasonably covered, it is not penalized anymore.
    Tensor penalty(DT_FLOAT, {});
    penalty.scalar<float>() =
        (cumulative_atten_prob_vec / target_seq_length_ratio_)
            .cwiseMax(0.001f)
            .cwiseMin(0.5f)
            .log()
            .sum().eval();
    const float coverage_penalty = target_seq_length_ratio_ *
                                   coverage_penalty_ *
                                   penalty.scalar<float>()();
    const float length_norm = std::pow(length + 5.0, length_normalization_) /
                              std::pow(5.0, length_normalization_);

    float global_score = 0.0;
    for (const auto& score : hypothesis.scores()) {
      global_score += score;
    }
    return global_score / length_norm + coverage_penalty;
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in_done_hyps = ctx->input(0);
    // Some sanity check on the tensor shapes.
    OP_REQUIRES(ctx, in_done_hyps.dim_size(1) % num_hyps_per_beam_ == 0,
                errors::InvalidArgument("Failed tensor shape sanity check."));

    int num_beams = in_done_hyps.dim_size(1) / num_hyps_per_beam_;

    const Tensor& in_src_seq_lens = ctx->input(1);
    OP_REQUIRES(
        ctx, in_src_seq_lens.dim_size(0) == num_beams,
        errors::InvalidArgument(
            "src_seq_lengths should be a 1-d Tensor of length num_beams. Got ",
            in_src_seq_lens.dim_size(0), " vs ", num_beams));
    std::vector<int32> src_seq_lengths(num_beams);
    for (int i = 0; i < num_beams; ++i) {
      src_seq_lengths[i] = in_src_seq_lens.flat<int>()(i);
    }
    // Set the output tensors.
    Tensor* out_topk_hyps = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{num_beams, k_},
                                             &out_topk_hyps));
    ComputeTopK(in_done_hyps, src_seq_lengths, k_, num_beams, out_topk_hyps);
    VLOG(1) << "TopKTerminatedHypsOp(" << num_hyps_per_beam_ << ") done";
  }

 private:
  int32 num_hyps_per_beam_;
  float length_normalization_;
  float coverage_penalty_;
  float target_seq_length_ratio_;
  int32 k_;
};

REGISTER_KERNEL_BUILDER(Name("TopKTerminatedHyps").Device(DEVICE_CPU),
                        TopKTerminatedHypsOp);

class UnpackHypOp : public OpKernel {
 public:
  explicit UnpackHypOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_seq_length", &max_seq_length_));
    DCHECK_GE(max_seq_length_, 0);
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in_hyps = ctx->input(0);
    const auto& t_in_hyps = in_hyps.flat<tstring>();
    const int batch_size = t_in_hyps.size();
    std::vector<Hypothesis> hyps(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      // TODO(yonghui): parallelize this loop.
      const tstring& t_in_hyps_i = t_in_hyps(i);
      if (!t_in_hyps(i).empty()) {
        hyps[i].ParseFromArray(t_in_hyps_i.data(), t_in_hyps_i.size());
      }
    }
    int max_seq_length = max_seq_length_;
    if (max_seq_length <= 0) {
      // Derive max_seq_length from input hyps.
      for (int i = 0; i < batch_size; ++i) {
        if (hyps[i].ids_size() > max_seq_length) {
          max_seq_length = hyps[i].ids_size();
        }
      }
    }
    Tensor* out_ids;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size, max_seq_length}),
                                  &out_ids));
    Tensor* out_seq_lens;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({batch_size}), &out_seq_lens));
    Tensor* out_scores;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(2, TensorShape({batch_size}), &out_scores));
    auto t_out_ids = out_ids->matrix<int32>();
    auto t_out_seq_lens = out_seq_lens->vec<int32>();
    auto t_out_scores = out_scores->vec<float>();
    t_out_ids.setZero();
    t_out_seq_lens.setZero();
    t_out_scores.setZero();
    for (int i = 0; i < batch_size; ++i) {
      const Hypothesis& hyp = hyps[i];
      // TODO(yonghui): parallelize this loop.
      if (hyp.ids_size() > 0) {
        for (int j = 0; j < hyp.ids_size() && j < max_seq_length; ++j) {
          t_out_ids(i, j) = hyp.ids(j);
        }
        t_out_seq_lens(i) = std::min(hyp.ids_size(), max_seq_length);
        t_out_scores(i) = hyp.normalized_score();
      }
    }
  }

 private:
  int32 max_seq_length_ = 0;
};

REGISTER_KERNEL_BUILDER(Name("UnpackHyp").Device(DEVICE_CPU), UnpackHypOp);

template <typename T>
class HypsFromBeamSearchOuts : public OpKernel {
 public:
  explicit HypsFromBeamSearchOuts(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eos_id", &eos_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_hyps_per_beam", &num_hyps_per_beam_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& hyps = ctx->input(0);
    const Tensor& prev_hyps = ctx->input(1);
    const Tensor& done_hyps = ctx->input(2);
    const Tensor& scores = ctx->input(3);
    const Tensor& atten_probs = ctx->input(4);
    const Tensor& eos_scores = ctx->input(5);
    const Tensor& eos_atten_probs = ctx->input(6);

    OP_REQUIRES(ctx, hyps.dims() == 2,
                errors::InvalidArgument(
                    "Failed tensor shape sanity check. hyps.dims() == 2. Got ",
                    hyps.dims()));
    OP_REQUIRES(
        ctx, prev_hyps.dims() == 2,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. prev_hyps.dims() == 2. Got ",
            prev_hyps.dims()));
    OP_REQUIRES(
        ctx, done_hyps.dims() == 2,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. done_hyps.dims() == 2. Got ",
            done_hyps.dims()));
    OP_REQUIRES(
        ctx, scores.dims() == 2,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. scores.dims() == 2. Got ",
            scores.dims()));
    OP_REQUIRES(
        ctx, atten_probs.dims() == 3,
        errors::InvalidArgument(
            "Failed tensor shape sanity check. atten_probs.dims() == 3. Got ",
            atten_probs.dims()));

    OP_REQUIRES(ctx, atten_probs.dim_size(1) == scores.dim_size(1),
                errors::InvalidArgument(
                    "Failed tensor shape sanity check. "
                    "atten_probs.dim_size(1) == scores.dim_size(1). Got ",
                    atten_probs.dim_size(1), " and ", scores.dim_size(1)));

    OP_REQUIRES(ctx, hyps.IsSameSize(prev_hyps),
                errors::InvalidArgument(
                    "Failed tensor shape sanity check. "
                    "hyps and prev_hyps should have the same shape. Got ",
                    hyps.shape().DebugString(), " and ",
                    prev_hyps.shape().DebugString()));

    OP_REQUIRES(ctx, hyps.IsSameSize(done_hyps),
                errors::InvalidArgument(
                    "Failed tensor shape sanity check. "
                    "hyps and done_hyps should have the same shape. Got ",
                    hyps.shape().DebugString(), " and ",
                    done_hyps.shape().DebugString()));

    OP_REQUIRES(
        ctx, hyps.IsSameSize(scores),
        errors::InvalidArgument(
            "Failed tensor shape sanity check. "
            "hyps and scores should have the same shape. Got ",
            hyps.shape().DebugString(), " and ", scores.shape().DebugString()));

    OP_REQUIRES(ctx, hyps.IsSameSize(eos_scores),
                errors::InvalidArgument(
                    "Failed tensor shape sanity check. "
                    "hyps and eos_scores should have the same shape. Got ",
                    hyps.shape().DebugString(), " and ",
                    eos_scores.shape().DebugString()));
    OP_REQUIRES(
        ctx, atten_probs.IsSameSize(eos_atten_probs),
        errors::InvalidArgument(
            "Failed tensor shape sanity check. "
            "atten_probs and eos_atten_probs should have the same shape. Got ",
            atten_probs.shape().DebugString(), " and ",
            eos_atten_probs.shape().DebugString()));

    auto t_hyps = hyps.matrix<int>();
    auto t_prev_hyps = prev_hyps.matrix<int>();
    auto t_done_hyps = done_hyps.matrix<bool>();
    auto t_scores = scores.matrix<T>();
    auto t_atten_probs = atten_probs.tensor<T, 3>();
    auto t_eos_scores = eos_scores.matrix<T>();
    auto t_eos_atten_probs = eos_atten_probs.tensor<T, 3>();
    const int seq_length = hyps.dim_size(0);
    const int num_hyps = hyps.dim_size(1);

    Tensor* out_hyps;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, hyps.shape(), &out_hyps));
    auto out_hyps_t = out_hyps->matrix<tstring>();

    // Use the same thread pool as topk operator.
    static thread::ThreadPool* workers =
        new thread::ThreadPool(Env::Default(), "topk", kNumWorkers);

    Shard(
        kNumWorkers, workers, num_hyps, seq_length * seq_length,
        [&](int64 start, int64 end) {
          std::vector<int> hyp_token_ids;
          std::vector<T> hyp_local_scores;
          std::vector<int> hyp_ids;
          Hypothesis terminated_hyps;
          const int num_beams = (num_hyps / num_hyps_per_beam_);
          for (int i = 0; i < seq_length; ++i) {
            for (int j = start; j < end; ++j) {
              // If the hyp is terminated, then assemble the output proto.
              if (t_done_hyps(i, j)) {
                // Reuse the buffers.
                hyp_token_ids.clear();
                hyp_local_scores.clear();
                hyp_ids.clear();
                terminated_hyps.Clear();

                // Walk through the token id matrix, and assemble id, score, and
                // prev_hyp_id for each step.
                hyp_token_ids.push_back(eos_id_);
                hyp_local_scores.push_back(t_eos_scores(i, j));
                int prev_hyp_id = j;
                hyp_ids.push_back(prev_hyp_id);
                for (int k = i - 1; k >= 0; --k) {
                  hyp_token_ids.push_back(t_hyps(k, prev_hyp_id));
                  hyp_local_scores.push_back(t_scores(k, prev_hyp_id));
                  prev_hyp_id = t_prev_hyps(k, prev_hyp_id);
                  hyp_ids.push_back(prev_hyp_id);
                }

                // Assemble terminated hyp.
                terminated_hyps.set_beam_id(j % num_beams);
                for (int l = hyp_local_scores.size() - 1; l >= 0; --l) {
                  terminated_hyps.add_scores(float(hyp_local_scores[l]));
                  terminated_hyps.add_ids(hyp_token_ids[l]);
                  const int cur_step = hyp_local_scores.size() - 1 - l;
                  auto* att_vec = terminated_hyps.add_atten_vecs();
                  for (int d = 0; d < atten_probs.dim_size(2); ++d) {
                    if (l == 0) {
                      att_vec->add_prob(static_cast<float>(
                          t_eos_atten_probs(cur_step, hyp_ids[0], d)));
                    } else {
                      att_vec->add_prob(static_cast<float>(
                          t_atten_probs(cur_step, hyp_ids[l - 1], d)));
                    }
                  }
                }
                out_hyps_t(i, j) = terminated_hyps.SerializeAsString();
              }
            }
          }
        });
  }

 private:
  int32 eos_id_ = 0;
  int32 num_hyps_per_beam_ = 0;
};

REGISTER_KERNEL_BUILDER(Name("HypsFromBeamSearchOuts")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        HypsFromBeamSearchOuts<float>);
REGISTER_KERNEL_BUILDER(Name("HypsFromBeamSearchOuts")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<bfloat16>("T"),
                        HypsFromBeamSearchOuts<bfloat16>);

struct DoneHypEntry {
  // This entry corresponds to done_hyps[time_idx, hyp_idx].
  // In range [0, max_seq_length).
  int32 time_idx;
  // In range [0, b * k).
  int32 hyp_idx;
  float normalized_score;
};

struct DoneHypEntryCompare {
  bool operator()(const DoneHypEntry& x, const DoneHypEntry& y) const {
    if (x.normalized_score > y.normalized_score) return true;
    if (x.normalized_score < y.normalized_score) return false;
    // Behavior mimics BetterTerminatedHyp: We prefer shorter hyps when all else
    // being equal.
    return x.time_idx < y.time_idx;
  }
};

struct DoneHypEntryExtract {
  float operator()(const DoneHypEntry& x) const { return x.normalized_score; }
};

using TopKDoneHyp =
    TopK<DoneHypEntry, DoneHypEntryCompare, DoneHypEntryExtract>;

struct PerBeamTopK {
  mutex mu;
  TopKDoneHyp top_k TF_GUARDED_BY(mu);

  PerBeamTopK(PerBeamTopK&& other)
      : top_k(other.top_k.k(), /*unused epsilon id*/ -1) {
    mutex_lock l(other.mu);
    top_k = std::move(other.top_k);
  }

  PerBeamTopK& operator=(PerBeamTopK&& other) {
    if (this == &other) {
      return *this;
    }
    if (this < &other) {
      mutex_lock l1(mu);
      mutex_lock l2(other.mu);
      top_k = std::move(other.top_k);
      return *this;
    }
    mutex_lock l1(other.mu);
    mutex_lock l2(mu);
    top_k = std::move(other.top_k);
    return *this;
  }

  PerBeamTopK(int num_hyps_per_beam)
      : top_k(num_hyps_per_beam, /*unused epsilon id*/ -1) {}
};

template <typename T>
class TopKFromBeamSearchOutsOp : public OpKernel {
 public:
  explicit TopKFromBeamSearchOutsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_hyps_per_beam", &num_hyps_per_beam_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_seq_length", &max_seq_length_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eos_id", &eos_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("target_seq_length_ratio",
                                     &target_seq_length_ratio_));
    DCHECK_GE(target_seq_length_ratio_, 0.0);
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("populate_topk_hyps", &populate_topk_hyps_));
  }

  void Compute(OpKernelContext* ctx) override {
    // Input tensor shapes are validated in SetShapeFn().
    const float length_normalization = ctx->input(9).scalar<float>()();
    const float coverage_penalty = ctx->input(10).scalar<float>()();
    OP_REQUIRES(
        ctx, length_normalization >= 0.0 && coverage_penalty >= 0.0,
        errors::InvalidArgument("Requires non-negative length_normalization "
                                "and coverage_penalty, got: ",
                                length_normalization, ", ", coverage_penalty));

    const Tensor& hyps = ctx->input(0);
    const Tensor& prev_hyps = ctx->input(1);
    const Tensor& done_hyps = ctx->input(2);
    const Tensor& cumulative_scores = ctx->input(3);
    const Tensor& eos_scores = ctx->input(4);

    auto t_hyps = hyps.matrix<int>();
    auto t_prev_hyps = prev_hyps.matrix<int>();
    auto t_done_hyps = done_hyps.matrix<bool>();
    auto t_cumulative_scores = cumulative_scores.matrix<T>();
    auto t_eos_scores = eos_scores.matrix<T>();

    const int seq_length = hyps.dim_size(0);
    // b * k
    const int num_hyps = hyps.dim_size(1);
    const int num_beams = num_hyps / num_hyps_per_beam_;

    if (coverage_penalty > 0.0) {
      const Tensor& cumu_atten = ctx->input(8);
      OP_REQUIRES(ctx,
                  cumu_atten.dims() == 3 &&
                      cumu_atten.dim_size(0) == seq_length &&
                      cumu_atten.dim_size(1) == num_hyps,
                  errors::InvalidArgument(
                      "input tensor `cumulative_atten_probs` must have shape [",
                      seq_length, ", ", num_hyps,
                      ", ?], got shape: ", cumu_atten.shape()));
      if (populate_topk_hyps_) {
        OP_REQUIRES(
            ctx, cumu_atten.dim_size(2) == ctx->input(6).dim_size(2),
            errors::InvalidArgument(
                "input tensors `atten_probs` and `cumulative_atten_probs` must "
                "have the same shape, got shapes: atten_probs.shape=",
                ctx->input(6).shape(),
                ", cumulative_atten_probs.shape=", cumu_atten.shape()));
      }
    }

    std::vector<PerBeamTopK> topk_vec;
    topk_vec.reserve(num_beams);
    for (int i = 0; i < num_beams; ++i) {
      topk_vec.emplace_back(num_hyps_per_beam_);
    }
    ctx->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        num_hyps, /*cost_per_unit=*/seq_length << 2,
        [&](int64 begin, int64 end) {
          for (int j = begin; j < end; ++j) {
            // Input is mod ordered.
            const int beam_id = j % num_beams;
            PerBeamTopK* per_beam_topk = &topk_vec[beam_id];
            for (int i = 0; i < seq_length; ++i) {
              // If the hyp is terminated, then compute its normalized score and
              // insert into topk.
              if (t_done_hyps(i, j)) {
                auto score = t_eos_scores(i, j);
                if (i > 0) {
                  score += t_cumulative_scores(i - 1, j);
                }
                DoneHypEntry e = {.time_idx = i,
                                  .hyp_idx = j,
                                  .normalized_score = NormalizedScore(
                                      ctx, score, i, j, length_normalization,
                                      coverage_penalty)};

                mutex_lock l(per_beam_topk->mu);
                per_beam_topk->top_k.Add(e);
              }
            }
          }
        });

    // Now that we have top k hyps for each beam, we go back and assemble the
    // actual sequences.
    Tensor* out_ids;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({num_hyps, max_seq_length_}),
                                  &out_ids));
    Tensor* out_seq_lens;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({num_hyps}), &out_seq_lens));
    Tensor* out_scores;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(2, TensorShape({num_hyps}), &out_scores));
    Tensor* out_topk_hyps;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(3, TensorShape{num_beams, num_hyps_per_beam_},
                                  &out_topk_hyps));

    auto t_out_ids = out_ids->matrix<int32>();
    auto t_out_seq_lens = out_seq_lens->vec<int32>();
    auto t_out_scores = out_scores->vec<float>();
    auto t_out_topk_hyps = out_topk_hyps->matrix<tstring>();
    t_out_ids.setZero();
    t_out_seq_lens.setZero();
    t_out_scores.setZero();

    ctx->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        num_beams, /* cost_per_unit=*/4 * num_hyps_per_beam_ * max_seq_length_,
        [&](int64 begin, int64 end) {
          for (int beam_id = begin; beam_id < end; ++beam_id) {
            mutex_lock l(topk_vec[beam_id].mu);
            auto beam_topk = topk_vec[beam_id].top_k.Get();
            std::sort(beam_topk.begin(), beam_topk.end(),
                      DoneHypEntryCompare());
            for (int hyp_id = 0; hyp_id < beam_topk.size(); ++hyp_id) {
              const auto& entry = beam_topk[hyp_id];
              // div ordering.
              const int idx = beam_id * num_hyps_per_beam_ + hyp_id;
              t_out_seq_lens(idx) = entry.time_idx + 1;
              t_out_scores(idx) = entry.normalized_score;

              // Walk through the token id matrix, and assemble the sequence of
              // ids.
              t_out_ids(idx, entry.time_idx) = eos_id_;
              int prev_hyp_id = entry.hyp_idx;
              for (int time_idx = entry.time_idx - 1; time_idx >= 0;
                   --time_idx) {
                // Sanity check: previous hyp should belong to the same beam.
                DCHECK_EQ(prev_hyp_id % num_beams, beam_id);
                t_out_ids(idx, time_idx) = t_hyps(time_idx, prev_hyp_id);
                prev_hyp_id = t_prev_hyps(time_idx, prev_hyp_id);
              }

              if (populate_topk_hyps_) {
                t_out_topk_hyps(beam_id, hyp_id) =
                    GetHypProto(ctx, entry, beam_id);
              }
            }
          }
        });
  }

 private:
  float NormalizedScore(OpKernelContext* ctx, float score, int time_index,
                        int hyp_index, float length_normalization,
                        float coverage_penalty_coef) const {
    const float length = time_index + 1;
    const float length_norm = std::pow(length + 5.0, length_normalization) /
                              std::pow(5.0, length_normalization);
    float coverage_penalty = 0.0f;
    if (coverage_penalty_coef > 0.0f) {
      const auto& cumulative_atten_probs = ctx->input(8).tensor<T, 3>();
      using Index = typename Eigen::Tensor<T, 3>::Index;
      Eigen::array<Index, 3> offsets = {time_index, hyp_index, 0};
      Eigen::array<Index, 3> extents = {1, 1,
                                        cumulative_atten_probs.dimension(2)};
      const auto& cumulative_atten_prob_vec =
          cumulative_atten_probs.slice(offsets, extents);
      // Coverage is capped at 0.5 so that so long as a word is
      // reasonably covered, it is not penalized anymore.
      Tensor penalty(DT_FLOAT, {});
      penalty.scalar<float>() =
          (cumulative_atten_prob_vec / static_cast<T>(target_seq_length_ratio_))
              .cwiseMax(static_cast<T>(0.001f))
              .cwiseMin(static_cast<T>(0.5f))
              .log()
              .sum()
              .template cast<float>()
              .eval();
      coverage_penalty = target_seq_length_ratio_ * coverage_penalty_coef *
                         penalty.scalar<float>()();
    }
    return score / length_norm + coverage_penalty;
  }

  string GetHypProto(OpKernelContext* ctx, const DoneHypEntry& entry,
                     int beam_id) const {
    const Tensor& hyps = ctx->input(0);
    const Tensor& prev_hyps = ctx->input(1);
    const Tensor& eos_scores = ctx->input(4);
    const Tensor& scores = ctx->input(5);
    const Tensor& atten_probs = ctx->input(6);
    const Tensor& eos_atten_probs = ctx->input(7);

    auto t_hyps = hyps.matrix<int>();
    auto t_prev_hyps = prev_hyps.matrix<int>();
    auto t_eos_scores = eos_scores.matrix<T>();
    auto t_scores = scores.matrix<T>();
    auto t_atten_probs = atten_probs.tensor<T, 3>();
    auto t_eos_atten_probs = eos_atten_probs.tensor<T, 3>();

    Hypothesis hyp_proto;
    hyp_proto.set_beam_id(beam_id);
    hyp_proto.set_normalized_score(entry.normalized_score);
    for (int time_idx = 0; time_idx <= entry.time_idx; ++time_idx) {
      hyp_proto.add_ids(0);
      hyp_proto.add_scores(0.);
      hyp_proto.add_atten_vecs();
    }
    hyp_proto.set_ids(entry.time_idx, eos_id_);
    hyp_proto.set_scores(entry.time_idx,
                         t_eos_scores(entry.time_idx, entry.hyp_idx));
    auto* atten_vecs = hyp_proto.mutable_atten_vecs(entry.time_idx);
    for (int d = 0; d < eos_atten_probs.dim_size(2); ++d) {
      atten_vecs->add_prob(t_eos_atten_probs(entry.time_idx, entry.hyp_idx, d));
    }
    int prev_hyp_id = entry.hyp_idx;
    for (int time_idx = entry.time_idx - 1; time_idx >= 0; --time_idx) {
      hyp_proto.set_ids(time_idx, t_hyps(time_idx, prev_hyp_id));
      hyp_proto.set_scores(time_idx, t_scores(time_idx, prev_hyp_id));
      atten_vecs = hyp_proto.mutable_atten_vecs(time_idx);
      for (int d = 0; d < atten_probs.dim_size(2); ++d) {
        atten_vecs->add_prob(t_atten_probs(time_idx, prev_hyp_id, d));
      }
      prev_hyp_id = t_prev_hyps(time_idx, prev_hyp_id);
    }
    return hyp_proto.SerializeAsString();
  }

  int32 num_hyps_per_beam_;
  int32 max_seq_length_;
  int32 eos_id_;
  float target_seq_length_ratio_;
  bool populate_topk_hyps_;

  TF_DISALLOW_COPY_AND_ASSIGN(TopKFromBeamSearchOutsOp);
};

REGISTER_KERNEL_BUILDER(Name("TopKFromBeamSearchOuts")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        TopKFromBeamSearchOutsOp<float>);
REGISTER_KERNEL_BUILDER(Name("TopKFromBeamSearchOuts")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<bfloat16>("T"),
                        TopKFromBeamSearchOutsOp<bfloat16>);

}  // namespace lingvo
}  // namespace tensorflow
