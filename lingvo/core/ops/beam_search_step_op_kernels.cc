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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/work_sharder.h"
#include "lingvo/core/ops/hyps.pb.h"
#include "lingvo/core/ops/simple_vocab.h"

namespace tensorflow {
namespace lingvo {

namespace {
constexpr int kNumWorkers = 8;
}  // namespace

namespace debug {
static string IdsToStr(const google::protobuf::RepeatedField<int32>& ids) {
  return debug::IdsToStr(std::vector<int32>(ids.begin(), ids.end()));
}
}  // namespace debug

bool IdsMatchUpToIndex(const std::vector<int>& cur_hyp_ids,
                       const std::vector<int>& other_hyp_ids, const int index) {
  CHECK_LE(index, cur_hyp_ids.size());
  CHECK_LE(index, other_hyp_ids.size());
  return std::equal(cur_hyp_ids.begin(), cur_hyp_ids.begin() + index,
                    other_hyp_ids.begin());
}

bool IsDuplicateHyp(const Hyp& cur_hyp, const Hyp& other_hyp,
                    const int epsilon_id) {
  const std::vector<int>& cur_hyp_ids = cur_hyp.prev_ids;
  const std::vector<int>& other_hyp_ids = other_hyp.prev_ids;
  // Note word_id refers to id of current label, which could be grapheme,
  // phoneneme, wordpiece, ectc.
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
// the predicted next step scores 'scores', return the best scored 'k+m'
// hypotheses where the first 'k' hypotheses are used for search in the next
// step while the remaining 'm' hypotheses are kept as alternatives to increase
// diversity. 'eos_id' is the end of beam id of the target language.
//
// eos_in_topk is filled with true/false to indicate whether or not the eos
// symbol is among the topk candidate for a hyp.
void ComputeTopKPlusM(const std::vector<Hyp>& hyps, const Tensor& scores,
                      const int32 k, const int32 m, const int32 eos_id,
                      const int32 eoc_id, const int32 num_beams,
                      const float valid_eos_max_logit_delta, bool is_first_step,
                      bool is_last_decoder_step, const Tensor& is_last_chunk,
                      bool merge_paths, bool allow_empty_terminated_hyp,
                      std::vector<bool>* eos_in_topk, std::vector<Hyp>* top_k,
                      std::vector<Hyp>* extra_m, std::vector<Hyp>* eos_hyps,
                      std::vector<int32>* terminal_syms) {
  VLOG(1) << "Topk clear, num_beams: " << num_beams;
  CHECK_EQ(hyps.size(), num_beams * k);
  CHECK_GE(m, 0);
  CHECK(eos_in_topk && top_k && extra_m && eos_hyps && terminal_syms);
  CHECK_EQ(hyps.size(), scores.dim_size(0));
  CHECK_LT(eos_id, scores.dim_size(1));
  int hyps_size = hyps.size();
  eos_in_topk->clear();
  top_k->clear();
  extra_m->clear();
  eos_in_topk->resize(hyps_size);
  eos_hyps->resize(hyps_size);
  terminal_syms->resize(hyps_size);
  static thread::ThreadPool* workers =
      new thread::ThreadPool(Env::Default(), "topk", kNumWorkers);
  const int num_ids = scores.dim_size(1);
  const auto scores_matrix = scores.matrix<float>();
  const int epsilon_id_for_path_merging = merge_paths ? eoc_id : -1;
  std::vector<
      TopK<Hyp, HigherScore, ExtractGlobalScore, InsertHypWithEpsilonDedupe>>
      merged_topk_vec(num_beams, TopK<Hyp, HigherScore, ExtractGlobalScore,
                                      InsertHypWithEpsilonDedupe>(
                                     m + k, epsilon_id_for_path_merging));
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
          // +1 to make sure that at least top-k hypotheses survive even with
          // the special treatment for eos.  +2 if we are also using eoc.
          const int topk_size = eoc_id >= 0 ? k + 2 : k + 1;
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
                                global_score, hyps[hyp_id].prev_ids});
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
                            global_score, hyps[hyp_id].prev_ids});
            }
          }

          auto entries = topk.Get();
          CHECK(!entries.empty()) << "No entries in TopK. This typically " <<
              "happens if your model is producing NaNs in the output.";
          std::sort(entries.begin(), entries.end(), HigherScore());
          const float eos_score_threshold =
              entries[0].global_score - valid_eos_max_logit_delta;
          VLOG(3) << "Best_score=" << entries[0].global_score
                  << " eos_score_threshold=" << eos_score_threshold;
          {
            const int beam_id = hyps[hyp_id].beam_id;
            mutex_lock l(mu_vec[beam_id]);
            for (const auto& e : entries) {
              if (e.word_id == eos_id) {
                VLOG(3) << "EOS hyp score=" << e.global_score
                        << " toks=" << debug::IdsToStr(e.prev_ids);
                // We move terminated hyps off of the beam.
                if (is_last_decoder_step ||
                    (e.global_score > eos_score_threshold)) {
                  (*eos_in_topk)[hyp_id] = true;
                  (*eos_hyps)[hyp_id] = e;
                  (*terminal_syms)[hyp_id] = eos_id;
                }
              } else if (eoc_id >= 0 && is_last_chunk.vec<bool>()(hyp_id) &&
                         e.word_id == eoc_id) {
                VLOG(3) << "last chunk hyp score=" << e.global_score
                        << " toks=" << debug::IdsToStr(e.prev_ids);
                // At the last chunk and output <epsilon>. We terminate the
                // hypothesis, even though <eos> was not predicted, and
                // indicate that the final symbol for the hypothesis is
                // <epsilon>, not <eos>.
                if (e.global_score > eos_score_threshold &&
                    // Only allow an empty hyp (all <epsilon>s) to be
                    // considered terminated, if explicitly permitted.
                    // 'prev_ids' contains only non-epsilons.
                    (allow_empty_terminated_hyp || !e.prev_ids.empty())) {
                  (*eos_in_topk)[hyp_id] = true;
                  (*eos_hyps)[hyp_id] = e;
                  (*terminal_syms)[hyp_id] = eoc_id;
                }
              } else {
                merged_topk_vec[beam_id].Add(e);
              }
            }
          }
        }
      });

  const int hyps_per_beam = k;
  top_k->resize(num_beams * hyps_per_beam);
  for (int i = 0; i < num_beams; ++i) {
    auto ith_topk = merged_topk_vec[i].Get();
    std::sort(ith_topk.begin(), ith_topk.end(), HigherScore());
    const int num_hyps =
        std::min(static_cast<int>(ith_topk.size()), hyps_per_beam);
    for (int j = 0; j < num_hyps; ++j) {
      (*top_k)[j * num_beams + i] = ith_topk[j];
    }
    for (int j = hyps_per_beam; j < ith_topk.size(); ++j) {
      extra_m->push_back(ith_topk[j]);
    }
  }
  VLOG(1) << "Topk done";
}

class BeamSearchStepOp : public OpKernel {
 public:
  explicit BeamSearchStepOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eos_id", &eos_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eoc_id", &eoc_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_size", &beam_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_hyps_per_beam", &num_hyps_per_beam_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("valid_eos_max_logit_delta",
                                     &valid_eos_max_logit_delta_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_paths", &merge_paths_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("allow_empty_terminated_hyp",
                                     &allow_empty_terminated_hyp_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ensure_full_beam", &ensure_full_beam_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("force_eos_in_last_step", &force_eos_in_last_step_));

    CHECK_GE(eos_id_, 0);
    CHECK_GT(beam_size_, 0.0);
    CHECK_GT(num_hyps_per_beam_, 0);
  }

 private:
  Tensor* ForwardOrCopyInputToOutput(OpKernelContext* ctx, int input_idx,
                                     int output_idx) {
    Tensor* output = nullptr;
    const Tensor& input = ctx->input(input_idx);
    CHECK(ctx->forward_input_or_allocate_output({input_idx}, output_idx,
                                                input.shape(), &output)
              .ok());
    if (!output->SharesBufferWith(input)) {
      // Copy the input data if we were unable to forward the underlying buffer.
      if (DataTypeCanUseMemcpy(input.dtype())) {
        if (input.NumElements() > 0) {
          StringPiece input_data = input.tensor_data();
          StringPiece output_data = output->tensor_data();
          memcpy(const_cast<char*>(output_data.data()), input_data.data(),
                 input_data.size());
        }
      } else if (input.dtype() == DT_STRING) {
        output->flat<string>() = input.flat<string>();
      }
    }
    return output;
  }

  string AssembleDoneHyp(const Hyp& hyp, const int32 terminal_sym,
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
      // we simpy take an average per-step score.
      const float score_this_step =
          (merge_paths_ ? average_step_score : t_out_scores(i, hyp_id));
      hypothesis.add_scores(score_this_step);
      auto* att_vec = hypothesis.add_atten_vecs();
      for (int j = 0; j < atten_probs.dim_size(1); ++j) {
        att_vec->add_prob(t_out_atten_probs(i, hyp_id, j));
      }
    }
    // Now add the terminal symbol.
    hypothesis.add_ids(terminal_sym);
    // As above, use the average per-step score for RNN-T and NT.
    const float score_this_step =
        merge_paths_ ? average_step_score : hyp.local_score;
    hypothesis.add_scores(score_this_step);
    auto* att_vec = hypothesis.add_atten_vecs();
    auto t_atten_probs = atten_probs.matrix<float>();
    for (int j = 0; j < atten_probs.dim_size(1); ++j) {
      att_vec->add_prob(t_atten_probs(hyp.hyp_id, j));
    }
    return hypothesis.SerializeAsString();
  }

 public:
  void Compute(OpKernelContext* ctx) override {
    const Tensor& scores = ctx->input(0);
    const Tensor& atten_probs = ctx->input(1);
    const Tensor& best_scores = ctx->input(2);
    const Tensor& cumulative_scores = ctx->input(3);
    const Tensor& in_scores = ctx->input(4);
    const Tensor& in_hyps = ctx->input(5);
    const Tensor& in_prev_hyps = ctx->input(6);
    const Tensor& in_done_hyps = ctx->input(7);
    const Tensor& in_atten_probs = ctx->input(8);
    const Tensor& is_last_chunk = ctx->input(9);
    const Tensor& cur_step = ctx->input(10);

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

    if (merge_paths_) {
      OP_REQUIRES(
          ctx, eoc_id_ >= 0,
          errors::InvalidArgument(
              "Paths can only be merged for an epsilon-emitting model (RNN-T "
              "or NT).  Epsilon id must be non-negative, but got: ",
              eoc_id_));
    }
    int num_beams = best_scores.dim_size(0);
    int num_hyps = cumulative_scores.dim_size(0);
    CHECK_EQ(num_hyps_per_beam_, num_hyps / num_beams);
    const int t = cur_step.scalar<int>()();
    CHECK_LT(t, in_hyps.dim_size(0));

    VLOG(2) << "BeamSearchStepOp(" << num_hyps_per_beam_ << ") step=" << t;
    auto t_cumulative_scores = cumulative_scores.vec<float>();
    std::vector<Hyp> hyps(num_hyps);
    for (int i = 0; i < num_hyps; ++i) {
      hyps[i].beam_id = i % num_beams;
      hyps[i].hyp_id = i;
      hyps[i].global_score = t_cumulative_scores(i);
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
          hyps[i].prev_ids.push_back(prev_id);
        }
      }
      VLOG(3) << "Step " << t << " hyp " << i
              << " score=" << hyps[i].global_score
              << " toks=" << debug::IdsToStr(hyps[i].prev_ids);
    }
    std::vector<Hyp> top_k_hyps;
    std::vector<Hyp> extra_m_hyps;
    std::vector<Hyp> eos_hyps;
    std::vector<bool> eos_in_topk;
    std::vector<int32> terminal_syms;
    const bool is_last_decoder_step =
        (t == (in_hyps.dim_size(0) - 1)) && force_eos_in_last_step_;
    ComputeTopKPlusM(hyps, scores, num_hyps_per_beam_, 0, eos_id_, eoc_id_,
                     num_beams, valid_eos_max_logit_delta_, t == 0,
                     is_last_decoder_step, is_last_chunk, merge_paths_,
                     allow_empty_terminated_hyp_, &eos_in_topk, &top_k_hyps,
                     &extra_m_hyps, &eos_hyps, &terminal_syms);

    Tensor* out_best_scores = NULL;
    Tensor* out_cumulative_scores = NULL;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, best_scores.shape(), &out_best_scores));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, cumulative_scores.shape(),
                                             &out_cumulative_scores));
    Tensor* out_scores = ForwardOrCopyInputToOutput(ctx, 4, 2);
    Tensor* out_hyps = ForwardOrCopyInputToOutput(ctx, 5, 3);
    Tensor* out_prev_hyps = ForwardOrCopyInputToOutput(ctx, 6, 4);
    Tensor* out_done_hyps = ForwardOrCopyInputToOutput(ctx, 7, 5);
    Tensor* out_atten_probs = ForwardOrCopyInputToOutput(ctx, 8, 6);
    Tensor* all_done;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(7, TensorShape({}), &all_done));

    auto t_out_best_scores = out_best_scores->vec<float>();
    auto t_out_cumulative_scores = out_cumulative_scores->vec<float>();
    auto t_out_scores = out_scores->matrix<float>();
    auto t_out_hyps = out_hyps->matrix<int>();
    auto t_out_prev_hyps = out_prev_hyps->matrix<int>();
    auto t_out_done_hyps = out_done_hyps->matrix<string>();
    auto t_out_atten_probs = out_atten_probs->tensor<float, 3>();
    auto t_all_done = all_done->scalar<bool>();

    // To initialize the two vectors.
    t_out_best_scores = best_scores.vec<float>();
    t_out_cumulative_scores = cumulative_scores.vec<float>();

    // Fill in all the output tensors.
    for (int i = 0; i < num_hyps; ++i) {
      const Hyp& hyp = top_k_hyps[i];
      t_out_scores(t, i) = hyp.local_score;
      t_out_cumulative_scores(i) = hyp.global_score;
      t_out_hyps(t, i) = hyp.word_id;
      t_out_prev_hyps(t, i) = hyp.hyp_id;
      t_out_atten_probs.chip(t, 0).chip(i, 0) =
          atten_probs.matrix<float>().chip(hyp.hyp_id, 0);
      if (eos_in_topk[i]) {
        // We have a good terminated hyp.
        const int beam_id = eos_hyps[i].beam_id;
        CHECK_EQ(beam_id, i % num_beams);
        VLOG(2) << "Top EOS hyp @step " << t
                << " score=" << eos_hyps[i].global_score
                << " toks=" << debug::IdsToStr(eos_hyps[i].prev_ids);
        // Update the best scores.
        if (eos_hyps[i].global_score > t_out_best_scores(beam_id)) {
          t_out_best_scores(beam_id) = eos_hyps[i].global_score;
        }
        string done_hyp = AssembleDoneHyp(
            eos_hyps[i], terminal_syms[i], t_out_prev_hyps, t_out_hyps,
            t_out_scores, t_out_atten_probs, atten_probs, t);
        t_out_done_hyps(t, i) = done_hyp;
      }
    }

    // Now check for all_done
    t_all_done() = true;
    if (ensure_full_beam_) {
      // First check how many EOS hyps we have.  If we have fewer than
      // num_hyps_per_beam for any beam, we are NOT done.
      for (int beam_id = 0; beam_id < num_beams; ++beam_id) {
        int num_done_hyps = 0;
        for (int index_in_beam = 0; index_in_beam < num_hyps_per_beam_;
             ++index_in_beam) {
          for (int time_step = 0; time_step < t; ++time_step) {
            int index = beam_id * num_hyps_per_beam_ + index_in_beam;;
            if (!t_out_done_hyps(time_step, index).empty()) {
              ++num_done_hyps;
            }
          }
        }
        if (num_done_hyps < num_hyps_per_beam_) {
          t_all_done() = false;
          break;
        }
      }
      if (t_all_done() == false) return;
    }
    // Now check for hyp quality.  If for any beam we still have hyps within
    // 'beam_size' of best score, we are NOT done.
    for (int i = 0; i < num_hyps; ++i) {
      const Hyp& hyp = top_k_hyps[i];
      const int beam_id = hyp.beam_id;
      CHECK_EQ(beam_id, i % num_beams);
      VLOG(3) << "Hyp score=" << hyp.global_score
              << " beam best=" << t_out_best_scores(beam_id)
              << " beam size=" << beam_size_;
      if (hyp.global_score > t_out_best_scores(beam_id) - beam_size_) {
        t_all_done() = false;
        break;
      }
    }
  }

 private:
  int eos_id_ = 0;
  int eoc_id_ = -1;
  float beam_size_ = 0.0;
  int num_hyps_per_beam_ = 0;
  float valid_eos_max_logit_delta_ = 0.0;
  bool merge_paths_ = false;
  bool allow_empty_terminated_hyp_ = true;
  bool ensure_full_beam_ = false;
  bool force_eos_in_last_step_ = false;
};

REGISTER_KERNEL_BUILDER(Name("BeamSearchStep").Device(DEVICE_CPU),
                        BeamSearchStepOp);

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
    // TODO(anjuli): Remove eoc_id_ which is no longer used.
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eoc_id", &eoc_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_paths", &merge_paths_));
    CHECK_GE(length_normalization_, 0.0);
    CHECK_GE(target_seq_length_ratio_, 0.0);
    CHECK_GE(coverage_penalty_, 0);
    CHECK_GT(num_hyps_per_beam_, 0);
    CHECK_GT(k_, 0);
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
    auto t_done_hyps = in_done_hyps.matrix<string>();
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
                    VLOG(2) << "Add to terminated top-k "
                            << " score=" << hypothesis.normalized_score()
                            << " toks=" << debug::IdsToStr(hypothesis.ids());
                    // TODO(xbing): avoid acquiring a mutex for each record.
                    mutex_lock l(mu_vec[hyp_id % num_beams]);
                    topk->Add(hypothesis);
                  }
                }
              }
            }
          });

    auto t_topk_hyps = topk_hyps->matrix<string>();
    for (int i = 0; i < num_beams; ++i) {
      auto ith_topk = topk_vec[i].Get();
      CHECK_LE(ith_topk.size(), k);
      std::sort(ith_topk.begin(), ith_topk.end(), BetterTerminatedHyp());
      for (int j = 0; j < ith_topk.size(); ++j) {
        t_topk_hyps(i, j) = ith_topk[j].SerializeAsString();
        VLOG(2) << "TopK(" << i << ", " << j << ") = "
                << debug::IdsToStr(ith_topk[j].ids());
      }
    }
  }

  float NormalizedScore(const Hypothesis& hypothesis,
                        const int src_size) const {
    int length = hypothesis.atten_vecs_size();
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
    const float coverage_penalty = penalty.scalar<float>()();
    const float length_norm = std::pow(length + 5.0, length_normalization_) /
                              std::pow(5.0, length_normalization_);

    float global_score = 0.0;
    for (const auto& score : hypothesis.scores()) {
      global_score += score;
    }
    const float normalized_score =
        global_score / length_norm +
        (target_seq_length_ratio_ * coverage_penalty_ * coverage_penalty);
    return normalized_score;
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
            in_src_seq_lens.dims(), " vs ", num_beams));
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
  int32 eoc_id_;
  bool merge_paths_ = false;
};

REGISTER_KERNEL_BUILDER(Name("TopKTerminatedHyps").Device(DEVICE_CPU),
                        TopKTerminatedHypsOp);

class UnpackHypOp : public OpKernel {
 public:
  explicit UnpackHypOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_seq_length", &max_seq_length_));
    CHECK_GE(max_seq_length_, 0);
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in_hyps = ctx->input(0);
    const auto& t_in_hyps = in_hyps.flat<string>();
    const int batch_size = t_in_hyps.size();
    std::vector<Hypothesis> hyps(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      // TODO(yonghui): parallelize this loop.
      if (!t_in_hyps(i).empty()) {
        hyps[i].ParseFromString(t_in_hyps(i));
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
            "Failed tensor shape sanity check. atten_probs.dims() == 2. Got ",
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

    OP_REQUIRES(ctx, hyps.IsSameSize(done_hyps),
                errors::InvalidArgument(
                    "Failed tensor shape sanity check. "
                    "hyps and done_hyps should have the same shape. Got ",
                    hyps.shape().DebugString(), " and ",
                    done_hyps.shape().DebugString()));

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
    auto out_hyps_t = out_hyps->matrix<string>();

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
                      att_vec->add_prob(
                          float(t_eos_atten_probs(cur_step, hyp_ids[l], d)));
                    } else {
                      att_vec->add_prob(
                          float(t_atten_probs(cur_step, hyp_ids[l], d)));
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

}  // namespace lingvo
}  // namespace tensorflow
