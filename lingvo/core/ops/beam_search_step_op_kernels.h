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

#ifndef LINGVO_CORE_OPS_BEAM_SEARCH_STEP_OP_KERNELS_H_
#define LINGVO_CORE_OPS_BEAM_SEARCH_STEP_OP_KERNELS_H_

#include <algorithm>   // std::sort
#include <functional>  // std::greater
#include <vector>

#include "lingvo/core/ops/hyps.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace lingvo {

// Simple tuple for book keeping during beam pruning.
struct Hyp {
  // The beam that this hyp belongs to.
  int32 beam_id;
  // The hypothesis id within a beam, ranges [0, num_hyps_per_beam).
  int32 hyp_id;
  // The id for the predicted next word (a.k.a expansion word, can be epsilon).
  int32 word_id;
  // Local score from the current step for word_id.
  float local_score;
  // Cumulative score up to the current step (inclusive).
  float global_score;
  // The (non-epsilon) token ids up to this step (before expansion).
  std::vector<int32> prev_labels;

  string DebugString() const {
    return strings::StrCat("beam_id=", beam_id, " hyp_id=", hyp_id,
                           " word_id=", word_id, " local_score=", local_score,
                           " global_score=", global_score,
                           " prev_labels=", str_util::Join(prev_labels, ","));
  }
};

struct HigherScore {
  bool operator()(const Hyp& x, const Hyp& y) const {
    // We only compare hyps belonging to the same beams.
    DCHECK_EQ(x.beam_id, y.beam_id);
    if (x.global_score > y.global_score) return true;
    if (x.global_score < y.global_score) return false;
    if (x.word_id < y.word_id) return true;
    if (x.word_id > y.word_id) return false;
    return x.hyp_id < y.hyp_id;
  }
};

// Similar to HigherScore, but optionally return Hyps with eos_id in it
// as the one with the higher score.
struct HigherScoreWithEos {
  int eos_id_;
  // Determines whether or not to prioritize Hyps with eos_id. If False,
  // this will behave exactly like HigherScore.
  bool include_eos_;

  HigherScoreWithEos(int eos_id, bool include_eos)
      : eos_id_(eos_id), include_eos_(include_eos) {}

  bool operator()(const Hyp& x, const Hyp& y) const {
    // We only compare hyps belonging to the same beams.
    DCHECK_EQ(x.beam_id, y.beam_id);
    // Note that we revert to HigherScore's behavior when _both_ paths contain
    // eos_id_ as the word_id.
    if (!(x.word_id == eos_id_ && y.word_id == eos_id_)) {
      if (x.word_id == eos_id_ && include_eos_) return true;
      if (y.word_id == eos_id_ && include_eos_) return false;
    }
    // The following behavior is the same as that of HigherScore's.
    if (x.global_score > y.global_score) return true;
    if (x.global_score < y.global_score) return false;
    if (x.word_id < y.word_id) return true;
    if (x.word_id > y.word_id) return false;
    return x.hyp_id < y.hyp_id;
  }
};

struct BetterTerminatedHyp {
  bool operator()(const Hypothesis& x, const Hypothesis& y) const {
    // We only compare hyps belonging to the same beams.
    DCHECK_EQ(x.beam_id(), y.beam_id());
    if (x.normalized_score() > y.normalized_score()) return true;
    if (x.normalized_score() < y.normalized_score()) return false;
    return x.ids_size() < y.ids_size();
  }
};

struct ExtractGlobalScore {
  float operator()(const Hyp& x) const { return x.global_score; }
};

struct ExtractNormalizedScore {
  float operator()(const Hypothesis& x) const { return x.normalized_score(); }
};

template <typename T>
struct Id {
  const T& operator()(const T& t) const { return t; }
};

template <typename T>
struct DefaultInsert {
  explicit DefaultInsert(int unused_epsilon_id) {}
  void operator()(const T& t, std::vector<T>* items) const {
    items->push_back(t);
  }
};

// Returns true if 'cur_hyp' ad 'other_hyp' represent the same label sequence
// when epsilons are ignored, and false otherwise.
bool IsDuplicateHyp(const Hyp& cur_hyp, const Hyp& other_hyp,
                    const int epsilon_id);

float LogSumExp(float a, float b);

// An insertion operator that first checks whether 'hyp'  is a duplicate of any
// hyp already in 'items'.  If so, these two hyps are merged.
// This check is only performed if we are using a model that emits epsilons
// (NT or RNN-T).  For models that do not emit epsilons (ie epsilon_id < 0)
// 'hyp' is always added to 'items', identical to DefaultInsert.
struct InsertHypWithEpsilonDedupe {
  explicit InsertHypWithEpsilonDedupe(int _epsilon_id)
      : epsilon_id(_epsilon_id), better_hyp() {}
  void operator()(const Hyp& hyp, std::vector<Hyp>* items) const {
    if (epsilon_id < 0) {
      items->push_back(hyp);
      return;
    }
    for (int i = 0; i < items->size(); ++i) {
      const Hyp& old_hyp = (*items)[i];
      if (IsDuplicateHyp(hyp, old_hyp, epsilon_id)) {
        VLOG(3) << "merging:";
        VLOG(3) << "hyp1=[" << hyp.DebugString() << "]";
        VLOG(3) << "hyp2=[" << old_hyp.DebugString() << "]";
        Hyp combined_hyp = better_hyp(hyp, old_hyp) ? hyp : old_hyp;
        combined_hyp.global_score =
            LogSumExp(hyp.global_score, old_hyp.global_score);
        (*items)[i] = combined_hyp;
        VLOG(3) << "combined=[" << combined_hyp.DebugString() << "]";
        return;
      }
    }
    items->push_back(hyp);
  }
  int epsilon_id;
  const HigherScore better_hyp;
};


// A helper class keeps track of top K highest ranked elements added.
// Comp(x, y) returns true iff x is ranked higher than y.
// Epsilon id should be set to -1 for models which do not use epsilon (e.g. LAS
// and any non-speech model), or to the id used for epsilon (end of chunk)
// for epsilon emitting models (RNN-T, NT).
//
// E.g.,
//    TopK topk<int32, std::less<int32>> (100, /* epsilon id */ -1);
//    topk.Add(100);
//    topk.Add(-100);
//    ...
//    result = topk.Get();
//    // results contains the smallest 100 int added to topk.
template <typename T, typename Comp = std::greater<T>, typename Extract = Id<T>,
          typename Insert = DefaultInsert<T>>
class TopK {
 public:
  explicit TopK(int k, int epsilon_id, int buffer_size_factor = 2)
      : k_(k), comp_(), extract_(), insert_(epsilon_id), selected_(false),
        buffer_size_factor_(buffer_size_factor) {}
  // eos_id and inlclude_eos flag will be passed on to the Comparator.
  explicit TopK(int k, int epsilon_id, int eos_id, bool include_eos,
                int buffer_size_factor = 2)
      : k_(k),
        comp_(Comp(eos_id, include_eos)),
        extract_(),
        insert_(epsilon_id),
        selected_(false),
        buffer_size_factor_(buffer_size_factor) {}

  int k() const { return k_; }

  using U = typename std::result_of<Extract(T)>::type;

  // Return an element that is less than or equal to the least element
  // of the top k.
  U Add(const T& e) {
    if (!selected_ || comp_(e, items_[k_ - 1])) {
      insert_(e, &items_);
      if (items_.size() >= buffer_size_factor_ * k_) Shrink();
    }
    if (!selected_) return std::numeric_limits<U>::lowest();
    return extract_(items_[k_ - 1]);
  }

  const std::vector<T>& Get() {
    if (items_.size() > k_) Shrink();
    return items_;
  }

  void Clear() {
    selected_ = false;
    items_.clear();
  }

 private:
  int k_;
  Comp comp_;
  Extract extract_;
  Insert insert_;
  bool selected_;  // Becomes true if k-th top element so far is known.
  std::vector<T> items_;
  // Buffer size is buffer_size_factor_ * k before invoking Shrink() in Add().
  // A reasonable default value is 2.
  int buffer_size_factor_;

  void Shrink() {
    // Pivot is the k-th element, i.e., items_[k_-1].
    std::nth_element(items_.begin(), items_.begin() + k_ - 1, items_.end(),
                     comp_);
    items_.resize(k_);
    selected_ = true;
  }
};

// Exposed for benchmarking purposes.
//
// Please see the definition for function comments.
void ComputeTopK(int step, const std::vector<Hyp>& hyps, const Tensor& scores,
                 const int32 k, const int32 eos_id, const int32 eoc_id,
                 const int32 num_beams, const float valid_eos_max_logit_delta,
                 const float local_eos_threshold, bool is_first_step,
                 bool is_last_decoder_step, const Tensor& is_last_chunk,
                 bool merge_paths, bool allow_empty_terminated_hyp,
                 bool force_eos_in_top_k, bool force_last_chunk_eoc_in_top_k,
                 int merged_topk_buffer_size_factor,
                 const std::vector<bool>& skip_beam,
                 std::vector<char>* eos_in_topk, std::vector<Hyp>* top_k,
                 std::vector<Hyp>* eos_hyps,
                 std::vector<int32>* terminal_symbols);

}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_BEAM_SEARCH_STEP_OP_KERNELS_H_
