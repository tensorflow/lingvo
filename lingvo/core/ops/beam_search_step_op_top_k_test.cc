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

#include <gmock/gmock.h>
#include "lingvo/core/ops/beam_search_step_op_kernels.h"

namespace tensorflow {
namespace lingvo {
namespace {

using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::SizeIs;

template <typename T>
void Populate(T* top_k) {
  // Add 3 distinct hyps.
  float bottom_of_topk;
  // Hyp struct consists of:
  //  beam_id, hyp_id, word_id, local_score, global_score, prev_ids
  bottom_of_topk = top_k->Add({0, 2, 3, -0.4, -2.0, {1, 2, 3}});
  EXPECT_THAT(bottom_of_topk, FloatEq(std::numeric_limits<float>::lowest()));
  bottom_of_topk = top_k->Add({0, 1, 8, -0.7, -1.3, {1, 7, 2}});
  EXPECT_THAT(bottom_of_topk, FloatEq(std::numeric_limits<float>::lowest()));
  bottom_of_topk = top_k->Add({0, 6, 2, -0.1, -1.7, {1, 9, 5}});
  // No resize yet, since we haven't gotten to k * 2 = 4 elements yet.
  EXPECT_THAT(bottom_of_topk, FloatEq(std::numeric_limits<float>::lowest()));
  bottom_of_topk = top_k->Add({0, 4, 3, -0.5, -2.5, {1, 2, 4}});
  // After fourth element we resize down to two best elements.
  EXPECT_THAT(bottom_of_topk, FloatNear(-1.7, 0.001));

  const auto& hyps = top_k->Get();
  EXPECT_THAT(hyps, SizeIs(2));
  EXPECT_THAT(hyps[0].hyp_id, Eq(1));
  EXPECT_THAT(hyps[0].global_score, FloatNear(-1.3, 0.001));
  EXPECT_THAT(hyps[1].hyp_id, Eq(6));
  EXPECT_THAT(hyps[1].global_score, FloatNear(-1.7, 0.001));

  // Add a dupe.
  top_k->Add({0, 5, 8, -0.7, -1.5, {1, 7, 2}});
}

template <typename T>
void PopulateWithEpsilons(T* top_k) {
  // Add 3 distinct hyps.
  float bottom_of_topk;
  bottom_of_topk = top_k->Add({0, 2, 3, -0.4, -2.0, {1, 2, 3}});
  EXPECT_THAT(bottom_of_topk, FloatEq(std::numeric_limits<float>::lowest()));
  bottom_of_topk = top_k->Add({0, 1, 8, -0.7, -1.3, {1, 7, 2}});
  EXPECT_THAT(bottom_of_topk, FloatEq(std::numeric_limits<float>::lowest()));
  bottom_of_topk = top_k->Add({0, 6, 2, -0.1, -1.7, {1, 3, 4, 9, 5}});
  // No resize yet, since we haven't gotten to k * 2 = 4 elements yet.
  EXPECT_THAT(bottom_of_topk, FloatEq(std::numeric_limits<float>::lowest()));
  bottom_of_topk = top_k->Add({0, 4, 3, -0.5, -2.5, {1, 2, 4, 3}});
  // After fourth element we resize down to two best elements.
  EXPECT_THAT(bottom_of_topk, FloatNear(-1.7, 0.001));

  const auto& hyps = top_k->Get();
  EXPECT_THAT(hyps, SizeIs(2));
  EXPECT_THAT(hyps[0].hyp_id, Eq(1));
  EXPECT_THAT(hyps[0].global_score, FloatNear(-1.3, 0.001));
  EXPECT_THAT(hyps[1].hyp_id, Eq(6));
  EXPECT_THAT(hyps[1].global_score, FloatNear(-1.7, 0.001));

  // Add a dupe.  Doing the dedupe requires comparing last prev id of this hyp
  // with cur id of candidate hyp since cur id for this hyp is epsilon.
  const int epsilon_id = 0;
  bottom_of_topk = top_k->Add({0, 5, epsilon_id, -0.7, -1.5, {1, 7, 2, 8}});
}

bool IsDupe(const Hyp& hyp1, const Hyp& hyp2) {
  if (hyp1.word_id != hyp2.word_id) {
    return false;
  }
  if (hyp1.prev_ids.size() != hyp2.prev_ids.size()) {
    return false;
  }
  return std::equal(hyp1.prev_ids.begin(), hyp1.prev_ids.end(),
                    hyp2.prev_ids.begin());
}

// Tests that when we use the default Insert, there is NO deduping.
TEST(TopKTest, TestInsertDefault) {
  const int k = 2;
  TopK<Hyp, HigherScore, ExtractGlobalScore> top_k(k, /* epsilon id */ -1);
  Populate<TopK<Hyp, HigherScore, ExtractGlobalScore>>(&top_k);
  // Check contents: The two items in the TopK are duplicates.
  const auto& new_hyps = top_k.Get();
  EXPECT_THAT(new_hyps, SizeIs(2));
  EXPECT_THAT(new_hyps[0].hyp_id, Eq(1));
  EXPECT_THAT(new_hyps[1].hyp_id, Eq(5));
  EXPECT_TRUE(IsDupe(new_hyps[0], new_hyps[1]));
}

// Tests that when we use InsertHypWithEpsilonDedupe but set the epsilon id
// to be less than zero, there is NO deduping.
TEST(TopKTest, TestInsertNoDedupe) {
  const int k = 2;
  TopK<Hyp, HigherScore, ExtractGlobalScore, InsertHypWithEpsilonDedupe> top_k(
      k, /* epsilon_id */ -1);
  Populate<
      TopK<Hyp, HigherScore, ExtractGlobalScore, InsertHypWithEpsilonDedupe>>(
      &top_k);
  // Check contents: The two items in the TopK are duplicates.
  const auto& new_hyps = top_k.Get();
  EXPECT_THAT(new_hyps, SizeIs(2));
  EXPECT_THAT(new_hyps[0].hyp_id, Eq(1));
  EXPECT_THAT(new_hyps[1].hyp_id, Eq(5));
  EXPECT_TRUE(IsDupe(new_hyps[0], new_hyps[1]));
}

// Tests that when we use InsertHypWithEpsilonDedupe and set the epsilon id
// to be greater than or equal to zero, there IS deduping.
TEST(TopKTest, TestInsertWithDedupe) {
  const int k = 2;
  const int epsilon_id = 0;
  TopK<Hyp, HigherScore, ExtractGlobalScore, InsertHypWithEpsilonDedupe> top_k(
      k, epsilon_id);
  Populate<
      TopK<Hyp, HigherScore, ExtractGlobalScore, InsertHypWithEpsilonDedupe>>(
      &top_k);
  // Check contents: The two items in the TopK are not duplicates.  The dupe
  // has been merged into one of them.
  const auto& new_hyps = top_k.Get();
  EXPECT_THAT(new_hyps, SizeIs(2));
  EXPECT_THAT(new_hyps[0].hyp_id, Eq(1));
  // Merging has happened so combined global score is:
  //   log(exp(-1.5) + exp(-1.3))
  EXPECT_THAT(new_hyps[0].global_score, FloatNear(-0.70186, 0.001));
  EXPECT_THAT(new_hyps[1].hyp_id, Eq(6));
  EXPECT_THAT(new_hyps[1].global_score, FloatNear(-1.7, 0.001));
  EXPECT_TRUE(!IsDupe(new_hyps[0], new_hyps[1]));
}

// Tests that the deduping ignores epsilons.
TEST(TopKTest, TestInsertWithDedupeEpsilon) {
  const int k = 2;
  const int epsilon_id = 0;
  TopK<Hyp, HigherScore, ExtractGlobalScore, InsertHypWithEpsilonDedupe> top_k(
      k, epsilon_id);
  PopulateWithEpsilons<
      TopK<Hyp, HigherScore, ExtractGlobalScore, InsertHypWithEpsilonDedupe>>(
      &top_k);
  // Check contents: The two items in the TopK are not duplicates.  The dupe
  // has been merged into one of them.
  const auto& new_hyps = top_k.Get();
  EXPECT_THAT(new_hyps, SizeIs(2));
  EXPECT_THAT(new_hyps[0].hyp_id, Eq(1));
  // Merging has happened so combined global score is:
  //   log(exp(-1.5) + exp(-1.3))
  EXPECT_THAT(new_hyps[0].global_score, FloatNear(-0.70186, 0.001));
  EXPECT_THAT(new_hyps[1].hyp_id, Eq(6));
  EXPECT_THAT(new_hyps[1].global_score, FloatNear(-1.7, 0.001));
  EXPECT_TRUE(!IsDupe(new_hyps[0], new_hyps[1]));
}

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
