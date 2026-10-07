/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "lingvo/core/ops/simple_vocab.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace tensorflow {
namespace lingvo {
namespace {

TEST(SimpleVocabTest, GetBowTokenIdsHandlesSparseIds) {
  Vocab vocab;
  ASSERT_TRUE(vocab
                  .Load({"<S>\t3", "</S>\t5", "<UNK>\t7", "plain\t-1",
                         std::string(kBowStr) + "hello\t100"},
                        true)
                  .ok());

  const std::vector<bool> bow_token_ids = vocab.GetBowTokenIds();

  ASSERT_EQ(101, bow_token_ids.size());
  EXPECT_TRUE(bow_token_ids[100]);
  EXPECT_FALSE(bow_token_ids[3]);
  EXPECT_FALSE(bow_token_ids[7]);
}

TEST(SimpleVocabTest, LoadRejectsNegativeBowTokenIds) {
  Vocab vocab;
  EXPECT_FALSE(vocab
                   .Load({"<S>\t3", "</S>\t5", "<UNK>\t7",
                          std::string(kBowStr) + "hello\t-1"},
                         true)
                   .ok());
}

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
