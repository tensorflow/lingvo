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

#include "lingvo/core/ops/text_packing.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tensorflow {
namespace lingvo {
namespace {

//                column 0                 column1
// batch 0 [ a a a - b b b b b - ]  [ A A A A B B B - - -]
// batch 1 [ c c c c c - d - e - ]  [ C C C - D - E - - -]
TEST(TextPackingTest, TestPacking) {
  TextPacking pack(2, 2, 10, 2, true);
  std::vector<TextPacking::PackingIndex> p(5);
  EXPECT_TRUE(pack.Add({3, 4}, &p[0]));  // (a a a, A A A A)
  EXPECT_TRUE(pack.Add({5, 3}, &p[1]));  // (b b b b b, B B B)
  EXPECT_TRUE(pack.Add({5, 3}, &p[2]));  // (c c c c c, C C C)
  EXPECT_TRUE(pack.Add({1, 1}, &p[3]));  // (d, D)
  EXPECT_TRUE(pack.Add({1, 1}, &p[4]));  // (e, E)
  EXPECT_EQ(p[0].batch, 0);
  EXPECT_EQ(p[0].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[0].seq, 1);
  EXPECT_EQ(p[1].batch, 0);
  EXPECT_EQ(p[1].time, std::vector<int>({4, 4}));
  EXPECT_EQ(p[1].seq, 2);
  EXPECT_EQ(p[2].batch, 1);
  EXPECT_EQ(p[2].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[2].seq, 1);
  EXPECT_EQ(p[3].batch, 1);
  EXPECT_EQ(p[3].time, std::vector<int>({6, 4}));
  EXPECT_EQ(p[3].seq, 2);
  EXPECT_EQ(p[4].batch, 1);
  EXPECT_EQ(p[4].time, std::vector<int>({8, 6}));
  EXPECT_EQ(p[4].seq, 3);
}

TEST(TextPackingTest, TestReset) {
  TextPacking pack(2, 2, 10, 2, true);
  std::vector<TextPacking::PackingIndex> p(5);
  EXPECT_TRUE(pack.Add({3, 4}, &p[0]));  // (a a a, A A A A)
  EXPECT_TRUE(pack.Add({5, 3}, &p[1]));  // (b b b b b, B B B)
  EXPECT_TRUE(pack.Add({5, 3}, &p[2]));  // (c c c c c, C C C)
  pack.Reset();
  EXPECT_TRUE(pack.Add({1, 1}, &p[3]));  // (d, D)
  EXPECT_TRUE(pack.Add({1, 1}, &p[4]));  // (e, E)
  EXPECT_EQ(p[0].batch, 0);
  EXPECT_EQ(p[0].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[0].seq, 1);
  EXPECT_EQ(p[1].batch, 0);
  EXPECT_EQ(p[1].time, std::vector<int>({4, 4}));
  EXPECT_EQ(p[1].seq, 2);
  EXPECT_EQ(p[2].batch, 1);
  EXPECT_EQ(p[2].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[2].seq, 1);
  // after reset
  EXPECT_EQ(p[3].batch, 0);
  EXPECT_EQ(p[3].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[3].seq, 1);
  EXPECT_EQ(p[4].batch, 0);
  EXPECT_EQ(p[4].time, std::vector<int>({2, 2}));
  EXPECT_EQ(p[4].seq, 2);
}
TEST(TextPackingTest, TestDoesNotFit) {
  TextPacking pack(2, 2, 10, 2, true);
  TextPacking::PackingIndex p;

  EXPECT_FALSE(pack.Add({11, 1}, &p));
  EXPECT_FALSE(pack.Add({1, 11}, &p));

  EXPECT_TRUE(pack.Add({3, 4}, &p));

  EXPECT_TRUE(pack.Add({5, 3}, &p));

  EXPECT_TRUE(pack.Add({5, 3}, &p));
  EXPECT_FALSE(pack.Add({5, 1}, &p));
  EXPECT_FALSE(pack.Add({1, 7}, &p));

  EXPECT_TRUE(pack.Add({1, 1}, &p));
  EXPECT_FALSE(pack.Add({3, 1}, &p));
  EXPECT_FALSE(pack.Add({1, 5}, &p));

  EXPECT_TRUE(pack.Add({1, 1}, &p));
  EXPECT_FALSE(pack.Add({3, 1}, &p));

  EXPECT_FALSE(pack.Add({1, 1}, &p));
}

//                column 0                 column1
// batch 0 [ a a a b b b b b d e ]  [ A A A A B B B D E -]
// batch 1 [ c c c c c - - - - - ]  [ C C C - - - - - - -]
TEST(TextPackingTest, TestDoNotAlign) {
  TextPacking pack(2, 2, 10, 1, true);
  std::vector<TextPacking::PackingIndex> p(5);
  EXPECT_TRUE(pack.Add({3, 4}, &p[0]));  // (a a a, A A A A)
  EXPECT_TRUE(pack.Add({5, 3}, &p[1]));  // (b b b b b, B B B)
  EXPECT_TRUE(pack.Add({5, 3}, &p[2]));  // (c c c c c, C C C)
  EXPECT_TRUE(pack.Add({1, 1}, &p[3]));  // (d, D)
  EXPECT_TRUE(pack.Add({1, 1}, &p[4]));  // (e, E)
  EXPECT_EQ(p[0].batch, 0);
  EXPECT_EQ(p[0].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[0].seq, 1);
  EXPECT_EQ(p[1].batch, 0);
  EXPECT_EQ(p[1].time, std::vector<int>({3, 4}));
  EXPECT_EQ(p[1].seq, 2);
  EXPECT_EQ(p[2].batch, 1);
  EXPECT_EQ(p[2].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[2].seq, 1);
  EXPECT_EQ(p[3].batch, 0);
  EXPECT_EQ(p[3].time, std::vector<int>({8, 7}));
  EXPECT_EQ(p[3].seq, 3);
  EXPECT_EQ(p[4].batch, 0);
  EXPECT_EQ(p[4].time, std::vector<int>({9, 8}));
  EXPECT_EQ(p[4].seq, 4);
}

TEST(TextPackingTest, TestDoNotPack) {
  TextPacking pack(2, 2, 10, 0, false);
  std::vector<TextPacking::PackingIndex> p(5);
  EXPECT_TRUE(pack.Add({3, 4}, &p[0]));   // (a a a, A A A A)
  EXPECT_TRUE(pack.Add({5, 3}, &p[1]));   // (b b b b b, B B B)
  EXPECT_FALSE(pack.Add({5, 3}, &p[2]));  // (c c c c c, C C C)
  EXPECT_FALSE(pack.Add({1, 1}, &p[3]));  // (d, D)
  EXPECT_FALSE(pack.Add({1, 1}, &p[4]));  // (e, E)
  EXPECT_EQ(p[0].batch, 0);
  EXPECT_EQ(p[0].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[0].seq, 1);
  EXPECT_EQ(p[1].batch, 1);
  EXPECT_EQ(p[1].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[1].seq, 1);
}

//          column 0        column1
// batch 0 [ a a a - ]  [ A A A A - - ]
// batch 1 [ b b c - ]  [ B C C C C C ]
TEST(TextPackingTest, TestDifferentTimesPerColumn) {
  TextPacking pack(2, 2, {4, 6}, 1, true, 0);
  std::vector<TextPacking::PackingIndex> p(5);
  EXPECT_TRUE(pack.Add({3, 4}, &p[0]));  // (a a a, A A A A)
  EXPECT_TRUE(pack.Add({2, 1}, &p[1]));  // (b b, B)
  EXPECT_TRUE(pack.Add({1, 5}, &p[2]));  // (c, C C C C C)
  EXPECT_EQ(p[0].batch, 0);
  EXPECT_EQ(p[0].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[0].seq, 1);
  EXPECT_EQ(p[1].batch, 1);
  EXPECT_EQ(p[1].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[1].seq, 1);
  EXPECT_EQ(p[2].batch, 1);
  EXPECT_EQ(p[2].time, std::vector<int>({2, 1}));
  EXPECT_EQ(p[2].seq, 2);
  EXPECT_FALSE(pack.Add({2, 1}, &p[3]));
  EXPECT_FALSE(pack.Add({1, 3}, &p[4]));
}

//          column 0        column1
// batch 0 [ a a d - ]  [ A A A A D - ]
// batch 1 [ b b - - ]  [ B - - - - - ]
// batch 2 [ c - - - ]  [ C C C C - - ]
TEST(TextPackingTest, TestSpreadFirstN) {
  // First 3 sequences will be assigned different rows.
  TextPacking pack(2, 3, {4, 6}, 1, true, 3);
  std::vector<TextPacking::PackingIndex> p(4);
  TextPacking::PackingIndex unused;
  EXPECT_TRUE(pack.Add({2, 4}, &p[0]));  // (a a a, A A A A)
  EXPECT_FALSE(pack.Add({5, 1}, &unused));
  EXPECT_TRUE(pack.Add({2, 1}, &p[1]));  // (b b, B)
  EXPECT_FALSE(pack.Add({1, 7}, &unused));
  EXPECT_TRUE(pack.Add({1, 4}, &p[2]));  // (c, C C C C C)
  EXPECT_TRUE(pack.Add({1, 1}, &p[3]));  // (d, D)
  EXPECT_EQ(p[0].batch, 0);
  EXPECT_EQ(p[0].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[0].seq, 1);
  EXPECT_EQ(p[1].batch, 1);
  EXPECT_EQ(p[1].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[1].seq, 1);
  EXPECT_EQ(p[2].batch, 2);
  EXPECT_EQ(p[2].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[2].seq, 1);
  EXPECT_EQ(p[3].batch, 0);
  EXPECT_EQ(p[3].time, std::vector<int>({2, 4}));
  EXPECT_EQ(p[3].seq, 2);
}

//          column 0        column1
// batch 0 [ a a d - ]  [ A A A A D - ]
// batch 1 [ b b c - ]  [ B C C C C - ]
// batch 2 [ - - - - ]  [ - - - - - - ]
TEST(TextPackingTest, TestSpreadFirstNSmall) {
  // First 2 sequences will be assigned different rows.
  TextPacking pack(2, 3, {4, 6}, 1, true, 2);
  std::vector<TextPacking::PackingIndex> p(4);
  EXPECT_TRUE(pack.Add({2, 4}, &p[0]));  // (a a, A A A A)
  EXPECT_TRUE(pack.Add({2, 1}, &p[1]));  // (b b, B)
  EXPECT_TRUE(pack.Add({1, 4}, &p[2]));  // (c, C C C C C)
  EXPECT_TRUE(pack.Add({1, 1}, &p[3]));  // (d, D)
  EXPECT_EQ(p[0].batch, 0);
  EXPECT_EQ(p[0].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[0].seq, 1);
  EXPECT_EQ(p[1].batch, 1);
  EXPECT_EQ(p[1].time, std::vector<int>({0, 0}));
  EXPECT_EQ(p[1].seq, 1);
  EXPECT_EQ(p[2].batch, 1);
  EXPECT_EQ(p[2].time, std::vector<int>({2, 1}));
  EXPECT_EQ(p[2].seq, 2);
  EXPECT_EQ(p[3].batch, 0);
  EXPECT_EQ(p[3].time, std::vector<int>({2, 4}));
  EXPECT_EQ(p[3].seq, 2);
}

// With use_last_fit set:
//          column 0        column1
// batch 0 [ a d d d ]  [ A A A A A D ]
// batch 1 [ b b c - ]  [ B B C - - - ]
//
// With use_last_fit unset:
//          column 0        column1
// batch 0 [ a c - - ]  [ A A A A A C ]
// batch 1 [ b b - - ]  [ B B - - - - ]
TEST(TextPackingTest, TestUseLastFit) {
  {
    TextPacking pack(2, 2, {4, 6}, 1, true, 0, /*use_last_fit=*/true);
    std::vector<TextPacking::PackingIndex> p(5);
    EXPECT_TRUE(pack.Add({1, 5}, &p[0]));  // (a, A A A A A)
    EXPECT_TRUE(pack.Add({2, 2}, &p[1]));  // (b b, B B)
    // starts searching in batch 1.
    EXPECT_TRUE(pack.Add({1, 1}, &p[2]));  // (c, C)
    // wraps around to search in batch 0.
    EXPECT_TRUE(pack.Add({3, 1}, &p[3]));   // (d d d, D)
    EXPECT_FALSE(pack.Add({3, 1}, &p[4]));  // (e e e, E)
    EXPECT_EQ(p[0].batch, 0);
    EXPECT_EQ(p[0].time, std::vector<int>({0, 0}));
    EXPECT_EQ(p[0].seq, 1);
    EXPECT_EQ(p[1].batch, 1);
    EXPECT_EQ(p[1].time, std::vector<int>({0, 0}));
    EXPECT_EQ(p[1].seq, 1);
    EXPECT_EQ(p[2].batch, 1);
    EXPECT_EQ(p[2].time, std::vector<int>({2, 2}));
    EXPECT_EQ(p[2].seq, 2);
    EXPECT_EQ(p[3].batch, 0);
    EXPECT_EQ(p[3].time, std::vector<int>({1, 5}));
    EXPECT_EQ(p[3].seq, 2);
  }
  {
    TextPacking pack(2, 2, {4, 6}, 1, true, 0, /*use_last_fit=*/false);
    std::vector<TextPacking::PackingIndex> p(4);
    EXPECT_TRUE(pack.Add({1, 5}, &p[0]));  // (a, A A A A A)
    EXPECT_TRUE(pack.Add({2, 2}, &p[1]));  // (b b, B B)
    // starts search in batch 0.
    EXPECT_TRUE(pack.Add({1, 1}, &p[2]));   // (c, C)
    EXPECT_FALSE(pack.Add({3, 1}, &p[3]));  // (d d d, D)
    EXPECT_EQ(p[0].batch, 0);
    EXPECT_EQ(p[0].time, std::vector<int>({0, 0}));
    EXPECT_EQ(p[0].seq, 1);
    EXPECT_EQ(p[1].batch, 1);
    EXPECT_EQ(p[1].time, std::vector<int>({0, 0}));
    EXPECT_EQ(p[1].seq, 1);
    EXPECT_EQ(p[2].batch, 0);
    EXPECT_EQ(p[2].time, std::vector<int>({1, 5}));
    EXPECT_EQ(p[2].seq, 2);
  }
}

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
