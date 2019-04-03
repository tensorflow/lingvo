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

#include "lingvo/core/ops/weighted_mix_record_yielder.h"
#include <error.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "lingvo/core/ops/input_common.h"
#include "lingvo/core/ops/record_yielder.h"

namespace tensorflow {
namespace lingvo {

void GeneratePlainTextTestData(const string& prefix, int n, int m) {
  for (int i = 0; i < n; ++i) {
    std::unique_ptr<WritableFile> file;
    TF_CHECK_OK(Env::Default()->NewWritableFile(
        io::JoinPath("/tmp", strings::StrCat(prefix, ".", i)),
        &file));
    for (int j = 0; j < m; ++j) {
      TF_CHECK_OK(file->Append(
          strings::Printf("%s:%010d\n", prefix.c_str(), m * i + j)));
    }
  }
}

// This only works for data generated using GeneratePlainTextTestData with
// properly specified prefixes.
std::unordered_map<std::string, float> ComputeInputSourceDistribution(
    const std::vector<string>& vals) {
  std::unordered_map<std::string, float> input_source_distribution;
  for (const string& val : vals) {
    const auto prefix_end = val.find(':');
    if (prefix_end != string::npos) {
      input_source_distribution[val.substr(0, prefix_end)] += 1.0;
    }
  }
  for (auto it = input_source_distribution.begin();
       it != input_source_distribution.end(); ++it) {
    it->second /= vals.size();
  }
  return input_source_distribution;
}

class MockRecordYielder : public RecordYielder {
 public:
  MOCK_METHOD2(Yield, Status(Rope* value, int* source_id));
  MOCK_METHOD0(Close, void());
  MOCK_CONST_METHOD0(current_epoch, int64());
};

TEST(RecordYielderTest, WeightedMixerBasicTest) {
  const int N = 10;
  const int M = 1000;
  GeneratePlainTextTestData("yielder1", N, M);
  GeneratePlainTextTestData("yielder2", N, M);

  BasicRecordYielder::Options opts1;
  opts1.file_pattern =
      strings::StrCat("text:", io::JoinPath("/tmp", "yielder1.*"));
  opts1.seed = 301;
  opts1.bufsize = 2000;
  opts1.parallelism = 1;
  BasicRecordYielder* yielder1 = BasicRecordYielder::New(opts1);

  BasicRecordYielder::Options opts2;
  opts2.file_pattern =
      strings::StrCat("text:", io::JoinPath("/tmp", "yielder2.*"));
  opts2.seed = 301;
  opts2.bufsize = 2000;
  opts2.parallelism = 1;
  BasicRecordYielder* yielder2 = BasicRecordYielder::New(opts2);
  WeightedMixRecordYielder* yielder =
      WeightedMixRecordYielder::New(301, {yielder1, yielder2}, {0.5, 0.5});

  std::vector<string> vals;
  Rope v;
  for (int i = 0; i < 2 * N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&v, nullptr));
    VLOG(1) << i << " " << v;
    vals.emplace_back(string(v));
  }

  auto input_source_distribution = ComputeInputSourceDistribution(vals);
  ASSERT_NEAR(input_source_distribution["yielder1"], 0.5, 0.01);
  ASSERT_NEAR(input_source_distribution["yielder2"], 0.5, 0.01);

  std::sort(vals.begin(), vals.end());
  auto new_end = std::unique(vals.begin(), vals.end());

  // Duplicates should be rare in the epoch thanks to 0.5/0.5 mix and equally
  // sized input sources.
  ASSERT_LT(vals.end() - new_end, vals.size() / 100);

  yielder->Close();
}

TEST(RecordYielderTest, WeightedMixerUnevenMixTest) {
  const int N = 10;
  const int M = 1000;
  GeneratePlainTextTestData("yielder1", N, M);
  GeneratePlainTextTestData("yielder2", N, M);

  BasicRecordYielder::Options opts1;
  opts1.file_pattern =
      strings::StrCat("text:", io::JoinPath("/tmp", "yielder1.*"));
  opts1.seed = 301;
  opts1.bufsize = 2000;
  opts1.parallelism = 1;
  opts1.source_id = 0;
  BasicRecordYielder* yielder1 = BasicRecordYielder::New(opts1);

  BasicRecordYielder::Options opts2;
  opts2.file_pattern =
      strings::StrCat("text:", io::JoinPath("/tmp", "yielder2.*"));
  opts2.seed = 301;
  opts2.bufsize = 2000;
  opts2.parallelism = 1;
  opts2.source_id = 1;
  BasicRecordYielder* yielder2 = BasicRecordYielder::New(opts2);
  WeightedMixRecordYielder* yielder =
      WeightedMixRecordYielder::New(301, {yielder1, yielder2}, {0.3, 0.7});

  std::vector<string> vals;
  std::vector<int> source_ids;
  Rope v;
  int32 yielder_id;
  for (int i = 0; i < 2 * N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&v, &yielder_id));
    VLOG(1) << i << " " << v;
    vals.emplace_back(string(v));
    source_ids.emplace_back(yielder_id);
  }

  auto input_source_distribution = ComputeInputSourceDistribution(vals);
  ASSERT_NEAR(input_source_distribution["yielder1"], 0.3, 0.01);
  ASSERT_NEAR(input_source_distribution["yielder2"], 0.7, 0.01);

  int32 sum_of_elems = std::accumulate(source_ids.begin(), source_ids.end(), 0);
  float ratio = (float)(sum_of_elems) / (float)(source_ids.size());
  ASSERT_NEAR(ratio, 0.7, 0.01);

  // Take couple 1024-sized batches from the vals, they should have roughly the
  // same distribution.
  for (int i = 0; i < 5; ++i) {
    auto batch_input_source_distribution =
        ComputeInputSourceDistribution(std::vector<string>(
            vals.begin() + i * 1024, vals.begin() + (i + 1) * 1024));
    ASSERT_NEAR(input_source_distribution["yielder1"], 0.3, 0.01);
    ASSERT_NEAR(input_source_distribution["yielder2"], 0.7, 0.01);
  }

  yielder->Close();
}

TEST(RecordYielderTest, WeightedMixerUnevenInputSourcesTest) {
  const int N = 10;
  const int M = 1000;
  GeneratePlainTextTestData("yielder1", N, M);
  // Second input source is 4 times larger than the first one.
  GeneratePlainTextTestData("yielder2", 4 * N, M);

  BasicRecordYielder::Options opts1;
  opts1.file_pattern =
      strings::StrCat("text:", io::JoinPath("/tmp", "yielder1.*"));
  opts1.seed = 301;
  opts1.bufsize = 2000;
  opts1.parallelism = 1;
  BasicRecordYielder* yielder1 = BasicRecordYielder::New(opts1);

  BasicRecordYielder::Options opts2;
  opts2.file_pattern =
      strings::StrCat("text:", io::JoinPath("/tmp", "yielder2.*"));
  opts2.seed = 301;
  opts2.bufsize = 2000;
  opts2.parallelism = 1;
  BasicRecordYielder* yielder2 = BasicRecordYielder::New(opts2);
  WeightedMixRecordYielder* yielder =
      WeightedMixRecordYielder::New(301, {yielder1, yielder2}, {0.5, 0.5});

  std::vector<string> vals;
  Rope v;
  // Iterate 8 times the total record count.
  for (int i = 0; i < 8 * 5 * N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&v, nullptr));
    VLOG(1) << i << " " << v;
    vals.emplace_back(string(v));
  }
  auto input_source_distribution = ComputeInputSourceDistribution(vals);
  ASSERT_NEAR(input_source_distribution["yielder1"], 0.5, 0.01);
  ASSERT_NEAR(input_source_distribution["yielder2"], 0.5, 0.01);

  // Each child yielder will be called half the time to maintain 50/50% split.
  // So out of (8 * 5 * N * M) total yields (4 * 5 * N * M) will be yielded by
  // each yielder.
  // Second yielder produces 4 * N * M elements per epoch so it should be at the
  // end of the 5th epoch | start of the 6th epoch.
  EXPECT_TRUE(yielder2->current_epoch() == 5 || yielder2->current_epoch() == 6);
  // First yielder produces 4 times less elements per epoch (N * M) so it
  // should be at the end of the 20th epoch | start of the 21st epoch.
  EXPECT_TRUE(yielder1->current_epoch() == 20 ||
              yielder1->current_epoch() == 21);
  yielder->Close();
}

TEST(RecordYielderTest, RecordYielderRetryLoop) {
  // It's safe to create mock record yielders on stack as they will be
  MockRecordYielder yielder1;
  MockRecordYielder yielder2;
  // Yielder1 always returns OK. Yielder2 returns DEADLINE_EXCEEDED 3 times in a
  // row and then returns OK.
  // Each of them yields max of 5 records and then saturates.
  EXPECT_CALL(yielder1, Yield(testing::_, testing::_))
      .Times(5)
      .WillRepeatedly(testing::Return(Status::OK()));
  EXPECT_CALL(yielder2, Yield(testing::_, testing::_))
      .Times(5)
      .WillRepeatedly(testing::Return(Status::OK()));
  EXPECT_CALL(yielder2, Yield(testing::_, testing::_))
      .Times(3)
      .WillRepeatedly(testing::Return(Status(error::DEADLINE_EXCEEDED, "")))
      .RetiresOnSaturation();

  WeightedMixRecordYielder* yielder =
      WeightedMixRecordYielder::New(304, {&yielder1, &yielder2}, {0.5, 0.5});

  Rope v;
  for (int i = 0; i < 10; ++i) {
    // Thanks to the seed selected every child yielder will be selected exactly
    // 5 times.
    TF_CHECK_OK(yielder->Yield(&v, nullptr));
  }
  EXPECT_CALL(yielder1, Close());
  EXPECT_CALL(yielder2, Close());
  yielder->Close();
}

TEST(RecordYielderDeathTest, WeightedMixerInconsistentYieldersAndWeights) {
  RecordYielder* yielder1 = nullptr;  // won't ever be used.
  RecordYielder* yielder2 = nullptr;  // won't ever be used.
  ASSERT_DEATH(WeightedMixRecordYielder::New(301, {yielder1, yielder2}, {0.5}),
               "2 yielders and 1 weights were provided");
}

TEST(RecordYielderDeathTest, WeightedMixerNoYielders) {
  ASSERT_DEATH(WeightedMixRecordYielder::New(301, {}, {}),
               "There should be at least one yielder provided");
}

TEST(RecordYielderDeathTest, WeightedMixerNegativeWeights) {
  RecordYielder* yielder1 = nullptr;  // won't ever be used.
  RecordYielder* yielder2 = nullptr;  // won't ever be used.
  ASSERT_DEATH(
      WeightedMixRecordYielder::New(301, {yielder1, yielder2}, {0.3, -0.1}),
      "All weights should be greater or equal to zero. Got -0.1");
}

}  // namespace lingvo
}  // namespace tensorflow
