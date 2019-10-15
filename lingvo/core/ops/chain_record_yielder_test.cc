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

#include "lingvo/core/ops/chain_record_yielder.h"

#include <error.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "lingvo/core/ops/input_common.h"
#include "lingvo/core/ops/record_yielder.h"
#include "lingvo/core/ops/yielder_test_helper.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace lingvo {

TEST(RecordYielderTest, ChainBasicTest) {
  const int N = 10;
  const int M = 1000;
  GeneratePlainTextTestData("chain_yielder1", N, M);
  GeneratePlainTextTestData("chain_yielder2", N, M);

  BasicRecordYielder::Options opts1;
  opts1.file_pattern =
      strings::StrCat("text:", io::JoinPath("/tmp",
                                            "chain_yielder1.*"));
  opts1.seed = 301;
  opts1.bufsize = 2000;
  opts1.parallelism = 1;

  BasicRecordYielder::Options opts2;
  opts2.file_pattern =
      strings::StrCat("text:", io::JoinPath("/tmp",
                                            "chain_yielder2.*"));
  opts2.seed = 301;
  opts2.bufsize = 2000;
  opts2.parallelism = 1;
  ChainRecordYielder* yielder = ChainRecordYielder::New({opts1, opts2});

  std::vector<string> vals;
  Record record;
  record.source_id = kDefaultSourceId;
  // Consume first yielder entirely.
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    VLOG(1) << i << " " << record.value;
    vals.emplace_back(string(record.value));
  }
  ASSERT_NEAR(ComputeInputSourceDistribution(vals)["chain_yielder1"],
              1.0, 0.001);
  vals.clear();
  // Consume second yielder entirely.
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    VLOG(1) << i << " " << record.value;
    vals.emplace_back(string(record.value));
  }
  ASSERT_NEAR(ComputeInputSourceDistribution(vals)["chain_yielder2"],
              1.0, 0.001);
  vals.clear();
  // Consume first yielder again.
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    VLOG(1) << i << " " << record.value;
    vals.emplace_back(string(record.value));
  }
  ASSERT_NEAR(ComputeInputSourceDistribution(vals)["chain_yielder1"],
              1.0, 0.001);

  yielder->Close();
}

TEST(RecordYielderDeathTest, ChainNoYielders) {
  ASSERT_DEATH(ChainRecordYielder::New({}),
               "There should be at least one set of options provided");
}

}  // namespace lingvo
}  // namespace tensorflow
