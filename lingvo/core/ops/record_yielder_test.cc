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

#include "lingvo/core/ops/record_yielder.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "lingvo/core/ops/input_common.h"

namespace tensorflow {
namespace lingvo {

void GeneratePlainTextTestData(const string& prefix, int n, int m) {
  for (int i = 0; i < n; ++i) {
    std::unique_ptr<WritableFile> file;
    TF_CHECK_OK(Env::Default()->NewWritableFile(
        io::JoinPath("/tmp", strings::StrCat(prefix, ".", i)),
        &file));
    for (int j = 0; j < m; ++j) {
      TF_CHECK_OK(file->Append(strings::Printf("%010d\n", m * i + j)));
    }
  }
}

TEST(RecordYielderTest, PlainTextYielderBasicTest) {
  const int N = 10;
  const int M = 1000;
  GeneratePlainTextTestData("basic", N, M);
  RecordYielder::Options opts;
  opts.file_pattern =
      strings::StrCat("text:", io::JoinPath("/tmp", "basic.*"));
  opts.seed = 301;
  opts.bufsize = 2000;
  opts.parallelism = 1;

  RecordYielder* yielder = RecordYielder::New(opts);
  std::vector<string> vals;
  Rope v;
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&v));
    VLOG(1) << i << " " << v;
    vals.emplace_back(string(v));
  }
  std::sort(vals.begin(), vals.end());
  auto new_end = std::unique(vals.begin(), vals.end());

  // One epoch should have no duplicates.
  ASSERT_EQ(new_end, vals.end());

  // Iterates another two epochs.
  for (int i = 0; i < 2 * N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&v));
  }

  // Iterates another 34 epochs.
  for (int i = 0; i < 2 * 17 * N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&v));
  }

  // End of the 37th epoch | start of the 38th epoch.
  EXPECT_TRUE(yielder->current_epoch() == 37 || yielder->current_epoch() == 38);
  yielder->Close();
}

void GenerateTfRecordTestData(const string& prefix, int n, int m) {
  for (int i = 0; i < n; ++i) {
    std::unique_ptr<WritableFile> file;
    TF_CHECK_OK(Env::Default()->NewWritableFile(
        io::JoinPath("/tmp", strings::StrCat(prefix, ".", i)),
        &file));
    io::RecordWriter writer(file.get());
    for (int j = 0; j < m; ++j) {
      TF_CHECK_OK(writer.WriteRecord(strings::Printf("%010d", m * i + j)));
    }
  }
}

TEST(RecordYielderTest, TfRecordYielderBasicTest) {
  const int N = 10;
  const int M = 1000;
  GenerateTfRecordTestData("basic", N, M);
  RecordYielder::Options opts;
  opts.file_pattern =
      strings::StrCat("tfrecord:", io::JoinPath("/tmp", "basic.*"));
  opts.seed = 301;
  opts.bufsize = 2000;
  opts.parallelism = 1;

  RecordYielder* yielder = RecordYielder::New(opts);
  std::vector<string> vals;
  Rope v;
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&v));
    VLOG(1) << i << " " << v;
    vals.emplace_back(string(v));
  }
  std::sort(vals.begin(), vals.end());
  auto new_end = std::unique(vals.begin(), vals.end());

  // One epoch should have no duplicates.
  ASSERT_EQ(new_end, vals.end());

  // Iterates another two epochs.
  for (int i = 0; i < 2 * N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&v));
  }

  // Iterates another 34 epochs.
  for (int i = 0; i < 2 * 17 * N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&v));
  }

  // End of the 37th epoch | start of the 38th epoch.
  EXPECT_TRUE(yielder->current_epoch() == 37 || yielder->current_epoch() == 38);
  yielder->Close();
}

int NumMatches(const std::vector<Rope>& vals1,
               const std::vector<Rope>& vals2) {
  CHECK_EQ(vals1.size(), vals2.size());
  int num_matches = 0;
  for (int i = 0; i < vals1.size(); ++i) {
    num_matches += vals1[i] == vals2[i];
  }
  return num_matches;
}

TEST(RecordYielderTest, ShufflesShard) {
  const int M = 32;
  GenerateTfRecordTestData("oneshard", 1 /* num_shards */, M);

  RecordYielder::Options opts;
  opts.file_pattern = strings::StrCat(
      "tfrecord:", io::JoinPath("/tmp", "oneshard.0"));
  opts.bufsize = M;
  opts.parallelism = 1;

  // Subsequent epochs should be yielded in different orders.
  std::vector<Rope> epoch1, epoch2;
  {
    opts.seed = 301;
    auto yielder = RecordYielder::New(opts);
    for (int i = 0; i < M; ++i) {
      Rope v;
      TF_CHECK_OK(yielder->Yield(&v));
      epoch1.push_back(v);
    }
    for (int i = 0; i < M; ++i) {
      Rope v;
      TF_CHECK_OK(yielder->Yield(&v));
      epoch2.push_back(v);
    }
    yielder->Close();
    EXPECT_LT(NumMatches(epoch1, epoch2), M);
  }

  // Ordering should change if the seed changes.
  std::vector<Rope> epoch1_different_seed;
  {
    opts.seed = 103;
    auto yielder = RecordYielder::New(opts);
    for (int i = 0; i < M; ++i) {
      Rope v;
      TF_CHECK_OK(yielder->Yield(&v));
      epoch1_different_seed.push_back(v);
    }
    yielder->Close();
    EXPECT_LT(NumMatches(epoch1, epoch1_different_seed), M);
  }
}

TEST(RecordYielder, Error) {
  RecordYielder::Options opts;
  opts.file_pattern = strings::StrCat(
      "tfrecord:", io::JoinPath("/tmp", "nothing.*"));
  auto yielder = RecordYielder::New(opts);
  Rope v;
  EXPECT_TRUE(errors::IsNotFound(yielder->Yield(&v)));
  yielder->Close();
}

TEST(RecordYielder, MatchFilesFromMultiplePatterns) {
  const int N = 2;
  const int M = 32;
  GenerateTfRecordTestData("twoshard", N /* num_shards */,
                           M /* record per shard */);
  RecordYielder::Options opts;
  const string path0 = io::JoinPath("/tmp", "twoshard.0");
  const string path1 = io::JoinPath("/tmp", "twoshard.1");
  opts.file_pattern = strings::StrCat("tfrecord:", path0, ",", path1);
  opts.bufsize = M;
  opts.parallelism = 1;
  std::vector<Rope> epoch;
  auto yielder = RecordYielder::New(opts);
  Rope v;
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&v));
    epoch.emplace_back(string(v));
  }
  auto new_end = std::unique(epoch.begin(), epoch.end());
  // If we iterated through both shards (rather than 1 shard twice), there
  // should be no duplicates, and we should be at the end of the first epoch.
  EXPECT_EQ(new_end, epoch.end());
  // End of the 1st epoch | start of the 2nd epoch.
  EXPECT_TRUE(yielder->current_epoch() == 1 || yielder->current_epoch() == 2);
  TF_CHECK_OK(yielder->Yield(&v));
  // End of the 2st epoch | start of the 3nd epoch.
  EXPECT_TRUE(yielder->current_epoch() == 2 || yielder->current_epoch() == 3);
  yielder->Close();
}

}  // namespace lingvo
}  // namespace tensorflow
