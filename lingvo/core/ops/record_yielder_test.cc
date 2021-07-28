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

#include <chrono>  // NOLINT(build/c++11)

#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "lingvo/core/ops/input_common.h"
#include "lingvo/core/ops/sequential_record_yielder.h"
#include "lingvo/core/ops/yielder_test_helper.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace lingvo {

TEST(RecordYielderTest, PlainTextYielderBasicTest) {
  const int N = 10;
  const int M = 1000;
  GeneratePlainTextTestData("basic", N, M);
  BasicRecordYielder::Options opts;
  opts.file_pattern = strings::StrCat(
      "text:", io::JoinPath("/tmp", "basic.*"));
  opts.seed = 301;
  opts.bufsize = 2000;
  opts.parallelism = 1;

  BasicRecordYielder* yielder = BasicRecordYielder::New(opts);
  std::vector<string> vals;
  Record record;
  record.source_id = kDefaultSourceId;
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    VLOG(1) << i << " " << record.value;
    vals.emplace_back(string(record.value));
  }
  std::sort(vals.begin(), vals.end());
  auto new_end = std::unique(vals.begin(), vals.end());

  // One epoch should have no duplicates.
  ASSERT_EQ(new_end, vals.end());

  // Iterates another two epochs.
  for (int i = 0; i < 2 * N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
  }

  // Iterates another 34 epochs.
  for (int i = 0; i < 2 * 17 * N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
  }

  // End of the 37th epoch | start of the 38th epoch.
  EXPECT_TRUE(yielder->current_epoch() == 37 || yielder->current_epoch() == 38);
  yielder->Close();
}

TEST(SequentialRecordYielderTest, SequentialRecordYielderBasicTest) {
  const int N = 10;
  const int M = 1000;
  GeneratePlainTextTestData("basic", N, M);
  const string& file_pattern = strings::StrCat(
      "text:", io::JoinPath("/tmp", "basic.*"));

  SequentialRecordYielder* yielder =
      SequentialRecordYielder::New(file_pattern, -1);
  Record record;
  record.source_id = kDefaultSourceId;
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    ASSERT_EQ(string(record.value), strings::Printf("basic:%010d", i));
  }

  // Iterate another epoch.
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    ASSERT_EQ(string(record.value), strings::Printf("basic:%010d", i));
  }

  yielder->Close();
}

TEST(SequentialRecordYielderTest, SequentialRecordYielderRepeatCount) {
  const int N = 10;
  const int M = 1000;
  GeneratePlainTextTestData("basic", N, M);
  const string& file_pattern = strings::StrCat(
      "text:", io::JoinPath("/tmp", "basic.*"));

  // Yield two epochs.
  SequentialRecordYielder* yielder =
      SequentialRecordYielder::New(file_pattern, 2);
  Record record;
  record.source_id = kDefaultSourceId;
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    ASSERT_EQ(string(record.value), strings::Printf("basic:%010d", i));
  }

  // Iterate another epoch.
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    ASSERT_EQ(string(record.value), strings::Printf("basic:%010d", i));
  }

  // Trying to yield one more element should throw an out of range error.
  Status s = yielder->Yield(&record);
  ASSERT_TRUE(errors::IsOutOfRange(s)) << s;

  yielder->Close();
}

void GenerateTfRecordTestData(const string& prefix, int n, int m,
                              const string& compression_type) {
  for (int i = 0; i < n; ++i) {
    std::unique_ptr<WritableFile> file;
    TF_CHECK_OK(Env::Default()->NewWritableFile(
        io::JoinPath("/tmp",
                     strings::StrCat(prefix, ".", i)),
        &file));
    io::RecordWriter writer(
        file.get(),
        io::RecordWriterOptions::CreateRecordWriterOptions(compression_type));
    for (int j = 0; j < m; ++j) {
      TF_CHECK_OK(writer.WriteRecord(strings::Printf("%010d", m * i + j)));
    }
  }
}

void GenerateShardedTfRecordTestData(const string& prefix, const string& suffix,
                                     int n, int m) {
  for (int i = 0; i < n; ++i) {
    std::unique_ptr<WritableFile> file;
    string filename = strings::Printf("%s-%05d-of-%05d", prefix.c_str(), i, n);
    if (!suffix.empty()) {
      strings::Appendf(&filename, "%s", suffix.c_str());
    }
    TF_CHECK_OK(Env::Default()->NewWritableFile(
        io::JoinPath("/tmp", filename), &file));
    io::RecordWriter writer(file.get());
    for (int j = 0; j < m; ++j) {
      TF_CHECK_OK(writer.WriteRecord(strings::Printf("%010d", m * i + j)));
    }
  }
}

typedef testing::TestWithParam<string> TfRecordYielderTest;

string PrefixFromCompressionType(const string& compression_type) {
  if (compression_type == io::compression::kGzip) {
    return "tfrecord_gzip:";
  } else if (compression_type != io::compression::kNone) {
    LOG(ERROR) << "Unknown compression type, using no compression";
  }
  return "tfrecord:";
}

TEST_P(TfRecordYielderTest, TfRecordYielderBasicTest) {
  const int N = 10;
  const int M = 1000;
  GenerateTfRecordTestData("basic", N, M, GetParam());
  BasicRecordYielder::Options opts;
  opts.file_pattern = strings::StrCat(
      PrefixFromCompressionType(GetParam()),
      io::JoinPath("/tmp", "basic.*"));
  opts.seed = 301;
  opts.bufsize = 2000;
  opts.parallelism = 1;

  BasicRecordYielder* yielder = BasicRecordYielder::New(opts);
  std::vector<string> vals;
  Record record;
  record.source_id = kDefaultSourceId;
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    VLOG(1) << i << " " << record.value;
    vals.emplace_back(string(record.value));
  }
  std::sort(vals.begin(), vals.end());
  auto new_end = std::unique(vals.begin(), vals.end());

  // One epoch should have no duplicates.
  ASSERT_EQ(new_end, vals.end());

  // Iterates another two epochs.
  for (int i = 0; i < 2 * N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
  }

  // Iterates another 34 epochs.
  for (int i = 0; i < 2 * 17 * N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
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

TEST_P(TfRecordYielderTest, ShufflesShard) {
  const int M = 32;
  GenerateTfRecordTestData("oneshard", 1 /* num_shards */, M, GetParam());

  BasicRecordYielder::Options opts;
  opts.file_pattern = strings::StrCat(
      PrefixFromCompressionType(GetParam()),
      io::JoinPath("/tmp", "oneshard.0"));
  opts.bufsize = M;
  opts.parallelism = 1;

  // Subsequent epochs should be yielded in different orders.
  std::vector<Rope> epoch1, epoch2;
  {
    opts.seed = 301;
    auto yielder = BasicRecordYielder::New(opts);
    for (int i = 0; i < M; ++i) {
      Record record;
      record.source_id = kDefaultSourceId;
      TF_CHECK_OK(yielder->Yield(&record));
      epoch1.push_back(record.value);
    }
    for (int i = 0; i < M; ++i) {
      Record record;
      record.source_id = kDefaultSourceId;
      TF_CHECK_OK(yielder->Yield(&record));
      epoch2.push_back(record.value);
    }
    yielder->Close();
    EXPECT_LT(NumMatches(epoch1, epoch2), M);
  }

  // Ordering should change if the seed changes.
  std::vector<Rope> epoch1_different_seed;
  {
    opts.seed = 103;
    auto yielder = BasicRecordYielder::New(opts);
    for (int i = 0; i < M; ++i) {
      Record record;
      record.source_id = kDefaultSourceId;
      TF_CHECK_OK(yielder->Yield(&record));
      epoch1_different_seed.push_back(record.value);
    }
    yielder->Close();
    EXPECT_LT(NumMatches(epoch1, epoch1_different_seed), M);
  }
}

TEST(RecordYielderDeathTest, Error) {
  BasicRecordYielder::Options opts;
  opts.file_pattern = strings::StrCat(
      "tfrecord:", io::JoinPath("/tmp", "nothing.*"));
  auto yielder = BasicRecordYielder::New(opts);
  Record record;
  record.source_id = kDefaultSourceId;
  auto status = yielder->Yield(&record);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr("Found no files at"));
  yielder->Close();
}

TEST_P(TfRecordYielderTest, MatchFilesFromMultiplePatterns) {
  const int N = 2;
  const int M = 32;
  GenerateTfRecordTestData("twoshard", N /* num_shards */,
                           M /* record per shard */, GetParam());
  BasicRecordYielder::Options opts;
  const string path0 =
      io::JoinPath("/tmp", "twoshard.0");
  const string path1 =
      io::JoinPath("/tmp", "twoshard.1");
  opts.file_pattern =
      strings::StrCat(PrefixFromCompressionType(GetParam()), path0, ",", path1);
  opts.bufsize = M;
  opts.parallelism = 1;
  std::vector<Rope> epoch;
  auto yielder = BasicRecordYielder::New(opts);
  Record record;
  record.source_id = kDefaultSourceId;
  for (int i = 0; i < N * M; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    epoch.emplace_back(string(record.value));
  }
  auto new_end = std::unique(epoch.begin(), epoch.end());
  // If we iterated through both shards (rather than 1 shard twice), there
  // should be no duplicates, and we should be at the end of the first epoch.
  EXPECT_EQ(new_end, epoch.end());
  // End of the 1st epoch | start of the 2nd epoch.
  EXPECT_TRUE(yielder->current_epoch() == 1 || yielder->current_epoch() == 2);
  TF_CHECK_OK(yielder->Yield(&record));
  // End of the 2st epoch | start of the 3nd epoch.
  EXPECT_TRUE(yielder->current_epoch() == 2 || yielder->current_epoch() == 3);
  yielder->Close();
}

INSTANTIATE_TEST_CASE_P(All, TfRecordYielderTest,
                        testing::Values(io::compression::kNone,
                                        io::compression::kGzip));

TEST(RecordYielder, MatchShardedFilePattern) {
  const int num_shards = 16;
  const int records_per_shard = 8;
  GenerateShardedTfRecordTestData("sharded_data", "", num_shards,
                                  records_per_shard);

  BasicRecordYielder::Options opts;
  opts.file_pattern = strings::StrCat(
      "tfrecord:",
      io::JoinPath("/tmp", "sharded_data@16"));
  opts.bufsize = records_per_shard;
  opts.parallelism = 1;
  std::vector<Rope> epoch;
  auto yielder = BasicRecordYielder::New(opts);
  Record record;
  record.source_id = kDefaultSourceId;
  for (int i = 0; i < num_shards * records_per_shard; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    epoch.emplace_back(string(record.value));
  }
  auto new_end = std::unique(epoch.begin(), epoch.end());
  // If we iterated through all shards (rather than 1 shard twice), there
  // should be no duplicates, and we should be at the end of the first epoch.
  EXPECT_EQ(new_end, epoch.end());
  // End of the 1st epoch | start of the 2nd epoch.
  EXPECT_TRUE(yielder->current_epoch() == 1 || yielder->current_epoch() == 2);
  TF_CHECK_OK(yielder->Yield(&record));
  // End of the 2st epoch | start of the 3nd epoch.
  EXPECT_TRUE(yielder->current_epoch() == 2 || yielder->current_epoch() == 3);
  yielder->Close();
}

TEST(RecordYielder, MatchWildcardShardedFilePattern) {
  const int num_shards = 9;
  const int records_per_shard = 8;
  GenerateShardedTfRecordTestData("sharded_data2", "", num_shards,
                                  records_per_shard);

  BasicRecordYielder::Options opts;
  opts.file_pattern = strings::StrCat(
      "tfrecord:",
      io::JoinPath("/tmp", "sharded_data2@*"));
  opts.bufsize = records_per_shard;
  opts.parallelism = 1;
  std::vector<Rope> epoch;
  auto yielder = BasicRecordYielder::New(opts);
  Record record;
  record.source_id = kDefaultSourceId;

  for (int i = 0; i < num_shards * records_per_shard; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    epoch.emplace_back(string(record.value));
  }
  auto new_end = std::unique(epoch.begin(), epoch.end());
  // If we iterated through all shards (rather than 1 shard twice), there
  // should be no duplicates, and we should be at the end of the first epoch.
  EXPECT_EQ(new_end, epoch.end());
  // End of the 1st epoch | start of the 2nd epoch.
  EXPECT_TRUE(yielder->current_epoch() == 1 || yielder->current_epoch() == 2);
  TF_CHECK_OK(yielder->Yield(&record));
  // End of the 2st epoch | start of the 3nd epoch.
  EXPECT_TRUE(yielder->current_epoch() == 2 || yielder->current_epoch() == 3);
  yielder->Close();
}

TEST(RecordYielder, MatchShardedFilePatternWithSuffix) {
  const int num_shards = 9;
  const int records_per_shard = 8;
  GenerateShardedTfRecordTestData("sharded_data3", ".record", num_shards,
                                  records_per_shard);

  BasicRecordYielder::Options opts;
  opts.file_pattern = strings::StrCat(
      "tfrecord:",
      io::JoinPath("/tmp", "sharded_data3@9.record"));
  opts.bufsize = records_per_shard;
  opts.parallelism = 1;
  std::vector<Rope> epoch;
  auto yielder = BasicRecordYielder::New(opts);
  Record record;
  record.source_id = kDefaultSourceId;

  for (int i = 0; i < num_shards * records_per_shard; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    epoch.emplace_back(string(record.value));
  }
  auto new_end = std::unique(epoch.begin(), epoch.end());
  // If we iterated through all shards (rather than 1 shard twice), there
  // should be no duplicates, and we should be at the end of the first epoch.
  EXPECT_EQ(new_end, epoch.end());
  // End of the 1st epoch | start of the 2nd epoch.
  EXPECT_TRUE(yielder->current_epoch() == 1 || yielder->current_epoch() == 2);
  TF_CHECK_OK(yielder->Yield(&record));
  // End of the 2st epoch | start of the 3nd epoch.
  EXPECT_TRUE(yielder->current_epoch() == 2 || yielder->current_epoch() == 3);
  yielder->Close();
}

TEST(RecordYielder, MatchIndirectFilePattern) {
  const int records_per_shard = 100;
  GenerateCheckpointPlainTextTestData("checkpoint", records_per_shard);

  BasicRecordYielder::Options opts;
  opts.file_pattern = strings::StrCat(
      "text_indirect:",
      io::JoinPath("/tmp", "checkpoint"));
  opts.bufsize = records_per_shard;
  opts.parallelism = 1;
  std::vector<Rope> epoch;
  auto yielder = BasicRecordYielder::New(opts);
  Record record;
  record.source_id = kDefaultSourceId;

  // Iterate over all but 1 record in the entire data file.
  for (int i = 0; i < records_per_shard - 1; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    epoch.emplace_back(string(record.value));
  }
  // Update checkpoint file to point to new data and iterate over final file.
  UpdateCheckpointPlainTextTestData("checkpoint", records_per_shard);
  TF_CHECK_OK(yielder->Yield(&record));
  epoch.emplace_back(string(record.value));
  auto new_end = std::unique(epoch.begin(), epoch.end());
  // If we iterated through only the first version of the file, there
  // should be no duplicates, and we should be at the end of the first epoch.
  EXPECT_EQ(new_end, epoch.end());
  // End of the 1st epoch | start of the 2nd epoch.
  EXPECT_TRUE(yielder->current_epoch() == 1 || yielder->current_epoch() == 2);

  // Now we should be iterating over the new file.
  TF_CHECK_OK(yielder->Yield(&record));
  epoch.emplace_back(string(record.value));
  new_end = std::unique(epoch.begin(), epoch.end());
  EXPECT_EQ(new_end, epoch.end());
  yielder->Close();
}

namespace {

class FakeIterator : public RecordIterator {
 public:
  FakeIterator(const string& pattern) : RecordIterator(), pattern_(pattern) {}
  bool Next(string* key, Rope* value) {
    if (pattern_.empty()) return false;
    *key = pattern_;
    *value = pattern_;
    pattern_ = "";
    return true;
  }

 private:
  std::string pattern_;
};

bool register_fake_iterator = RecordIterator::RegisterWithPatternParser(
    "fakeiter", [](const string& pattern) { return new FakeIterator(pattern); },
    [](const string& file_pattern, const RecordIterator::ParserOptions& options,
       std::vector<std::string>* shards) {
      shards->push_back(file_pattern);
      return Status::OK();
    });

}  // namespace

TEST(RecordYielder, RegisterFakeIterator) {
  ASSERT_TRUE(register_fake_iterator);
  BasicRecordYielder::Options options;
  options.file_pattern = "fakeiter:hello1";
  BasicRecordYielder* yielder = BasicRecordYielder::New(options);
  EXPECT_TRUE(yielder != nullptr);
  Record record;
  record.source_id = kDefaultSourceId;
  EXPECT_TRUE(yielder->Yield(&record).ok());
  EXPECT_EQ("hello1", record.value);
  EXPECT_TRUE(yielder->Yield(&record).ok());
  EXPECT_EQ("hello1", record.value);
  yielder->Close();
}

TEST(RecordYielder, Iota) {
  BasicRecordYielder::Options opts;
  opts.file_pattern = "iota:100";
  opts.bufsize = 16;
  opts.parallelism = 1;
  BasicRecordYielder* yielder = BasicRecordYielder::New(opts);
  std::vector<string> vals;
  Record record;
  record.source_id = kDefaultSourceId;
  for (int i = 0; i < 100; ++i) {
    TF_CHECK_OK(yielder->Yield(&record));
    VLOG(1) << i << " " << record.value;
    vals.emplace_back(string(record.value));
  }
  std::sort(vals.begin(), vals.end());
  auto new_end = std::unique(vals.begin(), vals.end());
  EXPECT_EQ(new_end, vals.end());
  yielder->Close();
}

TEST(RecordYielder, AdjustStaysLow) {
  BasicRecordYielder::Options opts;
  opts.file_pattern = "iota:100";
  opts.bufsize = 100;
  opts.bufsize_in_seconds = 1;
  opts.parallelism = 1;
  BasicRecordYielder* yielder = BasicRecordYielder::New(opts);

  // We didn't read anything for 5 seconds, so we expect that the buffer size
  // has stayed at the initial value of 16.
  std::this_thread::sleep_for(std::chrono::seconds(5));
  EXPECT_EQ(16, yielder->bufsize());
  yielder->Close();
}

TEST(RecordYielder, AdjustUp) {
  BasicRecordYielder::Options opts;
  opts.file_pattern = "iota:100";
  opts.bufsize = 100;
  opts.bufsize_in_seconds = 1;
  opts.parallelism = 1;
  BasicRecordYielder* yielder = BasicRecordYielder::New(opts);

  // We are reading 1000 records/second, so we expect that the buffer size
  // will be much higher than the initial value of 16.
  for (int i = 0; i < 50; i++) {
    for (int j = 0; j < 100; j++) {
      Record record;
      record.source_id = kDefaultSourceId;
      LOG(INFO) << "yield " << j;
      TF_CHECK_OK(yielder->Yield(&record));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  EXPECT_LT(30, yielder->bufsize());
  EXPECT_GE(100, yielder->bufsize());
  yielder->Close();
}

TEST(RecordIterator, GetFilePatternPrefix) {
  EXPECT_EQ("", RecordIterator::GetFilePatternPrefix("/foo/bar/*"));
  EXPECT_EQ("baz", RecordIterator::GetFilePatternPrefix("baz:/foo/bar/*"));

  string file_pattern;

  file_pattern = "/foo/bar/*";
  EXPECT_EQ("", RecordIterator::StripPrefixFromFilePattern(&file_pattern));
  EXPECT_EQ("/foo/bar/*", file_pattern);

  file_pattern = "baz:/foo/bar/*";
  EXPECT_EQ("baz", RecordIterator::StripPrefixFromFilePattern(&file_pattern));
  EXPECT_EQ("/foo/bar/*", file_pattern);
}

}  // namespace lingvo
}  // namespace tensorflow
