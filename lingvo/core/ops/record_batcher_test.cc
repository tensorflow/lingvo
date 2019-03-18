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

#include "lingvo/core/ops/record_batcher.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "lingvo/core/ops/input_common.h"

namespace tensorflow {
namespace lingvo {

void GenerateTestData(const string& filename, int n, bool random_value) {
  std::unique_ptr<WritableFile> file;
  TF_CHECK_OK(Env::Default()->NewWritableFile(filename, &file));
  io::RecordWriter writer(file.get());
  for (int i = 0; i < n; ++i) {
    if (random_value) {
      const string val(1 + (i % 100), 'x');  // Length of [1 .. 100]
      TF_CHECK_OK(writer.WriteRecord(val));
    } else {
      const string val = strings::Printf("%010d", i);
      TF_CHECK_OK(writer.WriteRecord(val));
    }
  }
}

class TestRP : public RecordProcessor {
 public:
  TestRP() {}

  ~TestRP() override {}

  Status Process(const Rope& record, int64* bucket_key,
                 TensorVec* sample) override {
    const string val = string(record);
    *bucket_key = val.size();
    Tensor t(DT_STRING, {});
    record.AppendTo(&t.scalar<string>()());
    sample->clear();
    sample->push_back(std::move(t));
    return Status::OK();
  }

  Status Merge(int64 bucket_id, const std::vector<TensorVec>& samples,
               TensorVec* batch) override {
    const int64 n = samples.size();
    Tensor t(DT_STRING, {n});
    for (int i = 0; i < samples.size(); ++i) {
      t.flat<string>()(i) = std::move(samples[i][0].scalar<string>()());
    }
    batch->clear();
    batch->push_back(std::move(t));
    return Status::OK();
  }
};

TEST(RecordBatcher, Basic) {
  const string filename = io::JoinPath("/tmp", "basic");
  GenerateTestData(filename, 1000, true /* random_value */);

  BasicRecordYielder::Options yopts;
  yopts.file_pattern = strings::StrCat("tfrecord:", filename);
  yopts.seed = 301;
  yopts.bufsize = 10;
  yopts.parallelism = 1;

  RecordBatcher::Options bopts;
  bopts.bucket_upper_bound = {20, 50, 90, 95};
  bopts.bucket_batch_limit = {8, 4, 2, 1};

  RecordBatcher batcher(bopts, BasicRecordYielder::New(yopts), new TestRP());
  int64 bucket_id;
  TensorVec batch;
  for (int i = 0; i < 1000; ++i) {
    batcher.GetNext(&bucket_id, &batch);
    ASSERT_LE(0, bucket_id);
    ASSERT_LT(bucket_id, bopts.bucket_upper_bound.size());
    const Tensor& t = batch[0];
    ASSERT_EQ(t.dims(), 1);
    ASSERT_LE(t.dim_size(0), bopts.bucket_batch_limit[bucket_id]);
    int maxlen = 0;
    for (int j = 0; j < t.dim_size(0); ++j) {
      auto len = t.vec<string>()(j).size();
      EXPECT_LE(len, bopts.bucket_upper_bound[bucket_id]);
      if (bucket_id != 0) {
        EXPECT_LT(bopts.bucket_upper_bound[bucket_id - 1], len);
      }
      maxlen = std::max<int>(maxlen, len);
    }
    VLOG(1) << bucket_id << " " << t.dim_size(0) << " " << maxlen;
  }
}

TEST(RecordBatcher, BasicMultiThread) {
  const string filename = io::JoinPath("/tmp", "basic");
  GenerateTestData(filename, 1000, true /* random_value */);

  BasicRecordYielder::Options yopts;
  yopts.file_pattern = strings::StrCat("tfrecord:", filename);
  yopts.seed = 301;
  yopts.bufsize = 10;
  yopts.parallelism = 1;

  RecordBatcher::Options bopts;
  bopts.bucket_upper_bound = {20, 50, 90, 95};
  bopts.bucket_batch_limit = {8, 4, 2, 1};
  bopts.num_threads = 4;

  RecordBatcher batcher(bopts, BasicRecordYielder::New(yopts), new TestRP());
  int64 bucket_id;
  TensorVec batch;
  for (int i = 0; i < 1000; ++i) {
    batcher.GetNext(&bucket_id, &batch);
    ASSERT_LE(0, bucket_id);
    ASSERT_LT(bucket_id, bopts.bucket_upper_bound.size());
    const Tensor& t = batch[0];
    ASSERT_EQ(t.dims(), 1);
    ASSERT_LE(t.dim_size(0), bopts.bucket_batch_limit[bucket_id]);
    int maxlen = 0;
    for (int j = 0; j < t.dim_size(0); ++j) {
      auto len = t.vec<string>()(j).size();
      EXPECT_LE(len, bopts.bucket_upper_bound[bucket_id]);
      if (bucket_id != 0) {
        EXPECT_LT(bopts.bucket_upper_bound[bucket_id - 1], len);
      }
      maxlen = std::max<int>(maxlen, len);
    }
    VLOG(1) << bucket_id << " " << t.dim_size(0) << " " << maxlen;
  }
}

TEST(RecordBatcher, LearnBuckets) {
  const string filename = io::JoinPath("/tmp", "basic");
  GenerateTestData(filename, 1000, true /* random_value */);

  BasicRecordYielder::Options yopts;
  yopts.file_pattern = strings::StrCat("tfrecord:", filename);
  yopts.seed = 301;
  yopts.bufsize = 10;
  yopts.parallelism = 1;

  RecordBatcher::Options bopts;
  bopts.bucket_upper_bound = {100, 100, 100, 100};
  bopts.bucket_batch_limit = {8, 8, 8, 8};
  bopts.bucket_adjust_every_n = 550;

  RecordBatcher batcher(bopts, BasicRecordYielder::New(yopts), new TestRP());
  int64 bucket_id;
  TensorVec batch;

  // For the first 500 batches we just make sure the batches are the right
  // size.
  for (int i = 0; i < 500; ++i) {
    batcher.GetNext(&bucket_id, &batch);
    ASSERT_LE(0, bucket_id);
    const Tensor& t = batch[0];
    ASSERT_EQ(8, t.dim_size(0));
  }

  // For the next 1000 batches we measure the max length distribution.
  std::vector<double> maxlens;
  std::vector<int64> batches;
  maxlens.resize(4, 0);
  batches.resize(4, 0);
  for (int i = 0; i < 1000; ++i) {
    batcher.GetNext(&bucket_id, &batch);
    const Tensor& t = batch[0];
    int maxlen = 0;
    for (int j = 0; j < t.dim_size(0); ++j) {
      int len = t.vec<string>()(j).size();
      maxlen = std::max<int>(maxlen, len);
    }
    maxlens[bucket_id] += maxlen;
    batches[bucket_id]++;
  }

  // The data has a uniform distribution of [1 .. 100]. So we expect
  // bucket boundaries around 25, 50, 75, 100, and roughly equal numbers of
  // batches of each ID.
  EXPECT_NEAR(250, batches[0], 25);
  EXPECT_NEAR(250, batches[1], 25);
  EXPECT_NEAR(250, batches[2], 25);
  EXPECT_NEAR(250, batches[3], 25);

  EXPECT_NEAR(25, maxlens[0] / batches[0], 5);
  EXPECT_NEAR(50, maxlens[1] / batches[1], 5);
  EXPECT_NEAR(75, maxlens[2] / batches[2], 5);
  EXPECT_NEAR(100, maxlens[3] / batches[3], 5);
}

TEST(RecordBatcher, FullEpoch) {
  const int N = 1000;
  const string filename =
      io::JoinPath("/tmp", "full_epoch");
  GenerateTestData(filename, N, false /* random_value */);

  BasicRecordYielder::Options yopts;
  yopts.file_pattern = strings::StrCat("tfrecord:", filename);
  yopts.seed = 301;
  yopts.bufsize = 10;
  yopts.parallelism = 1;

  RecordBatcher::Options bopts;
  bopts.bucket_upper_bound = {20, 50, 90, 120};
  bopts.bucket_batch_limit = {8, 4, 2, 1};
  bopts.flush_every_n = N;  // Same number of records in the data file.

  RecordBatcher batcher(bopts, BasicRecordYielder::New(yopts), new TestRP());
  int64 bucket_id;
  TensorVec batch;
  std::vector<string> records;
  while (records.size() < N) {
    batcher.GetNext(&bucket_id, &batch);
    const Tensor& t = batch[0];
    for (int j = 0; j < t.dim_size(0); ++j) {
      records.push_back(t.vec<string>()(j));
    }
  }
  ASSERT_EQ(N, records.size());
  // We expect to see exactly non-duplicated N records.
  std::sort(records.begin(), records.end());
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(strings::Printf("%010d", i), records[i]);
  }
}

}  // namespace lingvo
}  // namespace tensorflow
