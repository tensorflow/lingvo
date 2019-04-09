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

#ifndef LINGVO_CORE_OPS_RECORD_BATCHER_H_
#define LINGVO_CORE_OPS_RECORD_BATCHER_H_

#include <cstddef>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "lingvo/core/ops/mutex.h"
#include "lingvo/core/ops/record_yielder.h"

namespace tensorflow {
namespace lingvo {

// We use a vector of Tensor to represent a training example or a
// batch of merged training examples.  In the latter case, by
// convention, the 1st dimension of every Tensor represents the batch
// dimension.
typedef std::vector<Tensor> TensorVec;

// An interface for processing records yielded by RecordYielder into a
// training example (single one or batched).
class RecordProcessor {
 public:
  virtual ~RecordProcessor() {}

  // Parses 'record' (typically a protocol buffer) and fills in
  // 'sample' a vector of Tensor representing a training example.
  //
  // 'bucket_key' is an auxilary annotation extracted from the record
  // used by RecordBatcher to bucketize training examples. Typically,
  // 'bucket_key' is just a length measure about the record.
  virtual Status Process(const Rope& record, int64* bucket_key,
                         TensorVec* sample) = 0;

  // Gives a list of training 'samples', all of which returned by Process() and
  // bucketized into 'bucket_id'-th bucket, merges them into a single 'batch'.
  virtual Status Merge(int64 bucket_id, const std::vector<TensorVec>& samples,
                       TensorVec* batch) = 0;
};

// RecordBatcher takes a RecordYielder, batches the record yielded and
// converts them into training examples.
class RecordBatcher {
 public:
  // Uses 'yielder' to produce records and 'processor' to produce
  // training example batches.
  //
  // The bucketing scheme is specified by 'bucket_upper_bound' and
  // 'bucket_batch_limit'. Records with bucket_key within
  //   (bucket_upper_bound[i-1], bucket_upper_bound[i]]
  // is put into i-th bucket and as soon as i-th bucket contains
  // more than bucket_batch_limit[i] samples, RecordBatcher yields
  // one training batch.
  struct Options {
    // REQUIRES: bucket_upper_bound.size() == bucket_batch_limit.size()
    std::vector<int64> bucket_upper_bound;
    std::vector<int64> bucket_batch_limit;

    // If non-zero, optimize bucket_upper_bound values (except the last one)
    // every n records based on input lengths.
    int64 bucket_adjust_every_n = 0;

    // If non-zero, flushes all batches buffered so far every these
    // many records are yielded.
    int64 flush_every_n = 0;

    // Number of threads to use for record batcher, each thread
    // fills separate batches based on bucket limits.
    int64 num_threads = 1;
  };
  RecordBatcher(const Options& opts, RecordYielder* yielder,
                RecordProcessor* processor);

  ~RecordBatcher();

  // Returns the a training batch in 'batch' and the batch comes out
  // from 'bucket_id'-th bucket.
  void GetNext(int64* bucket_id, TensorVec* batch);

 private:
  typedef RecordBatcher ME;
  typedef std::vector<TensorVec> Batch;
  // FlushList is a list of bucket id and one batch for that bucket.
  typedef std::vector<std::pair<int64, Batch>> FlushList;

  // Owned.
  Options opts_;
  RecordYielder* yielder_ = nullptr;
  RecordProcessor* processor_ = nullptr;
  thread::ThreadPool* processor_thread_ = nullptr;
  thread::ThreadPool* merger_thread_ = nullptr;

  Mutex mu_;
  int64 curr_bucket_ GUARDED_BY(mu_) = -1;
  TensorVec curr_ GUARDED_BY(mu_);
  bool stop_ GUARDED_BY(mu_) = false;
  Condition curr_empty_;
  Condition curr_non_empty_;
  int64 records_yielded_ GUARDED_BY(mu_) = 0;
  int64 total_records_yielded_ GUARDED_BY(mu_) = 0;
  int64 total_records_skipped_ GUARDED_BY(mu_) = 0;
  std::vector<Batch> buckets_ GUARDED_BY(mu_);
  FlushList to_flush_ GUARDED_BY(mu_);
  Condition to_flush_empty_;
  Condition to_flush_non_empty_;
  std::time_t start_time_;  // Not necessary to guard.
  std::time_t last_log_update_time_ GUARDED_BY(mu_);
  int64 next_status_update_duration_seconds_ GUARDED_BY(mu_) = 60;

  std::vector<int64> length_histogram_;
  std::vector<int64> bucket_upper_bound_;

  // Conditions.
  bool CurrEmpty() const SHARED_LOCKS_REQUIRED(mu_) {
    return stop_ || curr_.empty();
  }

  bool CurrNonEmpty() const SHARED_LOCKS_REQUIRED(mu_) {
    return stop_ || !curr_.empty();
  }

  bool ToFlushEmpty() const SHARED_LOCKS_REQUIRED(mu_) {
    return stop_ || to_flush_.empty();
  }

  bool ToFlushNonEmpty() const SHARED_LOCKS_REQUIRED(mu_) {
    return stop_ || !to_flush_.empty();
  }

  void ProcessorLoop();
  void MergerLoop();

  void AdjustBuckets() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void FlushAllBuckets() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void IncrementHistogram(int64 bucket) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // For performance debugging.
  void WaitForCurrEmpty() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WaitForCurrNonEmpty() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WaitForToFlushEmpty() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WaitForToFlushNonEmpty() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RecordBatcher);
};

}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_RECORD_BATCHER_H_
