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

#include "absl/synchronization/mutex.h"
#include "lingvo/core/ops/record_yielder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

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

  // Parses 'record' (typically a protocol buffer) and fills
  // in 'sample' a vector of Tensor representing a training example.
  //
  // 'bucket_key' is an auxilary annotation extracted from the record
  // used by RecordBatcher to bucketize training examples. Typically,
  // 'bucket_key' is just a length measure about the record.
  virtual Status Process(const Record& record, int64* bucket_key,
                         TensorVec* sample) = 0;

  // Gives a list of training 'samples' returned by Process() and bucketed into
  // a bucket with size `bucket_size`, merges them into a single 'batch'.
  virtual Status Merge(int64 bucket_size, const std::vector<TensorVec>& samples,
                       TensorVec* batch) = 0;

  // Initializes the RecordProcessor. This should be invoked only once before
  // calling 'Process', since some processors expect some initialization with an
  // OpKernelContext before running.
  virtual Status Initialize(OpKernelContext* ctx) { return Status::OK(); }
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

    // Errors to treat as fatal.
    std::vector<string> fatal_errors;
  };
  RecordBatcher(const Options& opts, RecordYielder* yielder,
                RecordProcessor* processor);

  virtual ~RecordBatcher();

  // Returns the a training batch in 'batch' and the batch comes out
  // from 'bucket_id'-th bucket.
  // Processor threads will be launched on the first call to this function.
  Status GetNext(OpKernelContext* ctx, int64* bucket_id, TensorVec* batch);

 private:
  // Initializes the batcher and launches the processor threads, if not done.
  // This should be invoked on the first call of 'GetNext', since some
  // processors expect some initialization with an OpKernelContext before
  // running.
  Status EnsureInitialized(OpKernelContext* ctx);

  typedef RecordBatcher ME;
  struct Processed {
    int64 bucket_key;
    TensorVec sample;
  };
  typedef std::vector<Processed> Batch;

  // FlushList is a list of bucket id and one batch for that bucket.
  typedef std::vector<std::pair<int64, Batch>> FlushList;

  // Owned.
  Options opts_;
  RecordYielder* yielder_ = nullptr;
  RecordProcessor* processor_ = nullptr;
  thread::ThreadPool* processor_thread_ = nullptr;
  thread::ThreadPool* merger_thread_ = nullptr;
  absl::Mutex mu_;
  int64 curr_bucket_ ABSL_GUARDED_BY(mu_) = -1;
  TensorVec curr_ ABSL_GUARDED_BY(mu_);

  // True if either the yielder hits EOF or the destructor triggers.
  bool stop_ ABSL_GUARDED_BY(mu_) = false;

  // Status is not OK when a yielder hits an EOF.
  Status stop_status_ ABSL_GUARDED_BY(mu_);

  // True when the merger thread is finished.
  bool merger_loop_done_ ABSL_GUARDED_BY(mu_) = false;

  bool is_initialized_ = false;
  absl::Condition curr_empty_;
  absl::Condition curr_non_empty_;
  int64 records_yielded_ ABSL_GUARDED_BY(mu_) = 0;
  int64 total_records_yielded_ ABSL_GUARDED_BY(mu_) = 0;
  int64 total_records_skipped_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<Batch> buckets_ ABSL_GUARDED_BY(mu_);
  int64 processor_loop_done_count_ ABSL_GUARDED_BY(mu_) = 0;
  FlushList to_flush_ ABSL_GUARDED_BY(mu_);
  absl::Condition to_flush_empty_;
  absl::Condition to_flush_non_empty_;
  std::time_t start_time_;  // Not necessary to guard.
  std::time_t last_log_update_time_ ABSL_GUARDED_BY(mu_);
  int64 next_status_update_duration_seconds_ ABSL_GUARDED_BY(mu_) = 60;

  std::vector<int64> length_histogram_;
  std::vector<int64> bucket_upper_bound_;

  // Conditions.
  bool CurrEmpty() const ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return ((stop_ && stop_status_.ok()) ||  // The object is being destroyed
            curr_.empty());                  // We can push work onto curr_.
  }

  bool CurrNonEmpty() const ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return (!curr_.empty() ||    // There is data to deliver to GetNext().
            merger_loop_done_);  // There merger loop is done (no more data).
  }

  bool ToFlushEmpty() const ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return stop_ || to_flush_.empty();
  }

  bool ToFlushNonEmpty() const ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return ((stop_ && stop_status_.ok()) ||  // The object is being destroyed.
            !to_flush_.empty() ||            // There is work to flush.
            ProcessorsDone());  // All processor threads have exited.
  }

  bool ProcessorsDone() const ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return processor_loop_done_count_ == opts_.num_threads;
  }

  void ProcessorLoop();
  void MergerLoop();

  void AdjustBuckets() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void FlushAllBuckets() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void IncrementHistogram(int64 bucket) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // For performance debugging.
  void WaitForCurrEmpty() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WaitForCurrNonEmpty() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WaitForToFlushEmpty() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WaitForToFlushNonEmpty() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RecordBatcher);
};

}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_RECORD_BATCHER_H_
