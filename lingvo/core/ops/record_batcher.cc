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

// Implemention notes:
//
// * RecordBatcher uses two threadpools. processor_thread_ and
//   merger_thread_.
//
// * Processor threads
//
//   * call the yielder to get a record (a string) and process it into
//     a TensorVec.
//
//   * Processed TensorVec are put into buckets according to the
//     bucket key returned by processor->Process().
//
//   * When one bucket is full (according to bucket_batch_limit), all
//     TensorVec accumulated in that bucket is handed off to
//     to_flush_.
//
//   * If to_flush_ is non-empty, the processor thread blocks.
//
// * The only merger thread pulls off TensorVecs from to_flush_ and
//   calls processor->Merge() to merge samples into a single batch. It
//   then hands the merged batch into curr_. If there is already one
//   unconsumed batch, the merger thread blocks.
//
//   NOTE: merger_thread_ itself is single-threaded. We expect that if
//   processor->Merge() becomes bottleneck (memory copy bounded), we
//   can change the implemention of processor->Merge() to leverage
//   multi-threaded merge.
//
//
// Peak memory usage estimate given the above algorithm:
//    yielder buffer memory      // RecordYielder.file_buffer_size
//  + all TensorVecs in buckets  // Sum of RecordBatcher.bucket_batch_limit
//  + to_flush_                  // The batch to be merged.
//  + to_flush                   // The batch being merged.
//  + merged                     // The batch being merged.
//  + curr_                      // The batch to be consumed.
//
//  If we assume each sample is roughly M bytes (in its string format
//  or tensor format), each output batch has the batch size B, we can
//  estimate the peak memory usage of one RecordYielder+RecordBatcher
//  is roughly
//
//    M * (file_buffer_size + sum(bucket_batch_limit) * 2 + B * 2)

#include "lingvo/core/ops/record_batcher.h"

#include <algorithm>
#include <utility>

#include "tensorflow/core/platform/env.h"
#include "lingvo/core/ops/mutex.h"

namespace tensorflow {
namespace lingvo {

RecordBatcher::RecordBatcher(const Options& opts, RecordYielder* yielder,
                             RecordProcessor* processor)
    : opts_(opts),
      yielder_(yielder),
      processor_(processor),
      processor_thread_(new thread::ThreadPool(
          Env::Default(), ThreadOptions(), "record_batcher_processor",
          opts_.num_threads, /* low_latency_hint */ false)),
      merger_thread_(new thread::ThreadPool(Env::Default(), ThreadOptions(),
                                            "record_batcher_merger", 1,
                                            /* low_latency_hint */ false)),
      curr_empty_(this, &ME::CurrEmpty),
      curr_non_empty_(this, &ME::CurrNonEmpty),
      to_flush_empty_(this, &ME::ToFlushEmpty),
      to_flush_non_empty_(this, &ME::ToFlushNonEmpty) {
  CHECK_EQ(opts_.bucket_upper_bound.size(), opts_.bucket_batch_limit.size());
  buckets_.resize(opts_.bucket_upper_bound.size());
  for (int i = 0; i < opts_.num_threads; i++) {
    processor_thread_->Schedule([this]() { ProcessorLoop(); });
  }
  merger_thread_->Schedule([this]() { MergerLoop(); });
}

RecordBatcher::~RecordBatcher() {
  {
    MutexLock l(&mu_);
    stop_ = true;
  }
  delete processor_thread_;
  delete merger_thread_;
  yielder_->Close();
  delete processor_;
}

void RecordBatcher::GetNext(int64* bucket, TensorVec* batch) {
  MutexLock l(&mu_);
  WaitForCurrNonEmpty();
  *bucket = curr_bucket_;
  curr_bucket_ = -1;
  using std::swap;
  swap(*(batch), curr_);
  curr_.clear();
}

void RecordBatcher::ProcessorLoop() {
  int64 current_epoch = yielder_->current_epoch();
  while (true) {
    {
      MutexLock l(&mu_);
      if (stop_) return;
    }

    // Get the next record.
    Rope record;
    Status s = yielder_->Yield(&record);
    if (!s.ok()) {
      LOG(WARNING) << s;
      continue;
    }

    // Parse the record.
    int64 bucket;
    TensorVec sample;
    s = processor_->Process(record, &bucket, &sample);
    if (!s.ok()) {
      // Print error message except for CANCELLED error. NmtExampleProcessor
      // uses CANCELLED for data that are filtered out.
      if (!errors::IsCancelled(s)) {
        LOG(WARNING) << s;
      }
      continue;
    }

    // Figure out which bucket it belongs to.
    auto iter = std::lower_bound(opts_.bucket_upper_bound.begin(),
                                 opts_.bucket_upper_bound.end(), bucket);

    MutexLock l(&mu_);

    if (iter == opts_.bucket_upper_bound.end()) {
      VLOG(1) << "Skip. bucket out-of-range " << bucket;
      ++total_records_skipped_;
    } else {
      if (opts_.flush_every_n > 0 && records_yielded_ >= opts_.flush_every_n) {
        WaitForToFlushEmpty();
        if (stop_) return;
        if (opts_.flush_every_n > 0 &&
            records_yielded_ >= opts_.flush_every_n) {
          CHECK(to_flush_.empty());

          // Need to flush all buckets.
          records_yielded_ = 0;
          for (int i = 0; i < buckets_.size(); ++i) {
            if (!buckets_[i].empty()) {
              CHECK_LE(static_cast<int64>(buckets_[i].size()),
                       opts_.bucket_batch_limit[i]);
              to_flush_.push_back({i, std::move(buckets_[i])});
              buckets_[i].clear();
            }
          }
        }
      }

      // Figure out which buckets we should return to the consumer.
      // A bucket (id-th) is full.
      const int id = iter - opts_.bucket_upper_bound.begin();
      const int64 batch_limit = opts_.bucket_batch_limit[id];
      CHECK_LE(buckets_[id].size(), batch_limit);  // invariant.
      while (buckets_[id].size() == batch_limit) {
        if (!to_flush_.empty()) {
          WaitForToFlushEmpty();
          if (stop_) return;
          continue;
        }
        to_flush_.push_back({id, std::move(buckets_[id])});
        buckets_[id].clear();
      }
      buckets_[id].push_back(std::move(sample));
      CHECK_LE(buckets_[id].size(), batch_limit);  // invariant.

      ++records_yielded_;
      ++total_records_yielded_;
    }

    if (current_epoch != yielder_->current_epoch()) {
      if (current_epoch < 10) {
        LOG(INFO) << "Past end of epoch " << current_epoch
                  << ". Total records yielded: " << total_records_yielded_
                  << ". Total records skipped: " << total_records_skipped_
                  << ". Only logging first 10 epochs to INFO.";
      } else {
        VLOG(1) << "Past end of epoch " << current_epoch
                << ". Total records yielded: " << total_records_yielded_
                << ". Total records skipped: " << total_records_skipped_ << ".";
      }
      current_epoch = yielder_->current_epoch();
    }
  }
}

void RecordBatcher::MergerLoop() {
  FlushList to_flush;
  TensorVec merged;
  while (true) {
    {
      MutexLock l(&mu_);
      WaitForToFlushNonEmpty();
      if (stop_) return;
      to_flush = std::move(to_flush_);
      to_flush_.clear();
    }

    // Now, flush out batches we accumulated.  Typically, to_flush has
    // only 1 batch unless flush_every_n is > 0.
    for (auto& p : to_flush) {
      const int64 id = p.first;
      auto* samples = &p.second;
      merged.clear();
      Status s = processor_->Merge(id, *samples, &merged);
      samples->clear();
      if (!s.ok()) {
        LOG(WARNING) << "Failed to create a batch: " << s;
      } else {
        MutexLock l(&mu_);
        WaitForCurrEmpty();
        if (stop_) return;
        curr_bucket_ = id;
        curr_ = std::move(merged);
      }
    }
    to_flush.clear();
  }
}

}  // namespace lingvo
}  // namespace tensorflow
