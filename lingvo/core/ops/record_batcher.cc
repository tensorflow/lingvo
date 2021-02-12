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

// Implementation notes:
//
// * RecordBatcher uses two threadpools. processor_thread_ and
//   merger_thread_.
//
// * Processor threads
//
//   * call the yielder to get a Record and process it into a TensorVec.
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
//   can change the implementation of processor->Merge() to leverage
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

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"

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
      to_flush_non_empty_(this, &ME::ToFlushNonEmpty),
      bucket_upper_bound_(opts_.bucket_upper_bound) {
  CHECK_EQ(opts_.bucket_upper_bound.size(), opts_.bucket_batch_limit.size());
  buckets_.resize(opts_.bucket_upper_bound.size());
  length_histogram_.resize(opts_.bucket_upper_bound.back() + 1, 0);
  start_time_ = std::time(nullptr);
  {
    absl::MutexLock l(&mu_);
    last_log_update_time_ = start_time_;
  }

  merger_thread_->Schedule([this]() {
    MergerLoop();
    absl::MutexLock l(&mu_);
    merger_loop_done_ = true;
  });
}

RecordBatcher::~RecordBatcher() {
  {
    absl::MutexLock l(&mu_);
    stop_ = true;
  }
  delete processor_thread_;
  delete merger_thread_;
  yielder_->Close();
  delete processor_;
}

Status RecordBatcher::EnsureInitialized(OpKernelContext* ctx) {
  if (is_initialized_) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(processor_->Initialize(ctx));
  for (int i = 0; i < opts_.num_threads; i++) {
    processor_thread_->Schedule([this]() {
      ProcessorLoop();
      absl::MutexLock l(&mu_);
      processor_loop_done_count_++;
    });
  }
  is_initialized_ = true;
  LOG(INFO) << "batcher initialized";
  return Status::OK();
}

Status RecordBatcher::GetNext(OpKernelContext* ctx, int64* bucket,
                              TensorVec* batch) {
  absl::MutexLock l(&mu_);
  TF_RETURN_IF_ERROR(EnsureInitialized(ctx));
  // Wait for either curr to be non-empty, or for the merger thread to be
  // complete.
  WaitForCurrNonEmpty();

  // If the buffer is still empty, it must be because the merger loop is done
  // due to an EoF.
  if (curr_.empty()) {
    CHECK(merger_loop_done_);
    return stop_status_;
  }

  *bucket = curr_bucket_;
  curr_bucket_ = -1;
  using std::swap;
  swap(*(batch), curr_);
  curr_.clear();
  return Status::OK();
}

void RecordBatcher::IncrementHistogram(int64 bucket) {
  if (bucket > bucket_upper_bound_.back()) return;
  length_histogram_[bucket]++;
}

void RecordBatcher::AdjustBuckets() {
  // The length histogram is too big to compute with quickly.
  // We distill it down to a histogram with bins of equal cost (equal area).
  int64 ideal_cost = 0;
  for (int i = 0; i < length_histogram_.size(); i++) {
    ideal_cost += length_histogram_[i] * i;
  }
  if (ideal_cost == 0) {
    return;
  }

  // The histogram is pairs of (upper_length_bound, items_in_bucket).
  std::vector<std::pair<int64, int64>> compact_histogram;
  int64 cost_so_far = 0;
  int64 count_in_bucket = 0;
  int bucket_index = 0;
  const int kCompactBuckets = 500;
  for (int i = 0; i < length_histogram_.size(); i++) {
    cost_so_far += length_histogram_[i] * i;
    count_in_bucket += length_histogram_[i];
    int64 cost_target = ideal_cost * ((bucket_index + 1.0) / kCompactBuckets);
    if (cost_so_far >= cost_target &&
        compact_histogram.size() != kCompactBuckets) {
      compact_histogram.push_back(std::make_pair(i, count_in_bucket));
      count_in_bucket = 0;
      bucket_index++;
    }
  }
  // Make sure that the last bucket ends at the user-specified upper bound.
  compact_histogram.push_back(
      std::make_pair(bucket_upper_bound_.back(), count_in_bucket));

  // Clear the fine-grained histogram to prepare for the next cycle.
  length_histogram_.assign(opts_.bucket_upper_bound.back() + 1, 0);

  const int num_lengths = compact_histogram.size();
  const int num_buckets = bucket_upper_bound_.size();

  // Now we shrink the compact histogram even further using dynamic programming.
  // c[i, j, k]: the cumulative cost for the first i histogram groups, if the
  // next bucketing point is at index j (j > i), and there are k more bucketing
  // points available to use for the remaining elements.
  Tensor cost(DT_INT64, TensorShape({num_lengths, num_lengths, num_buckets}));
  auto c = cost.tensor<int64, 3>();

  // s(i, j) is the total cost of computing the items in histogram bucket i
  // when padding them out to the size of bucket j.
  auto s = [&compact_histogram](int i, int j) {
    const int64 bucket_j_width = compact_histogram[j].first;
    const int64 bucket_i_count = compact_histogram[i].second;
    return bucket_i_count * bucket_j_width;
  };

  for (int i = 0; i < num_lengths; i++) {
    for (int j = i + 1; j < num_lengths; j++) {
      // If there are no buckets left and j is the next bucket, the
      // cost is \sum_i<j s(i,j).
      c(i, j, 0) = s(i, j);
      if (i > 0) {
        c(i, j, 0) += c(i - 1, j, 0);
      }
      // When we have some buckets to use, we can choose to insert bucket
      // boundaries to reduce computation.
      for (int k = 1; k < num_buckets; k++) {
        if (i > 0) {
          // When we choose to put a bucket boundary here at position i, we
          // can compute these new items with minimal padding [s(i, i)].
          int64 cost_choose = c(i - 1, i, k - 1) + s(i, i);
          // If we don't put a bucket boundary here, we have to wait until
          // position j, which means extra padding. [s(i, j)].
          int64 cost_not_choose = c(i - 1, j, k) + s(i, j);
          c(i, j, k) = std::min(cost_choose, cost_not_choose);
        } else {
          c(i, j, k) = s(i, j);
        }
      }
    }
  }

  std::vector<int> buckets;
  buckets.push_back(num_lengths - 1);
  const int64 best_cost = c(num_lengths - 2, num_lengths - 1, num_buckets - 1);
  int64 remaining_cost = best_cost;
  for (int i = num_lengths - 2; i > 0; i--) {
    int buckets_left = num_buckets - buckets.size();
    if (buckets_left <= 0) break;
    int prev_bucket = buckets.back();
    int64 cost_choose = c(i - 1, i, buckets_left - 1) + s(i, i);
    int64 cost_not_choose =
        c(i - 1, prev_bucket, buckets_left) + s(i, prev_bucket);
    if (remaining_cost == cost_choose) {
      buckets.push_back(i);
      remaining_cost -= s(i, i);
    } else if (remaining_cost == cost_not_choose) {
      remaining_cost -= s(i, prev_bucket);
    } else {
      // This didn't make sense; keep the buckets as they are.
      LOG(WARNING) << "AdjustBuckets: backtrace failed.";
      return;
    }
  }

  std::reverse(buckets.begin(), buckets.end());

  // We keep the same maximum value that the user entered, but all other
  // boundaries are updated.
  std::vector<string> bucket_strings;
  for (int i = 0; i < buckets.size() - 1; i++) {
    bucket_upper_bound_[i] = compact_histogram[buckets[i]].first;
    bucket_strings.push_back(strings::StrCat(bucket_upper_bound_[i]));
  }
  bucket_strings.push_back(strings::StrCat(bucket_upper_bound_.back()));

  // Compute the amount of padding waste from choosing this bucket assignment.
  LOG(INFO) << "Buckets: [" << str_util::Join(bucket_strings, ", ") << "] "
            << "Waste: "
            << (best_cost - static_cast<float>(ideal_cost)) / best_cost;
}

void RecordBatcher::FlushAllBuckets() {
  for (int i = 0; i < buckets_.size(); ++i) {
    if (!buckets_[i].empty()) {
      CHECK_LE(static_cast<int64>(buckets_[i].size()),
               opts_.bucket_batch_limit[i]);
      to_flush_.push_back({i, std::move(buckets_[i])});
      buckets_[i].clear();
    }
  }
}

void RecordBatcher::ProcessorLoop() {
  // Multiply next_status_update_duration_seconds_ by 2 every update.
  const int64 status_update_duration_multiplier = 2;
  std::vector<int64> out_of_range_buckets;
  while (true) {
    {
      absl::MutexLock l(&mu_);
      if (stop_) {
        return;
      }
    }

    // Get the next record.
    Record record;
    record.source_id = kDefaultSourceId;
    Status s = yielder_->Yield(&record);
    // yielder may return OutOfRange to indicate end-of-epoch.
    // Set the out status appropriately and return.
    if (errors::IsOutOfRange(s)) {
      LOG(INFO) << "Yielder out of range: " << s;
      absl::MutexLock l(&mu_);
      stop_status_ = s;
      stop_ = true;
      return;
    }

    if (!s.ok()) {
      LOG(WARNING) << s;
      continue;
    }

    // Parse the record.
    int64 bucket;
    TensorVec sample;
    s = processor_->Process(record, &bucket, &sample);
    if (!s.ok()) {
      // Print error message. Some example processors use CANCELLED for data
      // that are filtered out. Print only first 10 such errors.
      if (errors::IsCancelled(s)) {
        {
          absl::MutexLock l(&mu_);
          ++total_records_skipped_;

          static int log_counter = 0;
          if (log_counter < 10) {
            log_counter++;
            LOG(WARNING) << s;
          }
        }
      } else {
        for (const auto& msg : opts_.fatal_errors) {
          if (absl::StrContains(s.error_message(), msg)) {
            // The error is in the list of fatal errors.
            LOG(FATAL) << s;
          }
        }
        if (errors::IsNotFound(s) || errors::IsPermissionDenied(s)) {
          // Terminates program if an unregistered custom op is used by
          // the processor, or any access permission denied error.
          //
          // Consider setting *out_status with s and returning, instead
          // of killing program?
          LOG(FATAL) << s;
        } else {
          LOG(WARNING) << s;
          {
            absl::MutexLock l(&mu_);
            ++total_records_skipped_;
          }
        }
      }
      continue;
    }

    absl::MutexLock l(&mu_);

    if (opts_.bucket_adjust_every_n > 0) {
      const int64 records_processed =
          total_records_yielded_ + total_records_skipped_;
      if (records_processed % opts_.bucket_adjust_every_n == 0 &&
          total_records_yielded_ > 0) {
        AdjustBuckets();
      }
      IncrementHistogram(bucket);
    }

    // Figure out which bucket it belongs to.
    auto iter = std::lower_bound(bucket_upper_bound_.begin(),
                                 bucket_upper_bound_.end(), bucket);

    if (iter == bucket_upper_bound_.end()) {
      VLOG(1) << "Skip. bucket out-of-range " << bucket;
      if (out_of_range_buckets.size() < 10) {
        out_of_range_buckets.push_back(bucket);
      }
      ++total_records_skipped_;
    } else {
      // Figure out which buckets we should return to the consumer.
      // A bucket (id-th) is full.
      const int id = iter - bucket_upper_bound_.begin();
      const int64 batch_limit = opts_.bucket_batch_limit[id];
      if (buckets_[id].size() + 1 == batch_limit) {
        WaitForToFlushEmpty();
        if (stop_) {
          return;
        }
      }
      // Invariant is either we don't need to flush this bucket after adding a
      // new element to it, or to_flush_ is empty and we can flush this bucket.
      CHECK(buckets_[id].size() + 1 < batch_limit || to_flush_.empty());
      buckets_[id].push_back({bucket, std::move(sample)});
      if (buckets_[id].size() == batch_limit) {
        to_flush_.push_back({id, std::move(buckets_[id])});
        buckets_[id].clear();
      }
      CHECK_LT(buckets_[id].size(), batch_limit);  // invariant.

      ++records_yielded_;
      ++total_records_yielded_;

      if (opts_.flush_every_n > 0 && records_yielded_ >= opts_.flush_every_n) {
        FlushAllBuckets();
        records_yielded_ = 0;
      }
    }

    std::time_t current_time = std::time(nullptr);
    if (current_time - last_log_update_time_ >
        next_status_update_duration_seconds_) {
      LOG(INFO) << current_time - start_time_
                << " total seconds passed. Total records yielded: "
                << total_records_yielded_
                << ". Total records skipped: " << total_records_skipped_;
      for (auto bucket : out_of_range_buckets) {
        LOG(INFO) << "Out-of-range sample: " << bucket;
      }
      CHECK_GT(total_records_yielded_, 0)
          << "No records were yielded. Most likely the input is misconfigured!";
      out_of_range_buckets.clear();
      last_log_update_time_ = current_time;
      next_status_update_duration_seconds_ *= status_update_duration_multiplier;
    }
  }
}

void RecordBatcher::MergerLoop() {
  FlushList to_flush;
  std::vector<TensorVec> samples;
  TensorVec merged;
  bool continue_loop = true;
  while (continue_loop) {
    {
      absl::MutexLock l(&mu_);
      WaitForToFlushNonEmpty();
      if (stop_ && stop_status_.ok()) {
        // The object is being destroyed, just exit.
        return;
      } else if (ProcessorsDone()) {
        // The yielder hit EOF and all processors are done.
        // Flush all buckets, and then signal to exit the merger
        // loop once all items are flushed to curr_.
        FlushAllBuckets();
        continue_loop = false;
      }

      to_flush = std::move(to_flush_);
      to_flush_.clear();
    }

    // Now, flush out batches we accumulated. Typically, to_flush has only 1
    // batch unless flush_every_n is > 0.
    for (auto& p : to_flush) {
      const int64 id = p.first;
      const int32 num = p.second.size();
      Tensor bucket_keys(DT_INT32, {num});
      auto t_bucket_keys = bucket_keys.flat<int32>();
      for (int i = 0; i < num; ++i) {
        auto processed = p.second[i];
        t_bucket_keys(i) = processed.bucket_key;
        samples.push_back(std::move(processed.sample));
      }
      merged.clear();
      Status s = processor_->Merge(bucket_upper_bound_[id], samples, &merged);
      samples.clear();
      if (!s.ok()) {
        LOG(WARNING) << "Failed to create a batch: " << s;
      } else {
        merged.push_back(bucket_keys);
        absl::MutexLock l(&mu_);
        WaitForCurrEmpty();

        // If stopped due to destructor, just exit, since there should be no
        // further calls to GetNext().
        if (stop_ && stop_status_.ok()) {
          return;
        }
        curr_bucket_ = id;
        curr_ = std::move(merged);
      }
    }
    to_flush.clear();
  }
}

}  // namespace lingvo
}  // namespace tensorflow
