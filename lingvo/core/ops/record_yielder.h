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

#ifndef LINGVO_CORE_OPS_RECORD_YIELDER_H_
#define LINGVO_CORE_OPS_RECORD_YIELDER_H_

#include <algorithm>
#include <atomic>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "lingvo/core/ops/mutex.h"
#include "lingvo/core/ops/rope.h"

namespace tensorflow {
namespace lingvo {

// RecordYielder produces value records from files in a random order.
//
// It guarantees that:
//   1) all records are yielded within every epoch;
//   2) each record is yielded only once within every epoch;
//   3) the order in which records are yielded are highly randomized.
//   4) the peak memory usage is roughly avg record size *
//      (opts.bufsize + opts.parellelism * 16).
//
// Usage example:
//   RecordYielder::Options opts;
//   opts.file_pattern = <file_pattern>;
//   opts.seed = 301;
//   opts.bufsize = 1000000;    // A randomized buffer with 1M records.
//   opts.parallelism = 8;      // Use 8 iterators to iterate through all files.
//   RecordYielder* yielder = RecordYielder::New(opts);
//   Rope val;
//   while (true) {
//     yielder->Yield(&val);
//     // process val
//   }
//   yielder->Close();
//
// RecordYielder can be accessed by multiple threads concurrently.
class RecordYielder {
 public:
  struct Options {
    // The set of files to yield records from.  file_pattern follows:
    // [<type name>:]<glob pattern>, where <type name> must have an
    // associated factory method through registration via New().
    string file_pattern;

    // Random seed. It determines how data files are shuffled.
    int64 seed = 0;

    // Each epoch, all files are first shuffled according to the
    // random seed and the epoch number, and then all files are
    // left-shifted by file_shuffle_shift_ratio * num_files slots.  If
    // file_shuffle_shift_ratio is not within [0, 1), the
    // implementation clips it to [0, 1).
    float file_shuffle_shift_ratio = 0;

    // Randomization buffer keeps these many records.
    int64 bufsize = 1;

    // Uses this many concurrent iterators to iterate through files.
    int32 parallelism = 1;
  };

  // Register a method to create a RecordYielder for the 'type_name'.
  typedef std::function<RecordYielder*(const RecordYielder::Options&)>
      FactoryMethod;
  static bool Register(const string& type_name, FactoryMethod method);

  // Returns a record yielder according to 'opts'.
  static RecordYielder* New(Options opts);

  // Yields one 'value'.
  Status Yield(Rope* value);

  // Stop this yielder and then delete it.
  void Close();

  // Returns the current epoch number.
  int64 current_epoch() const { return epoch_; }

 protected:
  explicit RecordYielder(const Options& opts);

  virtual ~RecordYielder();

  // Subclass should implement ShardLoop which processes all records
  // in the 'shard'.
  struct Shard {
    int index;                      // Shard index.
    std::vector<string> filenames;  // File names given to this shard.
    Notification done;        // Notified when this shard is done.
    Status status;                  // Shard status.
  };
  virtual void ShardLoop(Shard* shard) = 0;
  virtual Status MatchFiles(const string& patterns,
                            std::vector<string>* filenames);

  // Returns true iff 's' indicates the yielder should stop.
  bool ShouldFinish(const Status& s);

  // Adds 'values' into the random shuffling buffer buf_.
  bool Add(std::vector<Rope>* values);

 private:
  typedef RecordYielder ME;

  Options opts_;

  // Background threads. Owned.
  thread::ThreadPool* thread_;

  // Epoch number.
  std::atomic<int64> epoch_;

  Mutex mu_;

  // Turned to true when the yielder is deleted.
  bool stop_ GUARDED_BY(mu_) = false;
  Status status_ GUARDED_BY(mu_);

  // PRG used for randomization.
  std::mt19937_64 rnd_ GUARDED_BY(mu_);

  // Randomization buffer.
  std::vector<Rope> buf_ GUARDED_BY(mu_);

  // True iff we are draining an epoch.
  bool epoch_end_ = false;

  int64 num_records_yielded_in_epoch_ = 0;

  // Trigger when the main loop has exited.
  Notification main_loop_done_;

  // Conditions.
  Condition buf_empty_;
  bool BufEmpty() const SHARED_LOCKS_REQUIRED(mu_) {
    return stop_ || buf_.empty();
  }

  Condition buf_not_full_;
  bool BufNotFull() const SHARED_LOCKS_REQUIRED(mu_) {
    return stop_ || static_cast<int64>(buf_.size()) < opts_.bufsize;
  }

  Condition buf_enough_;
  bool BufEnough() const SHARED_LOCKS_REQUIRED(mu_) {
    // NOTE: Unless we are finishing an epoch, we want to make sure
    // the buf_ contains enough randomized elements before yielding any.
    return stop_ || !status_.ok() || (epoch_end_ && !buf_.empty()) ||
           (!epoch_end_ && static_cast<int64>(buf_.size()) >=
                               std::max<int64>(1, opts_.bufsize / 2));
  }

  void ExtractValue(Rope* value) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (opts_.seed == 0) {
      // Randomize at the consumer side as well.
      const auto index = rnd_() % buf_.size();
      *value = std::move(buf_[index]);
      if (index != buf_.size() - 1) {
        buf_[index] = std::move(buf_.back());
      }
    } else {
      *value = std::move(buf_.back());
    }
    buf_.pop_back();
  }

  void Start();
  void MainLoop();

  // For performance debugging.
  void WaitForBufEnough() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RecordYielder);
};

}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_RECORD_YIELDER_H_
