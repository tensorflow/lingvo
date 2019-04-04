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

// An interface to iterate sequentially a set of record (Rope).
class RecordIterator {
 public:
  virtual ~RecordIterator() {}

  // Get the next record. If EOF, returns false. Otherwise returns true and
  // fills in 'key' and 'value'.
  virtual bool Next(string* key, Rope* value) = 0;

  // Register a method to create a RecordIterator for the 'type_name'.
  typedef std::function<RecordIterator*(const string&)> FactoryMethod;
  static bool Register(const string& type_name, FactoryMethod method);

  // As above, but also register a custom method for parsing file_pattern
  // strings into lists of shards.
  typedef std::function<Status(const string&, std::vector<std::string>*)>
      PatternParserMethod;
  static bool RegisterWithPatternParser(const string& type_name,
                                        FactoryMethod method,
                                        PatternParserMethod parser_method);

  // Returns a record iterator for 'filename' of 'type_name'.
  static RecordIterator* New(const string& type_name, const string& filename);

  // Returns the prefix in a file pattern, or an empty string if not exist.
  // Example: "tfrecord:data_dir/data.tfrecord" => "tfrecord"
  static string GetFilePatternPrefix(const string& file_pattern);

  // Parse a file pattern into a list of matching files.
  static Status ParsePattern(const string& type_name,
                             const string& file_pattern_list,
                             std::vector<string>* filenames);
};

// RecordYielder defines an interface that should be used for producing value
// records from files in a random order. Most users should use
// BasicRecordYielder and BasicRecordYielder::New (see example below).
//
// RecordYielder guarantees that the order in which records are yielded are
// highly randomized.
//
// Usage example:
//   BasicRecordYielder::Options opts;
//   opts.file_pattern = <file_pattern>;
//   opts.seed = 301;
//   opts.bufsize = 1000000;    // A randomized buffer with 1M records.
//   opts.parallelism = 8;      // Use 8 iterators to iterate through all files.
//   RecordYielder* yielder = BasicRecordYielder::New(opts);
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
  virtual ~RecordYielder();

  // Yields one 'value' and the id of the source.
  // source_id is useful to specify the source when reading from multiple input
  // sources. To read from multiple input sources and keep track of the
  // source id, create a WeightedRecordYielder and create a BasicRecordYielder
  // for each source. Each BasicRecordYielder can be assigned a source_id,
  // which is assigned to the argument here.
  // A nullptr can be provided as input for source_id.
  virtual Status Yield(Rope* value, int* source_id) = 0;

  // Stop this yielder and then delete it.
  virtual void Close() = 0;
};

// BasicRecordYielder is a RecordYielder that implements a main loop and makes
// it possible to write a custom RecordYielder by only defining a shard loop.
// Most of the record yielders should inherit from this class.
//
// BasicRecordYielder guarantees that:
//   1) all records are yielded within every epoch;
//   2) each record is yielded only once within every epoch;
//   3) the order in which records are yielded are highly randomized.
//   4) the peak memory usage is roughly avg record size *
//      (opts.bufsize + opts.parellelism * 16).
class BasicRecordYielder : public RecordYielder {
 public:
  struct Options {
    // The set of files to yield records from.  file_pattern follows:
    // [<type name>:]<glob pattern>, where <type name> must have an
    // associated factory method through registration via New().
    //
    // TODO(zhifengc): Document it better the current support format.
    string file_pattern;

    // Random seed. It determines how data files are shuffled.
    int64 seed = 0;

    // Randomization buffer keeps these many records.
    int64 bufsize = 1;

    // Uses this many concurrent iterators to iterate through files.
    int32 parallelism = 1;

    // Source id to be supplied with yield.
    int32 source_id = 0;
  };

  // Returns a record yielder according to 'opts'. A caller is responsible for
  // calling Close when this yielder is no longer required. A caller shouldn't
  // delete the yielder.
  static BasicRecordYielder* New(Options opts);

  // Yields one 'value' and 'source_id' from which the value was read.
  Status Yield(Rope* value, int* source_id) override;

  // Stop this yielder and then delete it.
  void Close() override;

  // Returns the current epoch number.
  int64 current_epoch() const { return epoch_; }

 protected:
  explicit BasicRecordYielder(const Options& opts);

  ~BasicRecordYielder() override;

  // Subclass should implement ShardLoop which processes all records
  // in the 'shard'.
  struct Shard {
    int index;                      // Shard index.
    std::vector<string> filenames;  // File names given to this shard.
    Notification done;              // Notified when this shard is done.
    Status status;                  // Shard status.
  };
  void ShardLoop(Shard* shard);

  // Returns true iff 's' indicates the yielder should stop.
  bool ShouldFinish(const Status& s);

  // Adds 'values' into the random shuffling buffer buf_.
  bool Add(std::vector<Rope>* values);

 private:
  typedef BasicRecordYielder ME;

  Options opts_;
  string file_type_;

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

  TF_DISALLOW_COPY_AND_ASSIGN(BasicRecordYielder);
};

}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_RECORD_YIELDER_H_
