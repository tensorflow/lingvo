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

#include <string>
#include <unordered_map>

#include "lingvo/core/ops/record_yielder.h"

#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "lingvo/core/ops/mutex.h"

namespace tensorflow {
namespace lingvo {

namespace {

struct Factory {
  Mutex mu;
  std::unordered_map<string, RecordYielder::FactoryMethod> methods;
};

Factory* GetFactory() {
  static Factory* factory = new Factory;
  return factory;
}

// Returns the prefix in a file pattern, or an empty string if not exist.
// Example: "tfrecord:data_dir/data.tfrecord" => "tfrecord"
string GetFilePatternPrefix(const string& file_pattern) {
  const auto prefix_end = file_pattern.find(':');
  if (prefix_end == string::npos) {
    return "";
  }
  // The prefix should not contain '/'. If so, it's likely part of the path.
  const auto first_slash = file_pattern.find('/');
  if (first_slash != string::npos && first_slash < prefix_end) {
    return "";
  }
  return file_pattern.substr(0, prefix_end);
}

}  // end namespace

bool RecordYielder::Register(const string& type_name, FactoryMethod method) {
  Factory* factory = GetFactory();
  MutexLock l(&factory->mu);
  bool ret = factory->methods.insert({type_name, std::move(method)}).second;
  CHECK(ret) << "Possibly duplicated registration: " << type_name;
  return ret;
}

RecordYielder* RecordYielder::New(Options opts) {
  string prefix = GetFilePatternPrefix(opts.file_pattern);
  if (!prefix.empty()) {
    opts.file_pattern.erase(0, prefix.size() + 1);
  }

  Factory* factory = GetFactory();
  MutexLock l(&factory->mu);
  const auto iter = factory->methods.find(prefix);
  if (iter == factory->methods.end()) {
    LOG(FATAL) << "Unable to create RecordYielder for format \""
               << prefix << "\"";
  }
  const RecordYielder::FactoryMethod& method = iter->second;
  RecordYielder* yielder = method(opts);
  yielder->Start();
  return yielder;
}

RecordYielder::RecordYielder(const Options& opts)
    : opts_(opts),
      thread_(new thread::ThreadPool(Env::Default(), ThreadOptions(),
                                     "record_yielder", 1 + opts.parallelism,
                                     /* low_latency_hint */ false)),
      epoch_(0),
      rnd_(opts.seed),
      buf_empty_(this, &ME::BufEmpty),
      buf_not_full_(this, &ME::BufNotFull),
      buf_enough_(this, &ME::BufEnough) {
  LOG(INFO) << this << " Record yielder start";
  if (opts.seed == 0) {
    LOG(INFO) << "Randomly seed RecordYielder.";
    rnd_.seed(std::random_device{}());
  }
}

RecordYielder::~RecordYielder() {}

void RecordYielder::Start() {
  thread_->Schedule([this]() { MainLoop(); });
}

void RecordYielder::Close() {
  {
    MutexLock l(&mu_);
    stop_ = true;
  }
  main_loop_done_.WaitForNotification();
  delete thread_;
  thread_ = nullptr;
  LOG(INFO) << this << " Record yielder exit";
  delete this;
}

Status RecordYielder::Yield(Rope* value) {
  MutexLock l(&mu_);
  WaitForBufEnough();
  if (status_.ok()) {
    CHECK(!stop_ && !buf_.empty());
    ExtractValue(value);
    ++num_records_yielded_in_epoch_;
  }
  return status_;
}

bool RecordYielder::ShouldFinish(const Status& s) {
  MutexLock l(&mu_);
  status_.Update(s);
  return stop_ || !status_.ok();
}


void RecordYielder::MainLoop() {
  while (true) {
    ++epoch_;
    num_records_yielded_in_epoch_ = 0;
    LOG(INFO) << "Epoch " << epoch_ << " " << opts_.file_pattern;

    // Finds all files.
    std::vector<string> filenames;
    Status s = MatchFiles(opts_.file_pattern, &filenames);
    if (ShouldFinish(s)) break;

    if (filenames.empty()) {
      s = errors::NotFound("Found no files at ", opts_.file_pattern);
      if (ShouldFinish(s)) break;
    }

    int shuffle_seed = opts_.seed;
    if (opts_.seed == 0) {
      MutexLock l(&mu_);
      shuffle_seed = rnd_();
    }

    // Shuffles these files according to the epoch # and random seed.
    std::mt19937_64 shuffle_rnd(Hash64Combine(epoch_, shuffle_seed));
    std::shuffle(filenames.begin(), filenames.end(), shuffle_rnd);

    // Shards files and use one thread to go through each shard.
    const int N = opts_.parallelism;
    std::vector<Shard> shards(N);
    for (int i = 0; i < N; ++i) {
      Shard* shard = &shards[i];
      shard->index = i;
      for (int j = i; j < filenames.size(); j += N) {
        shard->filenames.push_back(filenames[j]);
      }
      thread_->Schedule([this, shard]() { ShardLoop(shard); });
    }
    for (int i = 0; i < N; ++i) {
      shards[i].done.WaitForNotification();
      s.Update(shards[i].status);
    }
    if (ShouldFinish(s)) break;

    // Do not start the next epoch until all buffered records are consumed.
    {
      MutexLock l(&mu_);
      epoch_end_ = true;
      mu_.Await(buf_empty_);
      epoch_end_ = false;
    }

    LOG(INFO) << "Epoch " << epoch_ << ": total records "
              << num_records_yielded_in_epoch_;
  }
  main_loop_done_.Notify();
}

bool RecordYielder::Add(std::vector<Rope>* values) {
  MutexLock l(&mu_);
  mu_.Await(buf_not_full_);
  while (BufNotFull() && !values->empty()) {
    // Adds values->back(). Swaps its position with another random
    // element.
    auto index = rnd_() % (buf_.size() + 1);
    if (index == buf_.size()) {
      buf_.push_back(std::move(values->back()));
    } else {
      buf_.push_back(std::move(buf_[index]));
      buf_[index] = std::move(values->back());
    }
    values->pop_back();
  }
  return stop_;
}

Status RecordYielder::MatchFiles(const string& patterns,
                                 std::vector<string>* filenames) {
  for (const auto& file_pattern : str_util::Split(patterns, ',')) {
    std::vector<string> files_per_glob;
    TF_RETURN_IF_ERROR(
        Env::Default()->GetMatchingPaths(file_pattern, &files_per_glob));
    filenames->insert(filenames->end(), files_per_glob.begin(),
                      files_per_glob.end());
  }
  return Status::OK();
}

// Record yielder for plain text files.
class PlainTextYielder : public RecordYielder {
 public:
  explicit PlainTextYielder(const Options& opts) : RecordYielder(opts) {}

 protected:
  void ShardLoop(Shard* shard) override {
    std::vector<Rope> values;
    const int64 kRecords = 16;
    for (const string& filename : shard->filenames) {
      if (ShouldFinish(Status::OK())) break;
      VLOG(1) << "Shard " << shard->index << " " << filename;
      std::unique_ptr<RandomAccessFile> file;
      shard->status = Env::Default()->NewRandomAccessFile(filename, &file);
      if (!shard->status.ok()) {
        break;
      }
      std::unique_ptr<io::RandomAccessInputStream> input_stream(
          new io::RandomAccessInputStream(file.get()));
      io::BufferedInputStream in(input_stream.get(), 2 << 20);

      string line;
      while (true) {
        shard->status = in.ReadLine(&line);
        if (errors::IsOutOfRange(shard->status)) {
          shard->status = Status::OK();
          break;
        }
        values.emplace_back(line);
        if (values.size() >= kRecords && Add(&values)) {
          shard->status = errors::Aborted("stopped");
          break;
        }
      }
    }
    // Adds the remaining values of this shard to buf_.
    while (!values.empty()) {
      Add(&values);
    }
    shard->done.Notify();
  }
};

// Record yielder for tfrecord files.
class TFRecordYielder : public RecordYielder {
 public:
  explicit TFRecordYielder(const Options& opts) : RecordYielder(opts) {}

 protected:
  void ShardLoop(Shard* shard) override {
    std::vector<Rope> values;
    const int64 kRecords = 16;
    for (const string& filename : shard->filenames) {
      if (ShouldFinish(Status::OK())) break;
      VLOG(1) << "Shard " << shard->index << " " << filename;
      std::unique_ptr<RandomAccessFile> file;
      shard->status = Env::Default()->NewRandomAccessFile(filename, &file);
      if (!shard->status.ok()) {
        break;
      }
      io::RecordReaderOptions opts;
      opts.buffer_size = 2 << 20;
      io::SequentialRecordReader reader(file.get());
      string record;
      while (true) {
        shard->status = reader.ReadRecord(&record);
        if (errors::IsOutOfRange(shard->status)) {
          shard->status = Status::OK();
          break;
        }
        values.emplace_back(record);
        if (values.size() >= kRecords && Add(&values)) {
          shard->status = errors::Aborted("stopped");
          break;
        }
      }
    }
    // Adds the remaining values of this shard to buf_.
    while (!values.empty()) {
      Add(&values);
    }
    shard->done.Notify();
  }
};

namespace {

bool register_text_yielder =
    RecordYielder::Register("text", [](const RecordYielder::Options& opts) {
      return new PlainTextYielder(opts);
    });

bool register_tf_record_yielder =
    RecordYielder::Register("tfrecord", [](const RecordYielder::Options& opts) {
      return new TFRecordYielder(opts);
    });

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
