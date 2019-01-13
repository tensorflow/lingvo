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
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "lingvo/core/ops/mutex.h"

namespace tensorflow {
namespace lingvo {

namespace {

struct Factory {
  Mutex mu;
  std::unordered_map<string, BasicRecordYielder::FactoryMethod> methods;
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

bool BasicRecordYielder::Register(const string& type_name,
                                  FactoryMethod method) {
  Factory* factory = GetFactory();
  MutexLock l(&factory->mu);
  bool ret = factory->methods.insert({type_name, std::move(method)}).second;
  CHECK(ret) << "Possibly duplicated registration: " << type_name;
  return ret;
}

BasicRecordYielder* BasicRecordYielder::New(Options opts) {
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
  const BasicRecordYielder::FactoryMethod& method = iter->second;
  BasicRecordYielder* yielder = method(opts);
  yielder->Start();
  return yielder;
}

RecordYielder::~RecordYielder() {}

BasicRecordYielder::BasicRecordYielder(const Options& opts)
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

BasicRecordYielder::~BasicRecordYielder() {}

void BasicRecordYielder::Start() {
  thread_->Schedule([this]() { MainLoop(); });
}

void BasicRecordYielder::Close() {
  {
    MutexLock l(&mu_);
    stop_ = true;
  }
  main_loop_done_.WaitForNotification();
  delete thread_;
  thread_ = nullptr;
  LOG(INFO) << this << "Basic record yielder exit";
  delete this;
}

Status BasicRecordYielder::Yield(Rope* value) {
  MutexLock l(&mu_);
  WaitForBufEnough();
  if (status_.ok()) {
    CHECK(!stop_ && !buf_.empty());
    ExtractValue(value);
    ++num_records_yielded_in_epoch_;
  }
  return status_;
}

bool BasicRecordYielder::ShouldFinish(const Status& s) {
  MutexLock l(&mu_);
  status_.Update(s);
  return stop_ || !status_.ok();
}

void BasicRecordYielder::MainLoop() {
  while (true) {
    ++epoch_;
    num_records_yielded_in_epoch_ = 0;
    LOG(INFO) << "Epoch " << epoch_ << " " << opts_.file_pattern;

    // Finds all files.
    std::vector<string> filenames;
    Status s = MatchFiles(opts_.file_pattern, &filenames);
    if (ShouldFinish(s)) break;

    if (filenames.empty()) {
      LOG(FATAL) << "Found no files at " << opts_.file_pattern;
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

bool BasicRecordYielder::Add(std::vector<Rope>* values) {
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

Status BasicRecordYielder::MatchFiles(const string& patterns,
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

RandomAccessFile* OpenOrDie(const string& filename) {
  std::unique_ptr<RandomAccessFile> file;
  TF_CHECK_OK(Env::Default()->NewRandomAccessFile(filename, &file));
  return file.release();
}

class PlainTextIterator : public RecordIterator {
 public:
  PlainTextIterator(const string& filename)
      : file_(OpenOrDie(filename)),
        stream_(file_.get()),
        buf_(&stream_, 2 << 20) {}

  bool Next(string* key, Rope* value) override {
    Status s = buf_.ReadLine(&line_);
    if (errors::IsOutOfRange(s)) return false;
    TF_CHECK_OK(s);
    *key = strings::Printf("%08lld", num_++);
    *value = line_;
    return true;
  }

 private:
  std::unique_ptr<RandomAccessFile> file_;
  io::RandomAccessInputStream stream_;
  io::BufferedInputStream buf_;
  int64 num_ = 0;
  string line_;
};

// Record yielder for plain text files.
class PlainTextYielder : public BasicRecordYielder {
 public:
  explicit PlainTextYielder(const Options& opts) : BasicRecordYielder(opts) {}

 protected:
  void ShardLoop(Shard* shard) override {
    std::vector<Rope> values;
    const int64 kRecords = 16;
    for (const string& filename : shard->filenames) {
      if (ShouldFinish(Status::OK())) break;
      VLOG(1) << "Shard " << shard->index << " " << filename;
      PlainTextIterator iter(filename);
      string key;
      Rope val;
      while (iter.Next(&key, &val)) {
        values.emplace_back(val);
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

class TFRecordIterator : public RecordIterator {
 public:
  TFRecordIterator(const string& filename)
      : file_(OpenOrDie(filename)), reader_(file_.get(), ReaderOptions()) {}

  bool Next(string* key, Rope* value) override {
    Status s = reader_.ReadRecord(&record_);
    if (errors::IsOutOfRange(s)) return false;
    *key = strings::Printf("%08lld", num_++);
    *value = record_;
    return true;
  }

 private:
  std::unique_ptr<RandomAccessFile> file_;
  io::SequentialRecordReader reader_;
  int64 num_ = 0;
  string record_;

  io::RecordReaderOptions ReaderOptions() {
    io::RecordReaderOptions opts;
    opts.buffer_size = 2LL << 20;  // 2MB.
    return opts;
  }
};

// Record yielder for tfrecord files.
class TFRecordYielder : public BasicRecordYielder {
 public:
  explicit TFRecordYielder(const Options& opts) : BasicRecordYielder(opts) {}

 protected:
  void ShardLoop(Shard* shard) override {
    std::vector<Rope> values;
    const int64 kRecords = 16;
    for (const string& filename : shard->filenames) {
      if (ShouldFinish(Status::OK())) break;
      VLOG(1) << "Shard " << shard->index << " " << filename;
      TFRecordIterator iter(filename);
      string key;
      Rope val;
      while (iter.Next(&key, &val)) {
        values.emplace_back(val);
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

bool register_text_yielder = BasicRecordYielder::Register(
    "text", [](const BasicRecordYielder::Options& opts) {
      return new PlainTextYielder(opts);
    });

bool register_tf_record_yielder = BasicRecordYielder::Register(
    "tfrecord", [](const BasicRecordYielder::Options& opts) {
      return new TFRecordYielder(opts);
    });

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
