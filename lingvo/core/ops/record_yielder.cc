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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/compression.h"
#include "lingvo/core/ops/record_yielder.h"

#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "lingvo/core/ops/mutex.h"

namespace tensorflow {
namespace lingvo {

namespace {

struct Factory {
  Mutex mu;
  std::unordered_map<string, RecordIterator::FactoryMethod> creators;
  std::unordered_map<string, RecordIterator::PatternParserMethod>
      pattern_parsers;
};

Factory* GetFactory() {
  static Factory* factory = new Factory;
  return factory;
}

// A sharded file pattern looks like /path/name@100, which is expanded to
// a glob pattern /path/name-?????-of-00100 by this function. The number of
// shards shouldn't exceed 5 digits.
Status MaybeExpandShardedFilePattern(const string& file_pattern,
                                     string* expanded) {
  const auto pos = file_pattern.find('@');
  if (pos == string::npos) {  // not a sharded file pattern
    *expanded = file_pattern;
    return Status::OK();
  }

  const string prefix = file_pattern.substr(0, pos);
  const string suffix = file_pattern.substr(pos + 1);

  uint32 num_shards = 0;
  if (!strings::safe_strtou32(suffix, &num_shards)) {
    return errors::InvalidArgument(
        strings::StrCat("Invalid sharded file pattern: ", file_pattern));
  }
  if (num_shards > 99999) {
    return errors::InvalidArgument(strings::StrCat(
        "The number of shards should not exceed 5 digits: ", num_shards));
  }

  *expanded =
      strings::Printf("%s-\?\?\?\?\?-of-%05d", prefix.c_str(), num_shards);
  return Status::OK();
}

// ParallelFilePatterns look like this
//  <path1>/a-*-of-10;<path2>/b-*-of-10,<path3>/c-*-of-10;<path4>/d-*-of-10
// Each "," separated pattern is a parallel file pattern.
// In a given parallel file pattern, all the ";" separated file patterns
// should have the same number of shards. This method creates ";" separated
// filenames aligning the shards. e,g. <path1>/a-01-of-10,<path2>/b-01-of-10
// <path1>/a-02-of-10,<path2>/b-02-of-10
// <path3>/c-01-of-10,<path2>/d-01-of-10
// <path3>/c-02-of-10,<path2>/d-02-of-10
Status MatchParallelFilePattern(const string& parallel_file_pattern,
                                std::vector<string>* filenames) {
  std::vector<string> parallel_filenames;
  for (const auto& file_pattern : str_util::Split(parallel_file_pattern, ';')) {
    string expanded_file_pattern;
    TF_RETURN_IF_ERROR(
        MaybeExpandShardedFilePattern(file_pattern, &expanded_file_pattern));
    std::vector<string> filenames_per_pattern;
    TF_RETURN_IF_ERROR(Env::Default()->GetMatchingPaths(
        expanded_file_pattern, &filenames_per_pattern));
    if (parallel_filenames.empty()) {
      parallel_filenames.swap(filenames_per_pattern);
      continue;
    }
    if (parallel_filenames.size() != filenames_per_pattern.size()) {
      return Status(tensorflow::error::INVALID_ARGUMENT,
                    "All file patterns in the parallel file pattern do not "
                    "have the same number of elements.");
    }
    for (auto it1 = parallel_filenames.begin(),
              it2 = filenames_per_pattern.begin();
         it1 != parallel_filenames.end(); ++it1, ++it2) {
      strings::StrAppend(&(*it1), ";", *it2);
    }
  }
  filenames->insert(filenames->end(),
                    std::make_move_iterator(parallel_filenames.begin()),
                    std::make_move_iterator(parallel_filenames.end()));

  return Status::OK();
}

}  // end namespace

bool RecordIterator::Register(const string& type_name, FactoryMethod method) {
  return RegisterWithPatternParser(type_name, std::move(method),
                                   RecordIterator::PatternParserMethod());
}

bool RecordIterator::RegisterWithPatternParser(
    const string& type_name, FactoryMethod method,
    PatternParserMethod parser_method) {
  Factory* factory = GetFactory();
  MutexLock l(&factory->mu);
  bool ret = factory->creators.insert({type_name, std::move(method)}).second;
  CHECK(ret) << "Possibly duplicated registration: " << type_name;
  if (parser_method) {
    factory->pattern_parsers.insert({type_name, std::move(parser_method)});
  }
  return ret;
}

RecordIterator* RecordIterator::New(const string& type_name,
                                    const string& filename) {
  Factory* factory = GetFactory();
  RecordIterator::FactoryMethod method;
  {
    MutexLock l(&factory->mu);
    const auto iter = factory->creators.find(type_name);
    CHECK(iter != factory->creators.end())
        << "Unable to create RecordIterator for format \"" << type_name << "\"";
    method = iter->second;
  }
  return method(filename);
}

// Returns the prefix in a file pattern, or an empty string if not exist.
// Example: "tfrecord:data_dir/data.tfrecord" => "tfrecord"
string RecordIterator::GetFilePatternPrefix(const string& file_pattern) {
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

Status RecordIterator::ParsePattern(const string& type_name,
                                    const string& file_pattern_list,
                                    std::vector<string>* filenames) {
  Factory* factory = GetFactory();
  RecordIterator::PatternParserMethod parser_method;
  {
    MutexLock l(&factory->mu);
    const auto iter = factory->pattern_parsers.find(type_name);
    if (iter != factory->pattern_parsers.end()) {
      parser_method = iter->second;
    }
  }
  if (parser_method) {
    return parser_method(file_pattern_list, filenames);
  }
  for (const auto& file_pattern : str_util::Split(file_pattern_list, ',')) {
    std::vector<string> files_per_glob;
    TF_RETURN_IF_ERROR(
        MatchParallelFilePattern(file_pattern, &files_per_glob));
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

class TFRecordIterator : public RecordIterator {
 public:
  TFRecordIterator(const string& filename, const string& compression_type)
      : file_(OpenOrDie(filename)),
        reader_(file_.get(), ReaderOptions(compression_type)) {}

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

  io::RecordReaderOptions ReaderOptions(const string& compression_type) {
    auto opts =
        io::RecordReaderOptions::CreateRecordReaderOptions(compression_type);
    opts.buffer_size = 2LL << 20;  // 2MB.
    return opts;
  }
};

namespace {

bool register_text_iterator = RecordIterator::Register(
    "text",
    [](const string& filename) { return new PlainTextIterator(filename); });

bool register_tf_record_iterator =
    RecordIterator::Register("tfrecord", [](const string& filename) {
      return new TFRecordIterator(filename, io::compression::kNone);
    });

bool register_tf_record_gzip_iterator =
    RecordIterator::Register("tfrecord_gzip", [](const string& filename) {
      return new TFRecordIterator(filename, io::compression::kGzip);
    });

}  // namespace

RecordYielder::~RecordYielder() {}

BasicRecordYielder* BasicRecordYielder::New(Options opts) {
  auto yielder = new BasicRecordYielder(opts);
  yielder->Start();
  return yielder;
}

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
  if (opts_.seed == 0) {
    LOG(INFO) << "Randomly seed RecordYielder.";
    rnd_.seed(std::random_device{}());
  }
  file_type_ = RecordIterator::GetFilePatternPrefix(opts_.file_pattern);
  if (!file_type_.empty()) {
    opts_.file_pattern.erase(0, file_type_.size() + 1);
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

Status BasicRecordYielder::Yield(Rope* value, int* source_id) {
  MutexLock l(&mu_);
  WaitForBufEnough();
  if (status_.ok()) {
    CHECK(!stop_ && !buf_.empty());
    ExtractValue(value);
    if (source_id) {
      *source_id = static_cast<int>(opts_.source_id);
    }
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
    Status s = RecordIterator::ParsePattern(file_type_, opts_.file_pattern,
                                            &filenames);
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


void BasicRecordYielder::ShardLoop(Shard* shard) {
  std::vector<Rope> values;
  const int64 kRecords = 16;
  for (const string& filename : shard->filenames) {
    if (ShouldFinish(Status::OK())) break;
    VLOG(1) << "Shard " << shard->index << " " << filename;
    std::unique_ptr<RecordIterator> iter(
        RecordIterator::New(file_type_, filename));
    string key;
    Rope val;
    while (iter->Next(&key, &val)) {
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

}  // namespace lingvo
}  // namespace tensorflow
