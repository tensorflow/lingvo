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

#include "lingvo/core/ops/record_yielder.h"

#include <chrono>  // NOLINT(build/c++11)
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <unordered_map>

#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "lingvo/core/ops/versioned_file_set.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace lingvo {

namespace {

// Number of records to batch for a single call to Add.
const int kRecordsPerAdd = 16;

struct Factory {
  absl::Mutex mu;
  std::unordered_map<string, RecordIterator::FactoryMethod> creators;
  std::unordered_map<string, RecordIterator::PatternParserMethod>
      pattern_parsers;
};

Factory* GetFactory() {
  static Factory* factory = new Factory;
  return factory;
}

// A sharded file pattern looks like /path/name@100.sst, which is expanded to
// a glob pattern /path/name-?????-of-00100.sst by this function. The .sst file
// extension is optional. The number of shards shouldn't exceed 5 digits. A
// wildcard is also support for the shards number: /path/name@*.sst is expanded
// to /path/name-?????-of-?????.sst.
Status MaybeExpandShardedFilePattern(StringPiece file_pattern,
                                     string* expanded) {
  for (int i = file_pattern.size() - 2; i >= 0; --i) {
    const char c = file_pattern[i];
    if (c == '/') {
      // This is not a valid sharded file pattern, do not expand it.
      *expanded = string(file_pattern);
      return Status::OK();
    }
    if (c == '@') {
      if (file_pattern[i + 1] == '*') {
        *expanded = strings::StrCat(file_pattern.substr(0, i),
                                    "-\?\?\?\?\?-of-\?\?\?\?\?",
                                    file_pattern.substr(i + 2));
        return Status::OK();
      } else {
        uint64 num_shards = 0;
        StringPiece suf = file_pattern.substr(i + 1);
        if (str_util::ConsumeLeadingDigits(&suf, &num_shards)) {
          if (num_shards <= 0 || num_shards >= 99999) {
            return errors::InvalidArgument(
                strings::StrCat("The number of shards should be non-zero and "
                                "not exceed 5 digits: ",
                                num_shards));
          }
          const string shard_pattern = strings::Printf(
              "-\?\?\?\?\?-of-%05d", static_cast<int>(num_shards));
          *expanded =
              strings::StrCat(file_pattern.substr(0, i), shard_pattern, suf);
          return Status::OK();
        }
      }
      break;
    }
  }
  *expanded = string(file_pattern);
  return Status::OK();
}

// ParallelFilePatterns look like this
//  <path1>/a-*-of-10;<path2>/b-*-of-10,<path3>/c-*-of-10;<path4>/d-*-of-10
// Each "," separated pattern is a parallel file pattern.
// In a given parallel file pattern, all the ";" separated file patterns
// should have the same number of shards. This method creates ";" separated
// filenames aligning the shards. e,g. <path1>/a-01-of-10;<path2>/b-01-of-10
// <path1>/a-02-of-10;<path2>/b-02-of-10
// <path3>/c-01-of-10;<path4>/d-01-of-10
// <path3>/c-02-of-10;<path4>/d-02-of-10
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
    if (filenames_per_pattern.empty()) {
      return tensorflow::errors::InvalidArgument(
          "Found no files at ", expanded_file_pattern);
    }
    if (parallel_filenames.empty()) {
      parallel_filenames.swap(filenames_per_pattern);
      continue;
    }
    if (parallel_filenames.size() != filenames_per_pattern.size()) {
      return tensorflow::errors::InvalidArgument(
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

// Input to CkptFilePattern should look like <path_to_ckpt_file>/ckpt
// This pattern parser will read the ckpt file, decode it into a
// tensorflow::CheckpointState proto and use the latest model_checkpoint_path
// as the actual file-pattern.
// model_checkpoint_path should be a legitimate file-pattern, for eg.
// <path>/a@* or <path>/a or <path>/a-?????-of-????? etc.
Status GetFilePatternsFromCkptFile(const string& fileset_file,
                                   std::vector<string>* filenames) {
  string fileset_str;
  VersionedFileSet fileset;
  TF_RETURN_IF_ERROR(
      ReadFileToString(Env::Default(), fileset_file, &fileset_str));

  CHECK(google::protobuf::TextFormat::ParseFromString(fileset_str, &fileset));

  string prefix = std::string(io::Dirname(fileset_file));
  for (const auto& file_pattern : fileset.current().file_pattern()) {
    TF_RETURN_IF_ERROR(MatchParallelFilePattern(
      io::JoinPath(prefix, file_pattern), filenames));
  }
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
  absl::MutexLock l(&factory->mu);
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
    absl::MutexLock l(&factory->mu);
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

string RecordIterator::StripPrefixFromFilePattern(string* file_pattern) {
  const string prefix = RecordIterator::GetFilePatternPrefix(*file_pattern);
  if (!prefix.empty()) file_pattern->erase(0, prefix.size() + 1);
  return prefix;
}

bool HasFilePattern(StringPiece str) {
  return absl::StrContains(str, "*") || absl::StrContains(str, "?") ||
         absl::StrContains(str, "@") || absl::StrContains(str, "[") ||
         absl::StrContains(str, "]");
}

Status RecordIterator::ParsePattern(
    const string& type_name, const string& file_pattern_list,
    const RecordIterator::ParserOptions& options,
    std::vector<string>* filenames) {
  Factory* factory = GetFactory();
  RecordIterator::PatternParserMethod parser_method;
  {
    absl::MutexLock l(&factory->mu);
    const auto iter = factory->pattern_parsers.find(type_name);
    if (iter != factory->pattern_parsers.end()) {
      parser_method = iter->second;
    }
  }
  if (parser_method) {
    return parser_method(file_pattern_list, options, filenames);
  }
  for (const auto& file_pattern : str_util::Split(file_pattern_list, ',')) {
    if (HasFilePattern(file_pattern)) {
      std::vector<string> files_per_glob;
      TF_RETURN_IF_ERROR(
          MatchParallelFilePattern(file_pattern, &files_per_glob));
      filenames->insert(filenames->end(), files_per_glob.begin(),
                        files_per_glob.end());
    } else {
      filenames->push_back(file_pattern);
    }
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
    *key = strings::Printf("%08lld", static_cast<long long>(num_++));
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
    *key = strings::Printf("%08lld", static_cast<long long>(num_++));
    *value = Rope(record_);
    return true;
  }

 private:
  std::unique_ptr<RandomAccessFile> file_;
  io::SequentialRecordReader reader_;
  int64 num_ = 0;
  tstring record_;

  io::RecordReaderOptions ReaderOptions(const string& compression_type) {
    auto opts =
        io::RecordReaderOptions::CreateRecordReaderOptions(compression_type);
    opts.buffer_size = 2LL << 20;  // 2MB.
    return opts;
  }
};

// An iterator generates [0 .. max_).
class IotaIterator : public RecordIterator {
 public:
  IotaIterator(const string& filename) {
    if (filename.empty() || !strings::safe_strto64(filename, &max_)) {
      max_ = kint64max;
    }
  }

  bool Next(string* key, Rope* value) override {
    if (num_ >= max_) {
      return false;
    }
    *key = strings::Printf("%010lld", static_cast<long long>(num_++));
    *value = *key;
    return true;
  }

 private:
  int64 max_ = kint64max;
  int64 num_ = 0;
};

namespace {

bool register_text_iterator = RecordIterator::Register(
    "text",
    [](const string& filename) { return new PlainTextIterator(filename); });

// Iterator for plain text files that checks indirect "ckpt" file for latest
// pointer to data files.
// For example, if filename is .../checkpoint, the pattern parser will open the
// checkpoint file and read file-patterns that it points to. If empty, it will
// return error and die.
bool register_indirect_text_iterator =
    RecordIterator::RegisterWithPatternParser(
        "text_indirect",
        [](const string& filename) { return new PlainTextIterator(filename); },
        [](const string& pattern, const RecordIterator::ParserOptions& options,
           std::vector<string>* outs) {
          TF_RETURN_IF_ERROR(GetFilePatternsFromCkptFile(pattern, outs));
          return Status::OK();
        });

bool register_tf_record_iterator =
    RecordIterator::Register("tfrecord", [](const string& filename) {
      return new TFRecordIterator(filename, io::compression::kNone);
    });

bool register_tf_record_gzip_iterator =
    RecordIterator::Register("tfrecord_gzip", [](const string& filename) {
      return new TFRecordIterator(filename, io::compression::kGzip);
    });

bool register_iota_iterator = RecordIterator::RegisterWithPatternParser(
    "iota", [](const string& filename) { return new IotaIterator(filename); },
    [](const string& pattern, const RecordIterator::ParserOptions& options,
       std::vector<string>* outs) {
      // The pattern is just a stringified number.
      *outs = {pattern};
      return Status::OK();
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
                                     "record_yielder", 2 + opts.parallelism,
                                     /* low_latency_hint */ false)),
      epoch_(1),
      rnd_(opts.seed),
      yields_(0),
      buf_empty_(this, &ME::BufEmpty),
      buf_not_full_(this, &ME::BufNotFull),
      buf_enough_(this, &ME::BufEnough) {
  LOG(INFO) << this << " Record yielder start";
  if (opts_.seed == 0) {
    LOG(INFO) << "Randomly seed RecordYielder.";
    rnd_.seed(std::random_device{}());
  }
  file_type_ = RecordIterator::StripPrefixFromFilePattern(&opts_.file_pattern);
  if (opts_.bufsize_in_seconds > 0) {
    bufsize_ = kRecordsPerAdd * opts_.parallelism;
  } else {
    bufsize_ = opts_.bufsize;
  }
}

BasicRecordYielder::BasicRecordYielder()
    : buf_empty_(this, &ME::BufEmpty),
      buf_not_full_(this, &ME::BufNotFull),
      buf_enough_(this, &ME::BufEnough) {}  // USED ONLY FOR TESTS.
BasicRecordYielder::~BasicRecordYielder() {}

void BasicRecordYielder::Start() {
  thread_->Schedule([this]() { MainLoop(); });
}

void BasicRecordYielder::Close() {
  {
    absl::MutexLock l(&mu_);
    stop_ = true;
  }
  main_loop_done_.WaitForNotification();
  delete thread_;
  thread_ = nullptr;
  LOG(INFO) << this << "Basic record yielder exit";
  delete this;
}

Status BasicRecordYielder::Yield(Record* record) {
  absl::MutexLock l(&mu_);
  WaitForBufEnough();
  ++yields_;

  if (status_.ok()) {
    CHECK(!stop_ && !buf_.empty());
    ExtractValue(&record->value);
    if (epoch_end_ && buf_.empty()) {
      ++epoch_;
    }
    record->source_id = static_cast<int>(opts_.source_id);
    ++num_records_yielded_in_epoch_;
  }
  return status_;
}

bool BasicRecordYielder::ShouldFinish(const Status& s) {
  absl::MutexLock l(&mu_);
  status_.Update(s);
  return stop_ || !status_.ok();
}

void BasicRecordYielder::AdjustBufferSizeLoop() {
  if (opts_.bufsize_in_seconds == 0) {
    // Nothing to do for a fixed buffer size.
    return;
  }

  // Wake up each second and adjust the buffer size.
  while (true) {
    {
      absl::MutexLock l(&mu_);

      // Quit if requested.
      if (stop_) break;

      // Smoothed bufsize_ estimate based on current yields_ requests.
      // With this set of parameters, the contribution of the current buffer
      // size decays as follows:
      //    10 seconds -> 90% (.99^10)
      //      1 minute -> 50% (.99^60)
      //     5 minutes ->  5% (.99^300)
      bufsize_ = 0.99 * static_cast<double>(bufsize_) +
                 0.01 * yields_ * opts_.bufsize_in_seconds;

      // Make sure the buffer is large enough to hold one batch of Add records
      // per thread.
      bufsize_ = std::max<int64>(opts_.parallelism * kRecordsPerAdd, bufsize_);

      // Make sure the buffer is not larger than the bufsize parameter in the
      // options.
      if (opts_.bufsize > 0) {
        bufsize_ = std::min<double>(opts_.bufsize, bufsize_);
      }
      VLOG(1) << "Yields:" << yields_ << " Bufsize:" << bufsize_
              << " Pattern:" << opts_.file_pattern;

      // Reset yields_ to zero to count another second of requests.
      yields_ = 0;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

void BasicRecordYielder::MainLoop() {
  absl::Notification adjust_done;
  thread_->Schedule([this, &adjust_done]() {
    AdjustBufferSizeLoop();
    adjust_done.Notify();
  });

  while (true) {
    num_records_yielded_in_epoch_ = 0;
    const int64 epoch = current_epoch();
    LOG(INFO) << "Epoch " << epoch << " " << opts_.file_pattern;

    // Finds all files.
    std::vector<string> filenames;
    RecordIterator::ParserOptions parser_options;
    parser_options.executor = thread_;
    parser_options.epoch = epoch;
    parser_options.num_input_replicas = opts_.num_input_replicas;
    parser_options.input_replica_id = opts_.input_replica_id;
    const int num_threads = thread_->NumThreads();
    // We want each shard to fit into the shuffle buffer. Since there can be
    // up to `num_threads` concurrent ShardLoop calls, we divide the shuffle
    // buffer by num_threads.
    parser_options.max_records_per_shard = static_cast<int64>(
        bufsize() * .9 / num_threads);
    // Ideally there should be enough number of shards to keep the thread pool
    // busy.
    parser_options.min_shards = opts_.parallelism * 4;
    Status s = RecordIterator::ParsePattern(file_type_, opts_.file_pattern,
                                            parser_options, &filenames);
    if (ShouldFinish(s)) break;

    if (filenames.empty()) {
      LOG(FATAL) << "Found no files at " << opts_.file_pattern;
    }

    int shuffle_seed = opts_.seed;
    if (opts_.seed == 0) {
      absl::MutexLock l(&mu_);
      shuffle_seed = rnd_();
    }

    // Shuffles these files according to the epoch # and random seed.
    std::mt19937_64 shuffle_rnd(Hash64Combine(current_epoch(), shuffle_seed));
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
      absl::MutexLock l(&mu_);
      epoch_end_ = true;
      mu_.Await(buf_empty_);
      epoch_end_ = false;
    }

    LOG(INFO) << "Epoch " << current_epoch() << ": total records "
              << num_records_yielded_in_epoch_;
  }

  adjust_done.WaitForNotification();
  main_loop_done_.Notify();
}

bool BasicRecordYielder::Add(std::vector<Rope>* values) {
  absl::MutexLock l(&mu_);
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
  for (const string& filename : shard->filenames) {
    if (ShouldFinish(Status::OK())) break;
    VLOG(1) << "Shard " << shard->index << " " << filename;
    std::unique_ptr<RecordIterator> iter(
        RecordIterator::New(file_type_, filename));
    int64 shard_record_count = 0;
    string key;
    Rope val;
    while (iter->Next(&key, &val)) {
      ++shard_record_count;
      LOG_EVERY_N_SEC(INFO, 10) << "Record " << shard_record_count
                                << ": key=" << absl::CEscape(key);
      values.emplace_back(val);
      if (values.size() >= kRecordsPerAdd && Add(&values)) {
        LOG(INFO) << "Shard stopped at " << shard_record_count
                  << "records from " << absl::CEscape(filename);
        shard->status = errors::Aborted("stopped");
        break;
      }
    }
    LOG(INFO) << "Emitted " << shard_record_count << " records from "
              << absl::CEscape(filename);
  }
  // Adds the remaining values of this shard to buf_.
  while (!values.empty()) {
    Add(&values);
  }
  shard->done.Notify();
}

}  // namespace lingvo
}  // namespace tensorflow
