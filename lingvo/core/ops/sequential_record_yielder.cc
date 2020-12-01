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

#include "lingvo/core/ops/sequential_record_yielder.h"

#include <algorithm>

#include "lingvo/core/ops/record_yielder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lingvo {

namespace {
constexpr int kInfinite = -1;
}  // namespace

SequentialRecordYielder::SequentialRecordYielder(const string& file_pattern,
                                                 const int64_t repeat_count)
    : file_type_(RecordIterator::GetFilePatternPrefix(file_pattern)),
      repeat_count_(repeat_count) {
  LOG(INFO) << this << "Sequential record yielder start";
  string mutable_file_pattern(file_pattern);
  if (!file_type_.empty()) {
    mutable_file_pattern.erase(0, file_type_.size() + 1);
  }
  // Finds all files.
  RecordIterator::ParserOptions parser_options;
  TF_CHECK_OK(RecordIterator::ParsePattern(file_type_, mutable_file_pattern,
                                           parser_options, &filenames_));
  std::sort(filenames_.begin(), filenames_.end());
  if (filenames_.empty()) {
    LOG(FATAL) << "Found no files at " << file_pattern;
  }

  CHECK(repeat_count == kInfinite || repeat_count > 0)
      << "Repeat count must either be -1 (infinite) or a positive integer.";

  record_iterator_ = std::unique_ptr<RecordIterator>(
      RecordIterator::New(file_type_, filenames_[0]));
}

SequentialRecordYielder* SequentialRecordYielder::New(
    const string& file_pattern, const int64_t repeat_count) {
  return new SequentialRecordYielder(file_pattern, repeat_count);
}

SequentialRecordYielder::~SequentialRecordYielder() {}

void SequentialRecordYielder::Close() {
  LOG(INFO) << this << "Sequential record yielder exit";
  delete this;
}

Status SequentialRecordYielder::Yield(Record* record) {
  string key;
  if (record_iterator_->Next(&key, &record->value)) {
    return Status::OK();
  }

  // No more records from current iterator, advance to next iterator.
  cur_file_index_ = (cur_file_index_ + 1) % filenames_.size();
  if (cur_file_index_ == 0) {
    ++num_repeats_;
    LOG(INFO) << "SequentialRecordYielder finished " << num_repeats_
              << " repeats.";
    if (repeat_count_ != kInfinite && num_repeats_ == repeat_count_) {
      return errors::OutOfRange("SequentialRecordYielder reached ",
                                repeat_count_, " repeats.");
    }
  }
  record_iterator_ = std::unique_ptr<RecordIterator>(
      RecordIterator::New(file_type_, filenames_[cur_file_index_]));
  return Yield(record);
}

}  // namespace lingvo
}  // namespace tensorflow
