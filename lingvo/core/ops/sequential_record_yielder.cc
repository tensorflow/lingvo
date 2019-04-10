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

#include <algorithm>

#include "lingvo/core/ops/record_yielder.h"
#include "lingvo/core/ops/sequential_record_yielder.h"

namespace tensorflow {
namespace lingvo {

SequentialRecordYielder::SequentialRecordYielder(const string& file_pattern)
    : file_type_(RecordIterator::GetFilePatternPrefix(file_pattern)) {
  LOG(INFO) << this << "Sequential record yielder start";
  string mutable_file_pattern(file_pattern);
  if (!file_type_.empty()) {
    mutable_file_pattern.erase(0, file_type_.size() + 1);
  }
  // Finds all files.
  TF_CHECK_OK(RecordIterator::ParsePattern(file_type_, mutable_file_pattern,
                                           &filenames_));
  std::sort(filenames_.begin(), filenames_.end());
  if (filenames_.empty()) {
    LOG(FATAL) << "Found no files at " << file_pattern;
  }

  record_iterator_ = std::unique_ptr<RecordIterator>(
      RecordIterator::New(file_type_, filenames_[0]));
}

SequentialRecordYielder* SequentialRecordYielder::New(
    const string& file_pattern) {
  return new SequentialRecordYielder(file_pattern);
}

SequentialRecordYielder::~SequentialRecordYielder() {}

void SequentialRecordYielder::Close() {
  LOG(INFO) << this << "Sequential record yielder exit";
  delete this;
}

Status SequentialRecordYielder::Yield(Rope* value, int* source_id) {
  string key;
  if (record_iterator_->Next(&key, value)) {
    return Status::OK();
  } else {
    // No more records from current iterator, advance to next iterator.
    cur_file_index_ = (cur_file_index_ + 1) % filenames_.size();
    record_iterator_ = std::unique_ptr<RecordIterator>(
        RecordIterator::New(file_type_, filenames_[cur_file_index_]));
    return Yield(value, source_id);
  }
}

}  // namespace lingvo
}  // namespace tensorflow
