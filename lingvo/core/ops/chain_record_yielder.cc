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

#include "lingvo/core/ops/chain_record_yielder.h"

namespace tensorflow {
namespace lingvo {

ChainRecordYielder::ChainRecordYielder(
    const std::vector<BasicRecordYielder::Options>& yielder_options)
    : current_yielder_idx_(0),
      current_yielder_epoch_(1),
      current_yielder_(nullptr),
      yielder_options_(yielder_options) {
  if (yielder_options.empty()) {
    LOG(FATAL) << "There should be at least one set of options provided.";
  }
  current_yielder_ = BasicRecordYielder::New(yielder_options_.front());
}

ChainRecordYielder* ChainRecordYielder::New(
    const std::vector<BasicRecordYielder::Options>& yielder_options) {
  return new ChainRecordYielder(yielder_options);
}

ChainRecordYielder::~ChainRecordYielder() {}

void ChainRecordYielder::Close() {
  {
    MutexLock l(&mu_);
    if (current_yielder_) {
      current_yielder_->Close();
    }
  }
  LOG(INFO) << this << "Chain record yielder exit";
  delete this;
}

Status ChainRecordYielder::Yield(Rope* value, int* source_id) {
  MutexLock l(&mu_);

  int epoch = current_yielder_->current_epoch();
  if (epoch > current_yielder_epoch_) {
    int new_idx = (current_yielder_idx_ + 1) % yielder_options_.size();
    if (current_yielder_) {
      current_yielder_->Close();
    }
    current_yielder_idx_ = new_idx;
    current_yielder_ = BasicRecordYielder::New(yielder_options_.at(new_idx));
    current_yielder_epoch_ = 1;
  }
  while (true) {
    // Retry indefinitely until we get an Ok status from the specific yielder.
    // This will stall the training if there is any unrecoverable error with
    // the child yielder.
    Status s = current_yielder_->Yield(value, source_id);
    if (!s.ok()) {
      LOG(WARNING) << s;
      continue;
    }
    return s;
  }
}

}  // namespace lingvo
}  // namespace tensorflow
