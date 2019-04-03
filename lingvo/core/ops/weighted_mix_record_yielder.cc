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

#include "lingvo/core/ops/weighted_mix_record_yielder.h"

namespace tensorflow {
namespace lingvo {

WeightedMixRecordYielder::WeightedMixRecordYielder(
    const int64 seed, const std::vector<RecordYielder*>& yielders,
    const std::vector<float>& input_source_weights)
    : rnd_(seed),
      sample_distribution_(input_source_weights.begin(),
                           input_source_weights.end()),
      yielders_(yielders) {
  if (yielders.size() != input_source_weights.size()) {
    LOG(FATAL) << "Unable to create WeightedMixRecordYielder: every yielder "
               << "should have a corresponding weight. " << yielders.size()
               << " yielders and " << input_source_weights.size()
               << " weights were "
               << "provided.";
  }
  if (yielders.empty()) {
    LOG(FATAL) << "There should be at least one yielder provided.";
  }

  for (float x : input_source_weights) {
    if (x < 0) {
      LOG(FATAL) << "All weights should be greater or equal to zero. Got " << x;
    }
  }
}

WeightedMixRecordYielder* WeightedMixRecordYielder::New(
    const int64 seed, const std::vector<RecordYielder*>& yielders,
    const std::vector<float>& input_source_weights) {
  WeightedMixRecordYielder* yielder =
      new WeightedMixRecordYielder(seed, yielders, input_source_weights);
  return yielder;
}

WeightedMixRecordYielder::~WeightedMixRecordYielder() {}

void WeightedMixRecordYielder::Close() {
  for (RecordYielder* yielder : yielders_) {
    yielder->Close();
  }
  LOG(INFO) << this << "Weighted mix record yielder exit";
  delete this;
}

Status WeightedMixRecordYielder::Yield(Rope* value, int* source_id) {
  size_t yielder_idx = 0;
  {
    MutexLock l(&mu_);
    yielder_idx = sample_distribution_(rnd_);
    // Release the lock immediately once we fix which yielder to use.
  }
  while (true) {
    // Retry indefinitely until we get an Ok status from the specific yielder.
    // This will stall the training if there is any unrecoverable error with
    // the child yielder.
    Status s = yielders_.at(yielder_idx)->Yield(value, source_id);
    if (!s.ok()) {
      LOG(WARNING) << s;
      continue;
    }
    return s;
  }
}

}  // namespace lingvo
}  // namespace tensorflow
