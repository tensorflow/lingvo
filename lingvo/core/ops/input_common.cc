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

#include "lingvo/core/ops/input_common.h"
#include "lingvo/core/ops/sequential_record_yielder.h"
#include "lingvo/core/ops/weighted_mix_record_yielder.h"

namespace tensorflow {
namespace lingvo {

RecordYielder* ConstructYielder(const string& file_pattern,
                                const std::vector<float>& input_source_weights,
                                int64 file_random_seed, int64 file_buffer_size,
                                int64 file_parallelism,
                                bool require_sequential_order) {
  std::vector<string> file_patterns;
  if (input_source_weights.empty()) {
    LOG(INFO) << "Input source weights are empty, fall back to legacy "
              << "behavior.";
    file_patterns.push_back(file_pattern);
  } else {
    file_patterns = str_util::Split(file_pattern, ',');
    CHECK_EQ(file_patterns.size(), input_source_weights.size())
        << "There should be exactly one "
        << "input_source_weight per coma-separated value "
        << "in file_pattern.";
  }
  std::vector<RecordYielder*> yielders;
  for (int i = 0; i < file_patterns.size(); ++i) {
    if (require_sequential_order) {
      CHECK_EQ(file_patterns.size(), 1)
          << "require_sequential_order does not support record mixing.";
      yielders.push_back(SequentialRecordYielder::New(file_patterns[i]));
    } else {
      BasicRecordYielder::Options yopts;
      yopts.file_pattern = file_patterns[i];
      if (file_random_seed == 0) {
        yopts.seed = 0;  // Let the yielder pick a random seed.
      } else {
        yopts.seed =
            (file_random_seed + i) % (std::numeric_limits<int32>::max() - 1);
        if (yopts.seed == 0) {
          // Add 1 to avoid 0.
          ++yopts.seed;
        }
      }
      yopts.bufsize = file_buffer_size;
      yopts.parallelism = file_parallelism;
      yopts.source_id = i;
      yielders.push_back(BasicRecordYielder::New(yopts));
    }
  }

  RecordYielder* yielder = nullptr;
  if (yielders.size() > 1) {
    yielder = WeightedMixRecordYielder::New(file_random_seed, yielders,
                                            input_source_weights);
  } else {
    yielder = yielders.front();
  }
  return yielder;
}

}  // namespace lingvo
}  // namespace tensorflow
