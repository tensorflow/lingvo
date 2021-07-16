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

std::vector<string> VerifyAndSplitFilePattern(
    const string& file_pattern,
    const std::vector<float>& input_source_weights) {
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

  return file_patterns;
}

std::vector<BasicRecordYielder::Options> CreatePerFileYielderOptions(
    const std::vector<string>& file_patterns,
    const BasicRecordYielder::Options& yopts_tpl) {
  std::vector<BasicRecordYielder::Options> yielder_options;

  for (int i = 0; i < file_patterns.size(); ++i) {
    BasicRecordYielder::Options yopts(yopts_tpl);
    yopts.file_pattern = file_patterns[i];
    if (yopts_tpl.seed == 0) {
      yopts.seed = 0;  // Let the yielder pick a random seed.
    } else {
      yopts.seed =
          (yopts_tpl.seed + i) % (std::numeric_limits<int32>::max() - 1);
      if (yopts.seed == 0) {
        // Add 1 to avoid 0.
        ++yopts.seed;
      }
    }
    // we use yopts_tpl.source_id as an offset for all yielder source_id values.
    yopts.source_id = yopts_tpl.source_id + i;
    yielder_options.push_back(yopts);
  }

  return yielder_options;
}

RecordYielder* ConstructMixYielderFromOptions(
    const std::vector<BasicRecordYielder::Options>& yielder_options,
    const std::vector<float>& input_source_weights, const int64 seed) {
  RecordYielder* yielder = nullptr;
  if (yielder_options.size() == 1) {
    yielder = BasicRecordYielder::New(yielder_options.front());
  } else {
    std::vector<RecordYielder*> yielders;
    yielders.reserve(yielder_options.size());
    for (const auto& yopts : yielder_options) {
      yielders.push_back(BasicRecordYielder::New(yopts));
    }
    yielder =
        WeightedMixRecordYielder::New(seed, yielders, input_source_weights);
  }
  return yielder;
}

RecordYielder* ConstructYielder(const string& file_pattern,
                                const std::vector<float>& input_source_weights,
                                const BasicRecordYielder::Options& yopts_tpl,
                                bool require_sequential_order,
                                int64 repeat_count) {
  std::vector<string> file_patterns =
      VerifyAndSplitFilePattern(file_pattern, input_source_weights);

  if (require_sequential_order) {
    CHECK_EQ(file_patterns.size(), 1)
        << "require_sequential_order does not support record mixing or "
        << "chaining.";
    return SequentialRecordYielder::New(file_patterns.front(), repeat_count);
  } else {
    CHECK_EQ(repeat_count, -1) << "Repeat count must not be set unless "
                                  "require_sequential_order is true.";
  }

  std::vector<BasicRecordYielder::Options> yielder_options =
      CreatePerFileYielderOptions(file_patterns, yopts_tpl);

  return ConstructMixYielderFromOptions(yielder_options, input_source_weights,
                                        yopts_tpl.seed);
}

void GetBasicRecordYielderOptions(OpKernelConstruction* ctx,
                                  BasicRecordYielder::Options* yopts) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("file_pattern", &(yopts->file_pattern)));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("file_random_seed", &(yopts->seed)));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("file_buffer_size", &(yopts->bufsize)));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("file_parallelism", &(yopts->parallelism)));
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("num_input_replicas", &(yopts->num_input_replicas)));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr("input_replica_id", &(yopts->input_replica_id)));

  OP_REQUIRES_OK(ctx, ctx->GetAttr("source_id_offset", &(yopts->source_id)));
}

}  // namespace lingvo
}  // namespace tensorflow
