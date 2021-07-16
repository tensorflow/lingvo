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

#ifndef LINGVO_CORE_OPS_INPUT_COMMON_H_
#define LINGVO_CORE_OPS_INPUT_COMMON_H_

#include <limits>

#include "lingvo/core/ops/record_batcher.h"
#include "lingvo/core/ops/record_yielder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace lingvo {

// Constructs a RecordYielder for the given file pattern. The returned instance
// can be a single BasicRecordYielder or mixes multiple yielders with weights.
// In either case `yopts_tpl` is used to construct BasicRecordYielder instances.
// However, when constructing multiple BasicRecordYielder instances, the
// random seeds will be either all 0 or all different across instances.
RecordYielder* ConstructYielder(const string& file_pattern,
                                const std::vector<float>& input_source_weights,
                                const BasicRecordYielder::Options& yopts_tpl,
                                bool require_sequential_order,
                                int64 repeat_count);

// Creates BasicRecordYielder::Options for each file in the file pattern. For
// use only by ConstructYielder. Exposed to enable testing.
std::vector<BasicRecordYielder::Options> CreatePerFileYielderOptions(
    const std::vector<string>& file_patterns,
    const BasicRecordYielder::Options& yopts_tpl);

void GetBasicRecordYielderOptions(OpKernelConstruction* ctx,
                                  BasicRecordYielder::Options* yopts);

// Base class for op kernels that emit training examples.
template <class RecordProcessorClass>
class InputOp : public OpKernel {
  static_assert(
      std::is_base_of<RecordProcessor, RecordProcessorClass>::value,
      "InputOp requires a RecordProcessor subclass as the template arg.");

 public:
  explicit InputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    BasicRecordYielder::Options yopts;
    GetBasicRecordYielderOptions(ctx, &yopts);
#define GETATTR(TYPE, FIELD) \
  TYPE FIELD;                \
  OP_REQUIRES_OK(ctx, ctx->GetAttr(#FIELD, &FIELD));

    GETATTR(string, file_pattern);
    GETATTR(std::vector<float>, input_source_weights);
    GETATTR(std::vector<int64>, bucket_upper_bound);
    GETATTR(std::vector<int64>, bucket_batch_limit);
    GETATTR(int64, bucket_adjust_every_n);
    GETATTR(int64, flush_every_n);
    GETATTR(int64, num_threads);
    GETATTR(bool, require_sequential_order);
    GETATTR(int64, repeat_count);
    GETATTR(std::vector<string>, fatal_errors);
#undef GETATTR
    OP_REQUIRES(
        ctx,
        std::is_sorted(bucket_upper_bound.begin(), bucket_upper_bound.end()),
        errors::InvalidArgument("Bucket_upper_bound is not sorted"));
    if (require_sequential_order) {
      num_threads = 1;
    }
    LOG(INFO) << "Create RecordProcessor; source_id: " << yopts.source_id;
    processor_ = new RecordProcessorClass(ctx);
    RecordYielder* yielder = CHECK_NOTNULL(
        ConstructYielder(file_pattern, input_source_weights, yopts,
                         require_sequential_order, repeat_count));
    LOG(INFO) << "Create batcher";
    RecordBatcher::Options bopts;
    bopts.bucket_upper_bound = bucket_upper_bound;
    bopts.bucket_batch_limit = bucket_batch_limit;
    bopts.bucket_adjust_every_n = bucket_adjust_every_n;
    bopts.flush_every_n = flush_every_n;
    bopts.num_threads = num_threads;
    bopts.fatal_errors = fatal_errors;
    batcher_ = new RecordBatcher(bopts, yielder, processor_);
  }

  ~InputOp() override { delete batcher_; }

  void Compute(OpKernelContext* ctx) override {
    int64 bucket_id;
    TensorVec batch;
    OP_REQUIRES_OK(ctx, batcher_->GetNext(ctx, &bucket_id, &batch));
    VLOG(1) << "Produce a batch from bucket : " << bucket_id;
    OP_REQUIRES(ctx, static_cast<int>(batch.size()) == ctx->num_outputs(),
                errors::Internal("Unexpected batch: ", batch.size()));
    for (int i = 0; i < batch.size(); ++i) {
      ctx->set_output(i, batch[i]);
    }
  }

 protected:
  // Not owned.
  RecordProcessorClass* processor_ = nullptr;

 private:
  // Owned.
  RecordBatcher* batcher_ = nullptr;
};

}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_INPUT_COMMON_H_
