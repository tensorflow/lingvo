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
#include "tensorflow/core/framework/attr_value.pb.h"
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
                                int64_t repeat_count);

// Creates BasicRecordYielder::Options for each file in the file pattern. For
// use only by ConstructYielder. Exposed to enable testing.
std::vector<BasicRecordYielder::Options> CreatePerFileYielderOptions(
    const std::vector<string>& file_patterns,
    const BasicRecordYielder::Options& yopts_tpl);

void GetBasicRecordYielderOptions(OpKernelConstruction* ctx,
                                  BasicRecordYielder::Options* yopts);

struct InputArgs {
  void Init(OpKernelConstruction* ctx) {
    GetBasicRecordYielderOptions(ctx, &yopts);
#define GETATTR(FIELD) OP_REQUIRES_OK(ctx, ctx->GetAttr(#FIELD, &FIELD));
    GETATTR(file_pattern);
    GETATTR(input_source_weights);
    GETATTR(bucket_upper_bound);
    GETATTR(bucket_batch_limit);
    GETATTR(bucket_adjust_every_n);
    GETATTR(flush_every_n);
    GETATTR(num_threads);
    num_merger_threads = num_threads;
    GETATTR(require_sequential_order);
    GETATTR(repeat_count);
    GETATTR(fatal_errors);
    OP_REQUIRES(
        ctx,
        std::is_sorted(bucket_upper_bound.begin(), bucket_upper_bound.end()),
        errors::InvalidArgument("Bucket_upper_bound is not sorted"));
    if (require_sequential_order) {
      num_threads = 1;
    }
#undef GETATTR
  }

  void GetRecordBatcherOptions(RecordBatcher::Options* opt) const {
    opt->bucket_upper_bound = bucket_upper_bound;
    opt->bucket_batch_limit = bucket_batch_limit;
    opt->bucket_adjust_every_n = bucket_adjust_every_n;
    opt->flush_every_n = flush_every_n;
    opt->num_threads = num_threads;
    opt->fatal_errors = fatal_errors;
  }

  bool require_sequential_order;
  int num_merger_threads = -1;
  int64_t bucket_adjust_every_n;
  int64_t flush_every_n;
  int64_t num_threads;
  int64_t repeat_count;
  BasicRecordYielder::Options yopts;
  string file_pattern;
  std::vector<float> input_source_weights;
  std::vector<int64_t> bucket_upper_bound;
  std::vector<int64_t> bucket_batch_limit;
  std::vector<string> fatal_errors;
};

// Base class for op kernels that emit training examples.
template <class RecordProcessorClass>
class InputOp : public OpKernel {
  static_assert(
      std::is_base_of<RecordProcessor, RecordProcessorClass>::value,
      "InputOp requires a RecordProcessor subclass as the template arg.");

 public:
  explicit InputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    args_.Init(ctx);
    LOG(INFO) << "Create RecordProcessor; source_id: " << args_.yopts.source_id;
    processor_ = new RecordProcessorClass(ctx);
    RecordYielder* yielder = CHECK_NOTNULL(ConstructYielder(
        args_.file_pattern, args_.input_source_weights, args_.yopts,
        args_.require_sequential_order, args_.repeat_count));
    LOG(INFO) << "Create batcher";
    RecordBatcher::Options bopts;
    args_.GetRecordBatcherOptions(&bopts);
    batcher_ = new RecordBatcher(bopts, yielder, processor_);
  }

  ~InputOp() override { delete batcher_; }

  void Compute(OpKernelContext* ctx) override {
    int64_t bucket_id;
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
  InputArgs args_;
};

template <class RecordProcessorClass>
class InputResource : public tensorflow::ResourceBase {
  static_assert(
      std::is_base_of<RecordProcessor, RecordProcessorClass>::value,
      "InputOp requires a RecordProcessor subclass as the template arg.");

 public:
  InputResource(const InputArgs& args, const NameAttrList& processor,
                const std::vector<int32>& dynamic_padding_dimensions,
                const std::vector<int32>& dynamic_padding_constants) {
    LOG(INFO) << "Create RecordProcessor; source_id: " << args.yopts.source_id;
    processor_ = new RecordProcessorClass(
        processor, args.num_merger_threads,
        dynamic_padding_dimensions, dynamic_padding_constants);
    RecordYielder* yielder = CHECK_NOTNULL(ConstructYielder(
        args.file_pattern, args.input_source_weights, args.yopts,
        args.require_sequential_order, args.repeat_count));
    LOG(INFO) << "Create batcher";
    RecordBatcher::Options bopts;
    args.GetRecordBatcherOptions(&bopts);
    batcher_ = new RecordBatcher(bopts, yielder, processor_);
  }

  std::string DebugString() const override { return "lingvo InputResource"; }

  ~InputResource() override { delete batcher_; }

  void GetNext(OpKernelContext* ctx) {
    int64_t bucket_id;
    TensorVec batch;
    OP_REQUIRES_OK(ctx, batcher_->GetNext(ctx, &bucket_id, &batch));
    VLOG(1) << "Produce a batch from bucket : " << bucket_id;
    OP_REQUIRES(ctx, static_cast<int>(batch.size()) == ctx->num_outputs(),
                errors::Internal("Unexpected batch: ", batch.size()));
    for (int i = 0; i < batch.size(); ++i) {
      ctx->set_output(i, batch[i]);
    }
  }

  Status EnsureInitialized(OpKernelContext* ctx) {
    return batcher_->EnsureInitialized(ctx);
  }

 protected:
  // Not owned - will be deleted in RecordBatcher.
  RecordProcessorClass* processor_ = nullptr;

 private:
  // Owned.
  BasicRecordYielder::Options yopts_;
  RecordBatcher* batcher_ = nullptr;
};

// Base class for op kernels that emit training examples.
template <class RecordProcessorClass>
class InputOpV2Create : public OpKernel {
  static_assert(
      std::is_base_of<RecordProcessor, RecordProcessorClass>::value,
      "InputOp requires a RecordProcessor subclass as the template arg.");

 public:
  explicit InputOpV2Create(OpKernelConstruction* ctx) : OpKernel(ctx) {
    args_.Init(ctx);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("processor", &processor_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_padding_dimensions",
                                     &dynamic_padding_dimensions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_padding_constants",
                                     &dynamic_padding_constants_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor handle_tensor;
    AllocatorAttributes attr;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                           &handle_tensor, attr));
    LOG(INFO) << "Create InputResource";
    auto* resource = new InputResource<RecordProcessorClass>(
        args_, processor_, dynamic_padding_dimensions_,
        dynamic_padding_constants_);
    // Function `EnsureInitialized` needs to access a registered
    // concrete_function in OpKernelContext. We need to call it here to ensure
    // that the same OpKernelContext is used. If we defer the call to
    // `InputOpV2GetNext::Compute()`, in Eager mode a different OpKernelContext
    // will be used and the concrete_function will not be found.
    OP_REQUIRES_OK(ctx, resource->EnsureInitialized(ctx));
    handle_tensor.scalar<ResourceHandle>()() =
        ResourceHandle::MakeRefCountingHandle(resource, ctx->device()->name(),
                                              /*dtypes_and_shapes=*/{},
                                              ctx->stack_trace());
    ctx->set_output(0, handle_tensor);
  }

 private:
  NameAttrList processor_;
  // The following are requred for GenericInputV2 only
  std::vector<int32> dynamic_padding_dimensions_;
  std::vector<int32> dynamic_padding_constants_;
  InputArgs args_;
};

template <class RecordProcessorClass>
class InputOpV2GetNext : public OpKernel {
 public:
  explicit InputOpV2GetNext(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    InputResource<RecordProcessorClass>* resource;
    const Tensor& handle_tensor = ctx->input(0);
    const ResourceHandle& handle = handle_tensor.scalar<ResourceHandle>()();
    typedef InputResource<RecordProcessorClass> resource_type;
    auto statusor = handle.GetResource<resource_type>();
    if (TF_PREDICT_FALSE(!statusor.ok())) {
      LOG(ERROR) << "Could not find the InputOpV2 resource: "
                 << statusor.status();
      return;
    }
    resource = *std::move(statusor);
    resource->GetNext(ctx);
  }
};

}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_INPUT_COMMON_H_
