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

#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "lingvo/core/ops/mutex.h"

namespace tensorflow {
namespace lingvo {
namespace {

typedef FunctionLibraryRuntime::Handle FHandle;

void SetRunOptions(OpKernelContext* ctx, FunctionLibraryRuntime::Options* opts,
                   bool always_collect_stats) {
  opts->step_id = ctx->step_id();
  opts->rendezvous = ctx->rendezvous();
  opts->cancellation_manager = ctx->cancellation_manager();
  if (always_collect_stats) {
    opts->stats_collector = ctx->stats_collector();
  }
  opts->runner = ctx->runner();
}

class CachedCallOp : public AsyncOpKernel {
 public:
  explicit CachedCallOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx), not_initing_(this, &ME::NotIniting) {
    flib_ = ctx->function_library();
    OP_REQUIRES(ctx, flib_ != nullptr, errors::Internal("No function library"));
    const NameAttrList* func;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func));
    OP_REQUIRES_OK(ctx, flib_->Instantiate(func->name(),
                                           AttrSlice(&func->attr()), &handle_));
  }

  ~CachedCallOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    mu_.Lock();

    while (true) {
      // First call.
      if (state_ == UNINIT) {
        break;
      }

      // Has called f and cached the result.
      if (state_ == INITED) {
        ctx->SetStatus(status_);
        for (int i = 0; i < rets_.size(); ++i) {
          ctx->set_output(i, rets_[i]);
        }
        mu_.Unlock();
        done();
        return;
      }

      // Another call is being executed.
      mu_.Await(not_initing_);
    }

    state_ = INITING;
    mu_.Unlock();

    // Call f once and cache the result.
    SetRunOptions(ctx, &opts_, true /* always_collect_stats */);
    flib_->Run(opts_, handle_, args_, &rets_,
               // Done callback
               [this, ctx, done](Status s) {
                 ctx->SetStatus(s);
                 for (int i = 0; i < rets_.size(); ++i) {
                   ctx->set_output(i, rets_[i]);
                 }
                 done();

                 MutexLock l(&mu_);
                 status_ = s;
                 state_ = INITED;
               });
  }

 private:
  typedef CachedCallOp ME;

  bool NotIniting() const SHARED_LOCKS_REQUIRED(mu_) {
    return state_ != INITING;
  }

  FunctionLibraryRuntime* flib_ = nullptr;
  FunctionLibraryRuntime::Options opts_;
  FHandle handle_;

  Mutex mu_;
  Condition not_initing_;
  enum State {
    UNINIT,
    INITING,
    INITED,
  };
  State state_ = UNINIT;
  Status status_;
  std::vector<Tensor> args_;
  std::vector<Tensor> rets_;
};

REGISTER_KERNEL_BUILDER(Name("CachedCall").Device(DEVICE_CPU), CachedCallOp);

}  // namespace
}  // end namespace lingvo
}  // end namespace tensorflow
