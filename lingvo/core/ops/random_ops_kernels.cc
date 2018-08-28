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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "lingvo/core/ops/mutex.h"

namespace tensorflow {
namespace lingvo {
namespace {

class RandomPermutationSequenceOp : public OpKernel {
 public:
  explicit RandomPermutationSequenceOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num", &num_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch", &batch_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("repeat", &repeat_));
    int64 seed;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed));
    if (seed == 0) {
      std::random_device device("/dev/urandom");
      seed = std::mt19937_64(device())();
    }
    rnd_.seed(seed);
    Fill();
  }

  void Compute(OpKernelContext* ctx) override {
    MutexLock l(&mu_);
    OP_REQUIRES(ctx, !ids_.empty() || repeat_,
                errors::OutOfRange("Epoch ended."));
    if (ids_.empty()) Fill();

    int start = 0;
    int n = std::min<int>(batch_, ids_.size());

    Tensor* out;
    const int out_size = repeat_ ? batch_ : n;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({out_size}), &out));
    for (int i = 0; i < n; ++i) {
      out->flat<int32>()(i) = ids_[ids_.size() - 1 - i];
    }
    ids_.resize(ids_.size() - n);
    start += n;

    while (repeat_ && ids_.empty()) {
      Fill();
      n = std::min<int>(batch_ - start, ids_.size());
      for (int i = 0; i < n; ++i) {
        out->flat<int32>()(start + i) = ids_[ids_.size() - 1 - i];
      }
      ids_.resize(ids_.size() - n);
      start += n;
    }
  }

 private:
  int32 num_;
  int32 batch_;
  bool repeat_;

  Mutex mu_;
  std::mt19937 rnd_;
  std::vector<int32> ids_;

  void Fill() {
    CHECK(ids_.empty());
    ids_.resize(num_);
    for (int i = 0; i < num_; ++i) ids_[i] = i;
    for (int i = num_ - 1; i > 0; --i) {
      const int32 pos = rnd_() % i;
      std::swap(ids_[i], ids_[pos]);
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("RandomPermutationSequence").Device(DEVICE_CPU),
                        RandomPermutationSequenceOp);
}  // namespace
}  // end namespace lingvo
}  // end namespace tensorflow
