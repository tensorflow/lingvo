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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace lingvo {
namespace {

static const int kUnknown = -1;

class AssertShapeMatchOp : public OpKernel {
 public:
  explicit AssertShapeMatchOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("msg", &msg_));
  }

  ~AssertShapeMatchOp() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& y = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(x.shape()),
                errors::InvalidArgument("x must be a vector."));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(y.shape()),
                errors::InvalidArgument("y must be a vector."));
    bool match = true;
    if (x.NumElements() != y.NumElements()) {
      match = false;
    } else {
      auto Tx = x.flat<int32>();
      auto Ty = y.flat<int32>();
      for (int i = 0; i < x.NumElements(); ++i) {
        if ((Tx(i) != kUnknown) && (Ty(i) != kUnknown) && (Tx(i) != Ty(i))) {
          match = false;
        }
      }
    }
    OP_REQUIRES(ctx, match,
                errors::InvalidArgument(msg_, " mismatch shape: x=[",
                                        x.SummarizeValue(10), "] y=[",
                                        y.SummarizeValue(10), "]"));
  }

 private:
  string msg_;
};
REGISTER_KERNEL_BUILDER(Name("AssertShapeMatch").Device(DEVICE_CPU),
                        AssertShapeMatchOp);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("AssertShapeMatch").Device(DEVICE_GPU).HostMemory("x").HostMemory("y"),
    AssertShapeMatchOp);
#endif

class AssertSameDim0Op : public OpKernel {
 public:
  explicit AssertSameDim0Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("msg", &msg_));
  }

  ~AssertSameDim0Op() override {}

  void Compute(OpKernelContext* ctx) override {
    if (ctx->num_inputs() == 0) {
      // A no-op for empty list of inputs.
      return;
    }
    const auto& x = ctx->input(0);
    OP_REQUIRES(ctx, !TensorShapeUtils::IsScalar(x.shape()),
                errors::InvalidArgument(msg_, " 0-th input is a scalar."));
    const auto dim0 = x.dim_size(0);
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      const auto& y = ctx->input(i);
      OP_REQUIRES(
          ctx, !TensorShapeUtils::IsScalar(y.shape()),
          errors::InvalidArgument(msg_, " ", i, "-th input is a scalar."));
      OP_REQUIRES(ctx, dim0 == y.dim_size(0),
                  errors::InvalidArgument(
                      msg_, " ", i, "-th input has a different dim0: ", dim0,
                      " ", y.dim_size(0)));
    }
  }

 private:
  string msg_;
};
REGISTER_KERNEL_BUILDER(Name("AssertSameDim0").Device(DEVICE_CPU),
                        AssertSameDim0Op);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("AssertSameDim0").Device(DEVICE_GPU),
                        AssertSameDim0Op);
#endif

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
