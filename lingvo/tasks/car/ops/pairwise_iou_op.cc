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

#include "lingvo/tasks/car/ops/box_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lingvo {
namespace {

class PairwiseIoUOp : public OpKernel {
 public:
  explicit PairwiseIoUOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] must be a matrix, but get ",
                                        a.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[0] must be a matrix, but get ",
                                        b.shape().DebugString()));
    OP_REQUIRES(ctx, 7 == a.dim_size(1),
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        a.shape().DebugString()));
    OP_REQUIRES(ctx, 7 == b.dim_size(1),
                errors::InvalidArgument("Matrix size-incompatible: In[1]: ",
                                        b.shape().DebugString()));

    const int n_a = a.dim_size(0);
    const int n_b = b.dim_size(0);

    Tensor* iou_a_b = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("iou", TensorShape({n_a, n_b}), &iou_a_b));

    auto t_iou_a_b = iou_a_b->matrix<float>();

    std::vector<box::Upright3DBox> box_a = box::ParseBoxesFromTensor(a);
    std::vector<box::Upright3DBox> box_b = box::ParseBoxesFromTensor(b);
    for (int i_a = 0; i_a < n_a; ++i_a) {
      for (int i_b = 0; i_b < n_b; ++i_b) {
        t_iou_a_b(i_a, i_b) = box_a[i_a].IoU(box_b[i_b]);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("PairwiseIou3D").Device(DEVICE_CPU),
                        PairwiseIoUOp);

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
