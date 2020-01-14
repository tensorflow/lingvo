/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "lingvo/tasks/car/ops/ps_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lingvo {
namespace car {

class SamplePointsOp : public OpKernel {
 public:
  explicit SamplePointsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string method;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("center_selector", &method));
    OP_REQUIRES(ctx, method == "uniform" || method == "farthest",
                errors::InvalidArgument(method, " is not valid."));
    if (method == "uniform") {
      opts_.cmethod = PSUtils::Options::C_UNIFORM;
    } else {
      CHECK_EQ(method, "farthest");
      opts_.cmethod = PSUtils::Options::C_FARTHEST;
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("neighbor_sampler", &method));
    OP_REQUIRES(ctx, method == "uniform" || method == "closest",
                errors::InvalidArgument(method, " is not valid."));
    if (method == "uniform") {
      opts_.nmethod = PSUtils::Options::N_UNIFORM;
    } else {
      CHECK_EQ(method, "closest");
      opts_.nmethod = PSUtils::Options::N_CLOSEST;
    }

    OP_REQUIRES_OK(ctx, ctx->GetAttr("neighbor_algorithm", &method));
    OP_REQUIRES(
        ctx, method == "auto" || method == "hash",
        errors::InvalidArgument(method, " is not a valid neighbor algorithm."));
    if (method == "hash") {
      opts_.neighbor_search_algorithm = PSUtils::Options::N_HASH;
    }

    LOG(INFO) << "Sampling options: " << opts_.DebugString();
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_centers", &opts_.num_centers));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("center_z_min", &opts_.center_z_min));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("center_z_max", &opts_.center_z_max));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_neighbors", &opts_.num_neighbors));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_distance", &opts_.max_dist));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("random_seed", &opts_.random_seed));
  }

  void Compute(OpKernelContext* ctx) override {
    PSUtils ps(opts_);
    auto ret = ps.Sample(ctx->input(0), ctx->input(1),
                         ctx->input(2).scalar<int32>()());
    ctx->set_output(0, ret.center);
    ctx->set_output(1, ret.center_padding);
    ctx->set_output(2, ret.indices);
    ctx->set_output(3, ret.indices_padding);
  }

 private:
  PSUtils::Options opts_;
};

REGISTER_KERNEL_BUILDER(Name("SamplePoints").Device(DEVICE_CPU),
                        SamplePointsOp);

}  // namespace car
}  // namespace lingvo
}  // namespace tensorflow
