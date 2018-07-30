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
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace lingvo {
namespace {

// Reads a text file containing 'step value' records, and finds the step that
// corresponds to the lowest-value record, within a given tolerance.
class BestStepOp : public OpKernel {
 public:
  explicit BestStepOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hist_file", &hist_file_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tol", &tol_));
    CHECK_GE(tol_, 0.0);
  }

  void Compute(OpKernelContext* ctx) override {
    int64 best_step = 0, last_step = 0;
    float best_val = 0.0;
    const Status status = ctx->env()->FileExists(hist_file_);
    if (status.ok()) {
      std::unique_ptr<RandomAccessFile> file;
      OP_REQUIRES_OK(ctx, ctx->env()->NewRandomAccessFile(hist_file_, &file));
      std::unique_ptr<io::RandomAccessInputStream> input_stream(
          new io::RandomAccessInputStream(file.get()));
      io::BufferedInputStream in(input_stream.get(), 4 << 10);
      string line;
      std::vector<float> rec;
      while (true) {
        const Status s = in.ReadLine(&line);
        if (errors::IsOutOfRange(s)) break;
        TF_CHECK_OK(s);
        CHECK(str_util::SplitAndParseAsFloats(line, ' ', &rec));
        CHECK_EQ(rec.size(), 2);
        last_step = rec[0];
        const float val = rec[1];
        if (best_step == 0 || val + tol_ < best_val) {
          best_step = last_step;
          best_val = val;
        }
      }
    }

    Tensor* res;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({2}), &res));
    res->vec<int64>()(0) = best_step;
    res->vec<int64>()(1) = last_step;
  }

 private:
  string hist_file_;
  float tol_ = 0.0;
};

REGISTER_KERNEL_BUILDER(Name("BestStep").Device(DEVICE_CPU), BestStepOp);

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
