/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <utility>

#include "lingvo/core/ops/preconditioner_captain.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"


const int kShampooComputeThreads = 64;

namespace tensorflow {
namespace lingvo {
namespace {

void MakePreconditionerCaptainOptionsFromContext(
    OpKernelConstruction* context, PreconditionerCaptainOptions* options) {
  OP_REQUIRES_OK(context,
                 context->GetAttr("preconditioner_compute_graphdef",
                                  &options->preconditioner_compute_graphdef));
  options->num_compute_threads = kShampooComputeThreads;
}

}  // namespace

PreconditionerCaptain* global_preconditioner_captain = nullptr;

void PreconditionerCaptainServiceInit(
    const PreconditionerCaptainOptions& options) {
  global_preconditioner_captain = new PreconditionerCaptain(options);
}

PreconditionerCaptain* get_or_create_preconditioner_captain(
    const PreconditionerCaptainOptions& options) {
  static std::once_flag global_preconditioner_captain_init_once;
  std::call_once(global_preconditioner_captain_init_once,
                  &PreconditionerCaptainServiceInit, options);
  return global_preconditioner_captain;
}

class GetPreconditioners : public OpKernel {
 public:
  explicit GetPreconditioners(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("keys", &keys_));
    MakePreconditionerCaptainOptionsFromContext(context, &options_);
  }

  void Compute(OpKernelContext* context) override {
    auto* preconditioner_captain =
        get_or_create_preconditioner_captain(options_);
    std::vector<bool> statuses(keys_.size());
    OpInputList shapes;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes));

    OpOutputList output_list;
    OP_REQUIRES_OK(context, context->output_list("outputs", &output_list));

    OpOutputList status_list;
    OP_REQUIRES_OK(context, context->output_list("statuses", &status_list));

    for (int i = 0; i < keys_.size(); ++i) {
      bool ok;
      Tensor output = preconditioner_captain->GetPreconditioner(keys_[i], &ok);
      if (ok) {
        output_list.set(i, output);
      } else {
        auto shape_t = shapes[i].flat<int32>();
        TensorShape shape;
        TF_CHECK_OK(TensorShapeUtils::MakeShape(shape_t.data(), shape_t.size(),
                                                &shape));
        Tensor zero_output(DT_FLOAT, shape);
        zero_output.flat<float>().setZero();
        output_list.set(i, zero_output);
      }
      Tensor status_t(DT_BOOL, TensorShape({}));
      status_t.scalar<bool>()() = ok;
      status_list.set(i, status_t);
    }
  }

 private:
  // Options for preconditioner.
  PreconditionerCaptainOptions options_;
  // Keys for the tensors.
  std::vector<string> keys_;
};

REGISTER_KERNEL_BUILDER(Name("GetPreconditioners").Device(DEVICE_CPU),
                        GetPreconditioners);

class ComputePreconditionersOp : public OpKernel {
 public:
  explicit ComputePreconditionersOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("keys", &keys_));
    MakePreconditionerCaptainOptionsFromContext(context, &options_);
    OP_REQUIRES_OK(context, context->GetAttr("sync", &sync_));
  }

  void Compute(OpKernelContext* context) override {
    auto* preconditioner_captain =
        get_or_create_preconditioner_captain(options_);
    OpInputList inputs;
    OP_REQUIRES_OK(context, context->input_list("inputs", &inputs));
    OpInputList exponents;
    OP_REQUIRES_OK(context, context->input_list("exponents", &exponents));
    const Tensor* global_step_t;
    OP_REQUIRES_OK(context, context->input("global_step", &global_step_t));
    const int global_step = global_step_t->scalar<int>()();

    for (int i = 0; i < inputs.size(); ++i) {
      Tensor statistics = inputs[i];
      Tensor exponent = exponents[i];
      preconditioner_captain->InsertGradientStatistics(
          keys_[i], statistics, exponent, global_step, sync_);
    }
  }

 private:
  // Options for preconditioner.
  PreconditionerCaptainOptions options_;
  // Keys for the tensors.
  std::vector<string> keys_;
  // Whether to run preconditioner synchronously.
  bool sync_ = false;
};

REGISTER_KERNEL_BUILDER(Name("ComputePreconditioners").Device(DEVICE_CPU),
                        ComputePreconditionersOp);

}  // namespace lingvo
}  // namespace tensorflow
