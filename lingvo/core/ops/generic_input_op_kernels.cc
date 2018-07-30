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

#include <functional>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/work_sharder.h"
#include "lingvo/core/ops/input_common.h"

namespace tensorflow {
namespace lingvo {
namespace {

typedef std::function<void()> Closure;
typedef std::function<void(Closure)> Runner;

// ThreadLocalRunner::PerThread() is a thread local object which owns a thread
// pool with one thread. That thread is configured to disable as much
// TensorFlow runtime parallelism as we can.
//
// NOTE: Maybe a cpu-local object will work better, and the thread in
// ThreadLocalRunner can be affined to one cpu.
class ThreadLocalRunner {
 public:
  static ThreadLocalRunner& PerThread() {
    thread_local ThreadLocalRunner tl_runner;
    return tl_runner;
  }

  ThreadLocalRunner() : pool_(Env::Default(), "single", 1) {
    runner_ = [this](Closure c) { pool_.Schedule(Wrapper(c)); };
  }

  Runner* runner() { return &runner_; }

 private:
  thread::ThreadPool pool_;
  Runner runner_;

  class Wrapper : Closure {
   public:
    explicit Wrapper(Closure c) : c_(std::move(c)) {}

    void operator()() const {
      ScopedPerThreadMaxParallelism scope(1);
      c_();
    }

   private:
    Closure c_;
  };
};

class GenericInputProcessor : public RecordProcessor {
 public:
  explicit GenericInputProcessor(OpKernelConstruction* ctx) {
    auto flib = ctx->function_library();
    OP_REQUIRES(ctx, flib != nullptr, errors::Internal("No function library"));
    OP_REQUIRES_OK(ctx, flib->Clone(&fld_, &pflr_, &flib_));
    const NameAttrList* func;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("processor", &func));
    OP_REQUIRES_OK(ctx, flib_->Instantiate(func->name(),
                                           AttrSlice(&func->attr()), &handle_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_merger_threads_));
    num_merger_threads_ = std::max(4, num_merger_threads_ / 4);  // An estimate.
    merger_ = new thread::ThreadPool(
        Env::Default(), ThreadOptions(), "generic_input_merger",
        num_merger_threads_, /* low_latency_hint */ false);
    merger_runner_ = [this](Closure c) { merger_->Schedule(c); };
  }

  ~GenericInputProcessor() { delete merger_; }

  Status Process(const Rope& record, int64* bucket_key,
                 TensorVec* sample) override {
    // We expect that this input processor is used in conjunction with
    // RecordBatcher, which uses multiple threads to call this input
    // processor's Process(). Therefore, there is not much need for
    // processing each individual record using multiple threads
    // (tf_compute).
    FunctionLibraryRuntime::Options opts;
    opts.runner = ThreadLocalRunner::PerThread().runner();

    // The input is a single scalar string tensor.
    TensorVec args(1);
    args[0] = Tensor(DT_STRING, {});
    record.AppendTo(&args[0].scalar<string>()());
    *bucket_key = 1;
    sample->clear();
    Status status;
    Notification done;
    flib_->Run(opts, handle_, args, sample, [&](const Status& s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    TF_RETURN_IF_ERROR(status);
    if (sample->size() < 2) {
      return errors::InvalidArgument(
          "Generic input processor must return at least 2 tensors. but got ",
          sample->size());
    }
    const auto& bucket_key_tensor = (*sample)[sample->size() - 1];
    if (bucket_key_tensor.dtype() != DT_INT32 ||
        !TensorShapeUtils::IsScalar(bucket_key_tensor.shape())) {
      return errors::InvalidArgument(
          "Bucket key tensor is not an int32 scalar: ",
          DataTypeString(bucket_key_tensor.dtype()));
    }
    *bucket_key = bucket_key_tensor.scalar<int32>()();
    sample->pop_back();
    return Status::OK();
  }

  Status Merge(int64 bucket_id, const std::vector<TensorVec>& samples,
               TensorVec* batch) override {
    CHECK(!samples.empty());
    const auto num_samples = samples.size();
    const auto num_outs = samples[0].size();

    // Validate that samples can be merged: samples[:][i] has the same
    // type and shape.
    for (int i = 1; i < samples.size(); ++i) {
      if (samples[i].size() != num_outs) {
        return errors::InvalidArgument("Samples have different sizes: ",
                                       samples[i].size(), " vs. ", num_outs);
      }
      for (int j = 0; j < num_outs; ++j) {
        if (samples[i][j].dtype() != samples[0][j].dtype()) {
          return errors::InvalidArgument("Mismatch data types of samples (", i,
                                         "/", j, "): ", samples[i][j].dtype(),
                                         " vs. ", samples[0][j].dtype());
        }
        if (samples[i][j].shape() != samples[0][j].shape()) {
          return errors::InvalidArgument(
              "Mismatch shape of samples (", i, "/", j,
              "): ", samples[i][j].shape().DebugString(), " vs. ",
              samples[0][j].shape().DebugString());
        }
      }
    }

    batch->clear();
    for (int i = 0; i < num_outs; ++i) {
      DataType dtype = samples[0][i].dtype();
      switch (dtype) {
        case DT_FLOAT:
        case DT_UINT8:
        case DT_INT32:
        case DT_INT64:
        case DT_STRING:
        case DT_BFLOAT16:
          break;
        default:
          return errors::Unimplemented(DataTypeString(dtype),
                                       " is not supported.");
      }
      TensorShape shape = samples[0][i].shape();
      shape.InsertDim(0, num_samples);
      // The merged tensor is 1-rank higher and its 1st dimension
      // is the num_samples.
      batch->push_back(Tensor(dtype, shape));
    }

    Sharder::Do(num_samples /* total */, 1000 /* cost_per_unit */,
                [&](int64 start, int64 limit) {
                  for (int i = 0; i < num_outs; ++i) {
                    DataType dtype = samples[0][i].dtype();
                    Tensor* merged = &(*batch)[i];
                    for (int j = start; j < limit; ++j) {
                      switch (dtype) {
#define CASE(T)                                                        \
  case DataTypeToEnum<T>::value:                                       \
    merged->flat_outer_dims<T>().chip<0>(j) = samples[j][i].flat<T>(); \
    break;
                        CASE(float);
                        CASE(int32);
                        CASE(int64);
                        CASE(string);
                        CASE(uint8);
                        CASE(bfloat16);
#undef CASE
                        default:
                          LOG(FATAL) << "Unexpected " << DataTypeString(dtype);
                      }
                    }
                  }
                },
                merger_runner_, 1 + num_merger_threads_);
    return Status::OK();
  }

 private:
  std::unique_ptr<FunctionLibraryDefinition> fld_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  FunctionLibraryRuntime* flib_ = nullptr;  // Not owned.
  FunctionLibraryRuntime::Handle handle_;
  int num_merger_threads_ = -1;
  thread::ThreadPool* merger_ = nullptr;
  Runner merger_runner_;

  TF_DISALLOW_COPY_AND_ASSIGN(GenericInputProcessor);
};

REGISTER_KERNEL_BUILDER(Name("GenericInput").Device(DEVICE_CPU),
                        InputOp<GenericInputProcessor>);
}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
