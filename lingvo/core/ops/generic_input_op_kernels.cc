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
#include <memory>

#include "absl/memory/memory.h"
#include "lingvo/core/ops/input_common.h"
#include "lingvo/core/ops/thread_local_runner.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace lingvo {
namespace {

typedef std::function<void()> Closure;
typedef std::function<void(Closure)> Runner;
// Creates a self-contained function library definition.
// This allows us to e.g. call functions when invoked from a tf.data.Dataset.
Status CreateFunctionLibraryDefinition(
    const FunctionLibraryDefinition* lib_def, const string& func_name,
    std::unique_ptr<FunctionLibraryDefinition>* result) {
  DCHECK(lib_def != nullptr);
  const FunctionDef* fdef = lib_def->Find(func_name);
  if (TF_PREDICT_FALSE(fdef == nullptr)) {
    return errors::FailedPrecondition(strings::StrCat(
        "Could not find required function definition ", func_name));
  }
  *result = absl::make_unique<FunctionLibraryDefinition>(
      lib_def->ReachableDefinitions(*fdef));
  return (*result)->CopyFunctionDefFrom(func_name, *lib_def);
}

// Helper class to invoke a user-provided function (processing logic).
class ProcessorFn {
 public:
  // Creates a user-supplied function that can be invoked from a cloned
  // runtime, with a self-contained function library definition.
  static Status Create(OpKernelContext* ctx, const NameAttrList* func,
                       std::unique_ptr<ProcessorFn>* out_function) {
    std::unique_ptr<FunctionLibraryDefinition> dummy_fld;
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr;
    FunctionLibraryRuntime* cloned_flib = nullptr;
    // Skip flib definition and instead use our own self-contained definition
    // that can call the user-supplied function.
    TF_RETURN_IF_ERROR(ctx->function_library()->Clone(&dummy_fld, &pflr,
                                                      &cloned_flib,
                                                      /*skip_flib_def=*/true));
    std::unique_ptr<FunctionLibraryDefinition> fld;
    TF_RETURN_IF_ERROR(CreateFunctionLibraryDefinition(
        ctx->function_library()->GetFunctionLibraryDefinition(), func->name(),
        &fld));
    FunctionLibraryRuntime::InstantiateOptions options;
    options.lib_def = fld.get();
    options.create_kernels_eagerly = true;
    FunctionLibraryRuntime::Handle handle;
    TF_RETURN_IF_ERROR(cloned_flib->Instantiate(
        func->name(), AttrSlice{&func->attr()}, options, &handle));
    // Using `new` to access a non-public constructor.
    *out_function = absl::WrapUnique(
        new ProcessorFn{cloned_flib, std::move(fld), std::move(pflr), handle});
    return Status::OK();
  }

  // Executes the user-defined function.
  Status Run(TensorVec&&args, TensorVec* output) {
    // We expect that this input processor is used in conjunction with
    // RecordBatcher, which uses multiple threads to call this input
    // processor's Process(). Therefore, there is not much need for
    // processing each individual record using multiple threads
    // (tf_compute).
    FunctionLibraryRuntime::Options opts;
    // Create a step container that uses resource manager to cleanup state
    // after the step is complete.
    ScopedStepContainer step_container(
        step_id_counter_.fetch_add(1),
        [this](const string& name) {
          auto status = flib_->device()->resource_manager()->Cleanup(name);
          if (!status.ok()) {
            LOG(ERROR) << "Error cleaning up resources:" << status;
          }
        },
        "GenericInputProcessor");
    opts.step_container = &step_container;
    opts.runner = ThreadLocalRunner::PerThread().runner();

    Status status;
    Notification done;
    flib_->Run(opts, handle_, args, output, [&](const Status& s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    return status;
  }

 private:
  explicit ProcessorFn(FunctionLibraryRuntime* flib,
                       std::unique_ptr<FunctionLibraryDefinition> fld,
                       std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                       FunctionLibraryRuntime::Handle handle)
      : fld_{std::move(fld)},
        pflr_{std::move(pflr)},
        flib_{flib},
        handle_{handle} {}

  std::unique_ptr<FunctionLibraryDefinition> fld_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  FunctionLibraryRuntime* flib_ = nullptr;  // Not owned.
  FunctionLibraryRuntime::Handle handle_;
  std::atomic_int_fast64_t step_id_counter_;
};

class GenericInputProcessor : public RecordProcessor {
 public:
  explicit GenericInputProcessor(OpKernelConstruction* ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("processor", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_merger_threads_));
    num_merger_threads_ = std::max(4, num_merger_threads_ / 4);  // An estimate.
    merger_ = new thread::ThreadPool(
        Env::Default(), ThreadOptions(), "generic_input_merger",
        num_merger_threads_, /* low_latency_hint */ false);
    merger_runner_ = [this](Closure c) { merger_->Schedule(c); };
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_padding_dimensions",
                                     &dynamic_padding_dimensions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dynamic_padding_constants",
                                     &dynamic_padding_constants_));
  }

  Status Initialize(OpKernelContext* ctx) override {
    return ProcessorFn::Create(ctx, &func_, &processor_fn_);
  }

  ~GenericInputProcessor() { delete merger_; }

  Status Process(const Record& record, int64* bucket_key,
                 TensorVec* sample) override {
    // Generates <source_id, record> pair as the resulting Tensors.
    TensorVec args(2);
    args[0] = Tensor(DT_INT32, {});
    args[0].scalar<int32>()() = record.source_id;
    args[1] = Tensor(DT_STRING, {});
    args[1].scalar<tensorflow::tstring>()().append(std::string(record.value));
    *bucket_key = 1;
    sample->clear();
    TF_RETURN_IF_ERROR(processor_fn_->Run(std::move(args), sample));
    if (sample->size() < 2) {
      LOG(FATAL)
          << "Generic input processor must return at least 2 tensors. but got "
          << sample->size();
    }
    const auto& bucket_key_tensor = (*sample)[sample->size() - 1];
    if (bucket_key_tensor.dtype() != DT_INT32 ||
        !TensorShapeUtils::IsScalar(bucket_key_tensor.shape())) {
      LOG(FATAL) << "Bucket key tensor is not an int32 scalar: "
                 << DataTypeString(bucket_key_tensor.dtype());
    }
    *bucket_key = bucket_key_tensor.scalar<int32>()();
    if (*bucket_key < 0) {
      return tensorflow::errors::Cancelled(
          strings::StrCat("Batch has negative bucket key: ", *bucket_key));
    }
    sample->pop_back();
    return Status::OK();
  }

  Status Merge(int64 bucket_size, const std::vector<TensorVec>& samples,
               TensorVec* batch) override {
    CHECK(!samples.empty());
    const auto num_samples = samples.size();
    const auto num_outs = samples[0].size();

    std::vector<TensorVec> padded_samples(samples.begin(), samples.end());
    if (!dynamic_padding_dimensions_.empty()) {
      CHECK(dynamic_padding_dimensions_.size() == num_outs);
      CHECK(dynamic_padding_constants_.size() == num_outs);

      for (int j = 0; j < num_outs; ++j) {
        const int pad_dim = dynamic_padding_dimensions_[j];
        if (pad_dim == -1) {
          continue;
        }
        const int pad_value = dynamic_padding_constants_[j];

        int64 max_length = 0;
        for (int i = 0; i < samples.size(); ++i) {
          max_length = std::max(max_length, samples[i][j].dim_size(pad_dim));
        }

        for (int i = 0; i < samples.size(); ++i) {
          const auto& src = samples[i][j];
          if (src.dims() > 0 && src.dim_size(pad_dim) < max_length) {
            DataType dtype = src.dtype();
            TensorShape dst_shape(src.shape());
            dst_shape.set_dim(pad_dim, max_length);
            Tensor dst(dtype, dst_shape);
            switch (dtype) {
#define CASE(T)                                                  \
  case DataTypeToEnum<T>::value:                                 \
    dst.flat<T>().setConstant(pad_value);                        \
    if (src.NumElements() > 0) {                                 \
      auto src_t = src.flat_inner_outer_dims<T, 2>(pad_dim - 1); \
      auto dst_t = dst.flat_inner_outer_dims<T, 2>(pad_dim - 1); \
      typedef Eigen::DSizes<Eigen::DenseIndex, 2> DSizes;        \
      dst_t.slice(DSizes(), DSizes(src_t.dimensions())) = src_t; \
    }                                                            \
    break

              CASE(float);
              CASE(int32);
              CASE(int64);
#undef CASE
              default:
                LOG(FATAL) << "Unexpected " << DataTypeString(dtype);
            }
            std::swap(padded_samples[i][j], dst);
          }
        }
      }
    }

    // Validate that samples can be merged: samples[:][i] has the same
    // type and shape.
    for (int i = 1; i < padded_samples.size(); ++i) {
      if (padded_samples[i].size() != num_outs) {
        LOG(FATAL) << "Samples have different sizes: " << samples[i].size()
                   << " vs. " << num_outs;
      }
      for (int j = 0; j < num_outs; ++j) {
        if (padded_samples[i][j].dtype() != padded_samples[0][j].dtype()) {
          LOG(FATAL) << "Mismatch data types of samples (" << i << "/" << j
                     << "): " << samples[i][j].dtype() << " vs. "
                     << samples[0][j].dtype();
        }
        if (padded_samples[i][j].shape() != padded_samples[0][j].shape()) {
          LOG(FATAL) << "Mismatch shape of samples (" << i << "/" << j
                     << "): " << samples[i][j].shape().DebugString() << " vs. "
                     << samples[0][j].shape().DebugString();
        }
      }
    }

    batch->clear();
    for (int i = 0; i < num_outs; ++i) {
      const Tensor& src = padded_samples[0][i];
      DataType dtype = src.dtype();
      switch (dtype) {
        case DT_FLOAT:
        case DT_UINT8:
        case DT_INT32:
        case DT_INT64:
        case DT_STRING:
        case DT_BFLOAT16:
        case DT_COMPLEX64:
        case DT_COMPLEX128:
        case DT_BOOL:
          break;
        default:
          LOG(FATAL) << DataTypeString(dtype) << " is not supported.";
      }
      TensorShape shape = src.shape();
      shape.InsertDim(0, num_samples);
      // The merged tensor is 1-rank higher and its 1st dimension
      // is the num_samples.
      if (num_samples == 1) {
        // Avoid memcpy if there is just one sample.
        Tensor reshaped(dtype);
        CHECK(reshaped.CopyFrom(src, shape));
        batch->push_back(reshaped);
      } else {
        batch->push_back(Tensor(dtype, shape));
      }
    }
    // If there is just one sample, 'batch' already has the copy.
    if (num_samples == 1) return Status::OK();

    Sharder::Do(num_samples /* total */, 1000 /* cost_per_unit */,
                [&](int64 start, int64 limit) {
                  for (int i = 0; i < num_outs; ++i) {
                    DataType dtype = padded_samples[0][i].dtype();
                    Tensor* merged = &(*batch)[i];
                    for (int j = start; j < limit; ++j) {
                      switch (dtype) {
#define CASE(T)                                                               \
  case DataTypeToEnum<T>::value:                                              \
    merged->flat_outer_dims<T>().chip<0>(j) = padded_samples[j][i].flat<T>(); \
    break
                        CASE(float);
                        CASE(int32);
                        CASE(int64);
                        CASE(tstring);
                        CASE(uint8);
                        CASE(bfloat16);
                        CASE(complex64);
                        CASE(complex128);
                        CASE(bool);
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
  NameAttrList func_;
  std::unique_ptr<ProcessorFn> processor_fn_;

  int num_merger_threads_ = -1;
  thread::ThreadPool* merger_ = nullptr;
  Runner merger_runner_;
  std::vector<int32> dynamic_padding_dimensions_;
  std::vector<int32> dynamic_padding_constants_;

  TF_DISALLOW_COPY_AND_ASSIGN(GenericInputProcessor);
};

REGISTER_KERNEL_BUILDER(Name("GenericInput").Device(DEVICE_CPU),
                        InputOp<GenericInputProcessor>);
}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
