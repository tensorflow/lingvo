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
#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace lingvo {
namespace {

template <typename T>
void Iota(std::vector<T>* vec) {
  std::iota(vec->begin(), vec->end(), T());
}

template <>
void Iota<tstring>(std::vector<tstring>* vec) {
  // Do nothing.
}

template <typename K, typename V>
class StaticMapOp : public OpKernel {
 public:
  explicit StaticMapOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::vector<K> keys;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keys", &keys));
    std::vector<V> vals;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vals", &vals));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unk", &unk_));

    if (keys.empty() && !vals.empty()) {
      keys.resize(vals.size());
      Iota(&keys);
    }

    if (!keys.empty() && vals.empty()) {
      vals.resize(keys.size());
      Iota(&vals);
    }

    OP_REQUIRES(ctx, keys.size() == vals.size(),
                errors::InvalidArgument("keys and vals are different sizes: ",
                                        keys.size(), " / ", vals.size()));

    for (int i = 0; i < keys.size(); ++i) {
      OP_REQUIRES(ctx, dict_.insert({keys[i], vals[i]}).second,
                  errors::InvalidArgument("keys have duplicates: ", keys[i]));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    Tensor* y;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

    auto tx = x.flat<K>();
    auto ty = y->flat<V>();
    int total = tx.size();
    auto workers = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, total, 250,
          [this, &tx, &ty](int64 start, int64 limit) {
            for (int64 i = start; i < limit; ++i) {
              auto it = dict_.find(tx(i));
              if (it != dict_.end()) {
                ty(i) = it->second;
              } else {
                ty(i) = unk_;
              }
            }
          });
  }

 protected:
  tensorflow::gtl::FlatMap<K, V> dict_;
  V unk_;
};

REGISTER_KERNEL_BUILDER(Name("StaticMapStringInt").Device(DEVICE_CPU),
                        StaticMapOp<tstring, int32>);
REGISTER_KERNEL_BUILDER(Name("StaticMapIntString").Device(DEVICE_CPU),
                        StaticMapOp<int32, tstring>);
REGISTER_KERNEL_BUILDER(Name("StaticMapIntInt").Device(DEVICE_CPU),
                        StaticMapOp<int32, int32>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("StaticMapStringInt")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y"),
                        StaticMapOp<tstring, int32>);

REGISTER_KERNEL_BUILDER(Name("StaticMapIntString")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y"),
                        StaticMapOp<int32, tstring>);

REGISTER_KERNEL_BUILDER(
    Name("StaticMapIntInt").Device(DEVICE_GPU).HostMemory("x").HostMemory("y"),
    StaticMapOp<int32, int32>);
#endif

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
