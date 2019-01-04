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

#ifndef LINGVO_CORE_OPS_TOKENIZER_OP_HEADERS_H_
#define LINGVO_CORE_OPS_TOKENIZER_OP_HEADERS_H_

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace lingvo {
namespace {

template <typename TokenizerClass>
class LabelToTokenIdOp : public OpKernel {
 public:
  explicit LabelToTokenIdOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("append_eos", &append_eos_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("maxlen", &maxlen_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pad_to_maxlen", &pad_to_maxlen_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& labels = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels.shape()),
                errors::InvalidArgument("labels must be a vector, but get ",
                                        labels.shape().DebugString()));
    const int batch = labels.NumElements();
    auto Tlabels = labels.flat<string>();
    Tensor token_ids(DT_INT32, TensorShape({batch, maxlen_}));
    auto Ttoken_ids = token_ids.matrix<int32>();
    Ttoken_ids.setZero();  // Sanity
    Tensor target_ids(DT_INT32, TensorShape({batch, maxlen_}));
    auto Ttarget_ids = target_ids.matrix<int32>();
    Ttarget_ids.setZero();  // Sanity
    Tensor paddings(DT_FLOAT, TensorShape({batch, maxlen_}));
    auto Tpaddings = paddings.matrix<float>();
    Tpaddings.setZero();  // Sanity
    int actual_maxlen = pad_to_maxlen_ ? maxlen_ : 0;
    for (int i = 0; i < batch; ++i) {
      VLOG(1) << i << " " << Tlabels(i);
      std::vector<int32> ids = TokenizerClass::StringToIds(Tlabels(i));
      if (ids.size() + 1 > maxlen_) {
        LOG(WARNING) << "Too long target " << ids.size() << " " << Tlabels(i);
        ids.resize(maxlen_ - 1);
      }
      const int id_size = ids.size();
      const int32 kSOS = 1;
      const int32 kEOS = 2;
      Ttoken_ids(i, 0) = kSOS;
      for (int j = 0; j < id_size; ++j) {
        Ttoken_ids(i, j + 1) = ids[j];
        Ttarget_ids(i, j) = ids[j];
        Tpaddings(i, j) = 0.0;  // padding = false
      }
      Ttarget_ids(i, id_size) = kEOS;
      Tpaddings(i, id_size) = append_eos_ ? 0.0 : 1.0;
      actual_maxlen = std::max(actual_maxlen, id_size + 1);
      for (int j = id_size + 1; j < maxlen_; ++j) {
        Ttoken_ids(i, j) = kEOS;
        Ttarget_ids(i, j) = kEOS;
        Tpaddings(i, j) = 1.0;  // padding = true
      }
    }

    Tensor out_token_ids(DT_INT32, TensorShape({batch, actual_maxlen}));
    Tensor out_target_ids(DT_INT32, TensorShape({batch, actual_maxlen}));
    Tensor out_paddings(DT_FLOAT, TensorShape({batch, actual_maxlen}));

    typedef const Eigen::DSizes<Eigen::DenseIndex, 2> DSize2;
    out_token_ids.matrix<int32>() =
        Ttoken_ids.slice(DSize2{0, 0}, DSize2{batch, actual_maxlen});
    out_target_ids.matrix<int32>() =
        Ttarget_ids.slice(DSize2{0, 0}, DSize2{batch, actual_maxlen});
    out_paddings.matrix<float>() =
        Tpaddings.slice(DSize2{0, 0}, DSize2{batch, actual_maxlen});

    OP_REQUIRES_OK(ctx, ctx->set_output("token_ids", out_token_ids));
    OP_REQUIRES_OK(ctx, ctx->set_output("target_ids", out_target_ids));
    OP_REQUIRES_OK(ctx, ctx->set_output("paddings", out_paddings));
  }

 private:
  bool append_eos_ = true;
  int maxlen_ = 0;
  bool pad_to_maxlen_ = true;
};

template <typename TokenizerClass>
class IdToTokenOp : public OpKernel {
 public:
  explicit IdToTokenOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& ids = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(ids.shape()),
                errors::InvalidArgument("token_ids must be a matrix, but get ",
                                        ids.shape().DebugString()));
    const Tensor& seq_lens = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(seq_lens.shape()),
                errors::InvalidArgument("seq_lens must be a vector, but get ",
                                        seq_lens.shape().DebugString()));
    const int batch = seq_lens.NumElements();
    OP_REQUIRES(ctx, batch == ids.dim_size(0),
                errors::InvalidArgument(
                    "batch size has to match between token_ids and seq_lens. ",
                    ids.shape().DebugString(), " vs. ",
                    seq_lens.shape().DebugString()));

    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch}), &out));
    const auto& t_ids = ids.matrix<int32>();
    const auto& t_seq_lens = seq_lens.vec<int32>();
    auto t_out = out->template vec<string>();
    for (int i = 0; i < batch; ++i) {
      const int len_i = std::max(0, t_seq_lens(i));
      std::vector<int32> ids_i(len_i);
      for (int j = 0; j < len_i; ++j) {
        ids_i[j] = t_ids(i, j);
      }
      std::vector<string> labels = TokenizerClass::IdToStrings(ids_i);
      t_out(i) = TokenizerClass::JoinLabels(labels);
    }
  }
};

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_TOKENIZER_OP_HEADERS_H_
