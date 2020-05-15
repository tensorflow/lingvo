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

#include <algorithm>
#include <numeric>
#include <random>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lingvo {
namespace {

static const int kBOS = 1;
static const int kEOS = 2;

class MassOp : public OpKernel {
 public:
  explicit MassOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  float mask_ratio_;
  int mask_minlen_;
  int mask_id_;
  int span_len_;
  float random_start_prob_;
  float keep_prob_;
  float rand_prob_;
  float mask_prob_;
  bool mask_target_;
  int vocab_size_;
  int first_unreserved_id_;

  std::mt19937 rng_;

  // Populates a vector of length seq_len with 1's in positions where
  // the corresponding token will be masked, and 0's elsewhere.
  void GenerateMask(std::vector<int>* mask);
  void ValidateInput(OpKernelContext* ctx, const Tensor& ids,
                     const Tensor& weights, const Tensor& actual_seq_len);

  template <typename T>
  void CopyTensorToMutableOutput(const TensorShape& shape, const string& name,
                                 const Tensor& source, OpKernelContext* ctx,
                                 Tensor** out) {
    OP_REQUIRES_OK(ctx, ctx->allocate_output(name, shape, out));
    (*out)->flat<T>() = source.flat<T>();
  }
};

MassOp::MassOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("mask_ratio", &mask_ratio_));
  CHECK_GT(mask_ratio_, 0);
  CHECK_LT(mask_ratio_, 1);
  OP_REQUIRES_OK(ctx, ctx->GetAttr("mask_minlen", &mask_minlen_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("mask_id", &mask_id_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("span_len", &span_len_));
  CHECK_GT(span_len_, 0);
  OP_REQUIRES_OK(ctx, ctx->GetAttr("random_start_prob", &random_start_prob_));
  CHECK_GE(random_start_prob_, 0);
  CHECK_LE(random_start_prob_, 1);
  OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_prob", &keep_prob_));
  CHECK_GE(keep_prob_, 0);
  CHECK_LE(keep_prob_, 1);
  OP_REQUIRES_OK(ctx, ctx->GetAttr("rand_prob", &rand_prob_));
  CHECK_GE(rand_prob_, 0);
  CHECK_LE(rand_prob_, 1);
  OP_REQUIRES_OK(ctx, ctx->GetAttr("mask_prob", &mask_prob_));
  CHECK_GE(mask_prob_, 0);
  CHECK_LE(mask_prob_, 1);
  CHECK_EQ(keep_prob_ + rand_prob_ + mask_prob_, 1);
  OP_REQUIRES_OK(ctx, ctx->GetAttr("mask_target", &mask_target_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_size", &vocab_size_));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr("first_unreserved_id", &first_unreserved_id_));

  rng_.seed(7743);  // seed the random generator
}

void MassOp::ValidateInput(OpKernelContext* ctx, const Tensor& ids,
                           const Tensor& weights,
                           const Tensor& actual_seq_len) {
  // Verify shapes and sizes
  OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(ids.shape()),
              errors::InvalidArgument("ids must be matrix, but got ",
                                      ids.shape().DebugString()));
  OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(weights.shape()),
              errors::InvalidArgument("weights must be matrix, but got ",
                                      weights.shape().DebugString()));
  OP_REQUIRES(ctx, TensorShapeUtils::IsVector(actual_seq_len.shape()),
              errors::InvalidArgument("actual_seq_len must be vector, but got ",
                                      actual_seq_len.shape().DebugString()));
  OP_REQUIRES(ctx, ids.dim_size(0) > 0,
              errors::InvalidArgument("batch size must be > 0"));
  OP_REQUIRES(ctx, ids.dim_size(1) > 0,
              errors::InvalidArgument("max seq length must be > 0"));
  OP_REQUIRES(ctx, ids.dim_size(0) == weights.dim_size(0),
              errors::InvalidArgument("inconsistent batch size"));
  OP_REQUIRES(ctx, ids.dim_size(0) == actual_seq_len.dim_size(0),
              errors::InvalidArgument("inconsistent batch size"));
  OP_REQUIRES(ctx, ids.dim_size(1) == weights.dim_size(1),
              errors::InvalidArgument("inconsistent seq length"));
}

void MassOp::GenerateMask(std::vector<int>* mask) {
  int seq_len = mask->size();
  int mask_len = seq_len * mask_ratio_;

  // skip short sentences
  if (mask_len < mask_minlen_) return;

  int num_segments = std::ceil(static_cast<float>(mask_len) / span_len_);
  int non_mask_len = seq_len - mask_len;
  // segments = [span_len, span_len, ..., span_len]
  std::vector<int> segments(num_segments + non_mask_len, span_len_);
  // if there is a remainder, last segment will be a different length
  if (mask_len % span_len_ > 0) {
    // now segments are [span_len, span_len, ..., remaining_mask_len]
    segments.back() = mask_len % span_len_;
  }
  // non-masked segments of length 1 are represented by 0
  // now segments are [0, 0, ..., span_len, span_len, ... remaining_mask_len]
  std::fill(segments.begin(), segments.begin() + non_mask_len, 0);
  std::uniform_real_distribution<> realdistr(0.0, 1.0);
  float px = realdistr(rng_);
  float fixed_start_prob = 1 - random_start_prob_;
  if (px >= random_start_prob_ + fixed_start_prob / 2) {
    // place a segment at position 0, shuffle the rest
    std::swap(segments.front(), segments.back());
    if (num_segments > 1) {
      std::shuffle(segments.begin() + 1, segments.end(), rng_);
    }
  } else if (px >= random_start_prob_) {
    // keep a segment at final position, shuffle the rest
    if (num_segments > 1) {
      std::shuffle(segments.begin(), segments.end() - 1, rng_);
    }
  } else {
    // shuffle all segments
    std::shuffle(segments.begin(), segments.end(), rng_);
  }
  int idx = 0;
  for (int n : segments) {
    if (n == 0) {
      // Each 0 in segments represents a non-masked token; i.e. mask will be 0
      (*mask)[idx++] = 0;
    } else {
      // Each n in segments represents a masked segment of length n tokens
      for (int i = 0; i < n; i++) {
        (*mask)[idx++] = 1;
      }
    }
  }
}

void MassOp::Compute(OpKernelContext* ctx) {
  // get inputs
  const Tensor& ids = ctx->input(0);
  const Tensor& weights = ctx->input(1);
  const Tensor& actual_seq_len = ctx->input(2);
  auto Tactual_seq_len = actual_seq_len.vec<int32>();
  ValidateInput(ctx, ids, weights, actual_seq_len);
  OP_REQUIRES_OK(ctx, ctx->status());

  int batch_size = ids.dim_size(0);
  int max_seq_len = ids.dim_size(1);

  // create outputs
  Tensor* src_ids;
  Tensor* tgt_ids;
  Tensor* tgt_weights;
  Tensor* tgt_labels;
  CopyTensorToMutableOutput<int32>(TensorShape({batch_size, max_seq_len}),
                                   "src_ids", ids, ctx, &src_ids);
  CopyTensorToMutableOutput<int32>(TensorShape({batch_size, max_seq_len}),
                                   "tgt_ids", ids, ctx, &tgt_ids);
  CopyTensorToMutableOutput<float>(TensorShape({batch_size, max_seq_len}),
                                   "tgt_weights", weights, ctx, &tgt_weights);
  CopyTensorToMutableOutput<int32>(TensorShape({batch_size, max_seq_len}),
                                   "tgt_labels", ids, ctx, &tgt_labels);
  auto Tsrc_ids = src_ids->matrix<int32>();
  auto Ttgt_ids = tgt_ids->matrix<int32>();
  auto Ttgt_weights = tgt_weights->matrix<float>();

  // for each example, mask the source and target to implement MASS.
  for (int i = 0; i < batch_size; i++) {
    int seq_len = Tactual_seq_len(i);
    if (seq_len <= 0) {
      LOG(WARNING) << "Skipping zero-length sequence";
      continue;
    }
    if (seq_len > max_seq_len) seq_len = max_seq_len;
    std::vector<int> mask(seq_len, 0);
    GenerateMask(&mask);

    // Right shift tgt and drop EOS
    for (int j = max_seq_len - 1; j > 0; j--) {
      if (Ttgt_ids(i, j - 1) == kEOS) {
        Ttgt_ids(i, j) = 0;
        Ttgt_weights(i, j) = 0.0;
      } else {
        Ttgt_ids(i, j) = Ttgt_ids(i, j - 1);
        Ttgt_weights(i, j) = Ttgt_weights(i, j - 1);
      }
    }
    Ttgt_ids(i, 0) = kBOS;
    Ttgt_weights(i, 0) = 1.0;

    std::uniform_int_distribution<> vocabdistr(first_unreserved_id_,
                                               vocab_size_);

    // mask source and target tokens.
    if (std::accumulate(mask.begin(), mask.end(), 0) > 0) {
      for (int j = 0; j < mask.size(); j++) {
        if (mask[j] == 1) {
          std::uniform_real_distribution<> realdistr(0.0, 1.0);
          float tx = realdistr(rng_);
          if (tx >= mask_prob_ + rand_prob_) {
            // Keep token
          } else if (tx >= mask_prob_) {
            Tsrc_ids(i, j) = vocabdistr(rng_);  // replace with random token
          } else {
            Tsrc_ids(i, j) = mask_id_;  // replace with mask id
          }
        } else if (mask_target_) {
          // either source is masked or target is masked.
          Ttgt_ids(i, j) = mask_id_;
          // where target is masked, target weights are zero
          Ttgt_weights(i, j) = 0;
        }
      }
    }
  }
}

REGISTER_KERNEL_BUILDER(Name("Mass").Device(DEVICE_CPU), MassOp);

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
