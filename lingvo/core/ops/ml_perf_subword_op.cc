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

#include "lingvo/core/ops/ml_perf_subword_op.h"

#include "unicode/uchar.h"
#include "unicode/utf8.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace lingvo {

Status MlPerfSubword::Load(const string& vocab_glob) {
  std::vector<string> vocab_filenames;
  TF_CHECK_OK(Env::Default()->GetMatchingPaths(vocab_glob, &vocab_filenames))
      << "Unable to match vocab pattern: " << vocab_glob;
  CHECK_EQ(vocab_filenames.size(), 1)
      << "Did not match exactly one file with pattern: " << vocab_glob;
  const string& vocab_filename = vocab_filenames[0];

  string content;
  TF_RETURN_IF_ERROR(
      ReadFileToString(Env::Default(), vocab_filename, &content));

  return LoadLines(str_util::Split(content, '\n'));
}

Status MlPerfSubword::LoadLines(const std::vector<string>& lines) {
  for (StringPiece line : lines) {
    if (line.empty()) continue;
    // Strip surrounding single quotes.
    auto len = line.size();
    CHECK_GT(line.size(), 2);
    auto subtoken = string(line.substr(1, len - 2));
    id_to_token_.push_back(subtoken);
  }
  return Status::OK();
}

// This is a direct port of the tokenizer decode method in the MLPerf
// reference implementation for Translate/Transformer.
void MlPerfSubword::Decode(const std::vector<int32>& ids, string* out) {
  std::vector<string> subtokens_raw(ids.size());
  for (const auto& id : ids) {
    subtokens_raw.emplace_back(id_to_token_[id]);
  }
  string inter = absl::StrJoin(subtokens_raw, "");

  std::vector<std::string> subtokens = absl::StrSplit(inter, '_');

  std::vector<bool> token_is_alnum;
  for (const auto& token : subtokens) {
    int token_end = 0;
    UChar32 c;
    U8_NEXT(token, token_end, token.length(), c);
    token_is_alnum.push_back(u_isalnum(c));
  }
  std::vector<string> ret;

  for (int i = 0; i < subtokens.size(); ++i) {
    const auto& token = subtokens[i];
    if (i > 0 && token_is_alnum[i - 1] & token_is_alnum[i]) {
      ret.push_back(" ");
    }
    ret.push_back(token);
  }
  *out = absl::StrJoin(ret, "");
}

class MlPerfSubwordIdToStringOp : public OpKernel {
 public:
  explicit MlPerfSubwordIdToStringOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    string vocab_filepath;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_filepath", &vocab_filepath));
    OP_REQUIRES_OK(ctx, vocab_.Load(vocab_filepath));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* token_ids;
    const Tensor* seq_lengths;
    OP_REQUIRES_OK(ctx, ctx->input("token_ids", &token_ids));
    OP_REQUIRES_OK(ctx, ctx->input("seq_lengths", &seq_lengths));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(token_ids->shape()),
                errors::InvalidArgument("token_ids must be a matrix, but get ",
                                        token_ids->shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(seq_lengths->shape()),
        errors::InvalidArgument("seq_lengths must be a vector, but get ",
                                seq_lengths->shape().DebugString()));

    const int batch = seq_lengths->NumElements();
    OP_REQUIRES(
        ctx, batch == token_ids->dim_size(0),
        errors::InvalidArgument("batch size has to match between token_ids and "
                                "seq_lengths"));
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch}), &out));
    const auto& t_ids = token_ids->matrix<int32>();
    const auto& t_seq_lens = seq_lengths->vec<int32>();
    auto t_out = out->template vec<tstring>();

    for (int i = 0; i < batch; ++i) {
      const int len_i = std::max(0, t_seq_lens(i));
      std::vector<int32> ids_i(len_i);
      for (int j = 0; j < len_i; ++j) {
        ids_i[j] = t_ids(i, j);
      }
      string decode_output;
      vocab_.Decode(ids_i, &decode_output);
      t_out(i) = decode_output;
    }
  }

 private:
  MlPerfSubword vocab_;
};

REGISTER_KERNEL_BUILDER(Name("MlPerfSubwordIdToString").Device(DEVICE_CPU),
                        MlPerfSubwordIdToStringOp);

}  // namespace lingvo
}  // namespace tensorflow
