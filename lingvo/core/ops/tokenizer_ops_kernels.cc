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
#include "lingvo/core/ops/tokenizer_op_headers.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "lingvo/core/ops/ascii_tokenizer.h"
#include "lingvo/core/ops/simple_vocab.h"

namespace tensorflow {
namespace lingvo {
namespace {

REGISTER_KERNEL_BUILDER(Name("AsciiToTokenId").Device(DEVICE_CPU),
                        LabelToTokenIdOp<AsciiTokenizer>);

REGISTER_KERNEL_BUILDER(Name("IdToAscii").Device(DEVICE_CPU),
                        IdToTokenOp<AsciiTokenizer>);

class StrToVocabTokensOp : public OpKernel {
 public:
  explicit StrToVocabTokensOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("append_eos", &append_eos_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("maxlen", &maxlen_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pad_to_maxlen", &pad_to_maxlen_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_filepath", &vocab_filepath_));
    bool load_token_ids_from_vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("load_token_ids_from_vocab",
                                     &load_token_ids_from_vocab));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("delimiter", &delimiter_));
    CHECK_GT(maxlen_, 0);
    OP_REQUIRES_OK(ctx,
                   vocab_.Load(vocab_filepath_, load_token_ids_from_vocab));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* labels;
    OP_REQUIRES_OK(ctx, ctx->input("labels", &labels));
    const auto& t_label = labels->vec<string>();
    const int32 b_size = labels->dim_size(0);
    Tensor token_ids(DT_INT32, TensorShape({b_size, maxlen_}));
    Tensor target_ids(DT_INT32, TensorShape({b_size, maxlen_}));
    Tensor paddings(DT_FLOAT, TensorShape({b_size, maxlen_}));

    auto t_token_ids = token_ids.tensor<int32, 2>();
    auto t_target_ids = target_ids.tensor<int32, 2>();
    auto t_paddings = paddings.tensor<float, 2>();
    t_token_ids.setZero();
    t_target_ids.setZero();
    t_paddings.setZero();

    int actual_maxlen = pad_to_maxlen_ ? maxlen_ : 0;
    for (int i = 0; i < b_size; ++i) {
      t_token_ids(i, 0) = vocab_.sos_id();

      string label(t_label(i));
      VLOG(1) << "Label " << label;
      std::vector<string> tokens;
      if (delimiter_.length() > 0) {
        tokens = str_util::Split(label, delimiter_, str_util::SkipWhitespace());
      } else {
        // Split by the empty delimiter.
        for (int i = 0; i < label.size(); ++i) {
          tokens.push_back(string(1, label[i]));
        }
      }

      VLOG(1) << "#Tokens " << tokens.size() << " "
              << str_util::Join(tokens, "/");
      int cur_char = 0;
      for (const auto& token : tokens) {
        const int token_id = vocab_.TokenToId(token);
        t_target_ids(i, cur_char) = token_id;
        t_paddings(i, cur_char) = 0.0;
        // If the number of tokens is longer than the max length - truncate.
        if (cur_char + 1 >= maxlen_) {
          cur_char++;
          LOG(INFO) << "Label: \"" << label << "\" contained " << tokens.size()
                    << " tokens, and was truncated to size: " << maxlen_ << " ("
                    << tokens.size() - maxlen_ << " tokens were ignored).";
          break;
        }
        t_token_ids(i, cur_char + 1) = token_id;
        cur_char++;
      }
      if (cur_char < maxlen_) {
        // There was no truncation, t_token_ids is ahead by 1 over t_target_ids
        // and t_paddings
        t_target_ids(i, cur_char) = vocab_.eos_id();
        t_paddings(i, cur_char) = append_eos_ ? 0.0 : 1.0;
        ++cur_char;
      }
      actual_maxlen = std::max(actual_maxlen, cur_char);
      for (; cur_char < maxlen_; ++cur_char) {
        t_token_ids(i, cur_char) = vocab_.eos_id();
        t_target_ids(i, cur_char) = vocab_.eos_id();
        t_paddings(i, cur_char) = 1.0;
      }
    }

    Tensor out_token_ids(DT_INT32, TensorShape({b_size, actual_maxlen}));
    Tensor out_target_ids(DT_INT32, TensorShape({b_size, actual_maxlen}));
    Tensor out_paddings(DT_FLOAT, TensorShape({b_size, actual_maxlen}));

    typedef const Eigen::DSizes<Eigen::DenseIndex, 2> DSize2;
    out_token_ids.matrix<int32>() =
        t_token_ids.slice(DSize2{0, 0}, DSize2{b_size, actual_maxlen});
    out_target_ids.matrix<int32>() =
        t_target_ids.slice(DSize2{0, 0}, DSize2{b_size, actual_maxlen});
    out_paddings.matrix<float>() =
        t_paddings.slice(DSize2{0, 0}, DSize2{b_size, actual_maxlen});

    OP_REQUIRES_OK(ctx, ctx->set_output("token_ids", out_token_ids));
    OP_REQUIRES_OK(ctx, ctx->set_output("target_ids", out_target_ids));
    OP_REQUIRES_OK(ctx, ctx->set_output("paddings", out_paddings));
  }

 private:
  string vocab_filepath_;
  bool append_eos_ = true;
  int maxlen_ = 0;
  bool pad_to_maxlen_ = true;
  string delimiter_;
  Vocab vocab_;
};

REGISTER_KERNEL_BUILDER(Name("StrToVocabTokens").Device(DEVICE_CPU),
                        StrToVocabTokensOp);

class NgramIdToTokenOp : public OpKernel {
 public:
  explicit NgramIdToTokenOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ngram_vocab_filepath", &vocab_filepath_));
    OP_REQUIRES_OK(ctx, vocab_.Load(vocab_filepath_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ngram_separator", &ngram_separator_));
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
    auto t_out = out->template vec<string>();
    for (int i = 0; i < batch; ++i) {
      const int len_i = std::max(0, t_seq_lens(i));
      std::vector<int32> ids_i(len_i);
      for (int j = 0; j < len_i; ++j) {
        ids_i[j] = t_ids(i, j);
      }
      auto labels = vocab_.IdsToTokens(ids_i);
      t_out(i) = str_util::Join(labels, ngram_separator_.c_str());
    }
  }

 private:
  string vocab_filepath_;
  Vocab vocab_;
  string ngram_separator_;
};

REGISTER_KERNEL_BUILDER(Name("NgramIdToToken").Device(DEVICE_CPU),
                        NgramIdToTokenOp);

class BpeIdsToWordsOp : public OpKernel {
 public:
  explicit BpeIdsToWordsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_filepath", &vocab_filepath_));
    string contents;
    OP_REQUIRES_OK(ctx, ReadFileToString(
        Env::Default(), vocab_filepath_, &contents));
    std::vector<string> lines = str_util::Split(contents, '\n',
                                                str_util::SkipEmpty());
    for (const string& line : lines) {
      std::vector<string> parts = str_util::Split(line, ' ');
      id_to_string_map_.push_back(parts[0]);
    }
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
    auto t_out = out->template vec<string>();
    for (int i = 0; i < batch; ++i) {
      const int len_i = std::max(0, t_seq_lens(i));
      std::vector<string> labels;
      for (int j = 0; j < len_i; ++j) {
        string label = id_to_string_map_[t_ids(i, j)];
        std::size_t pos = label.find("@@");
        if (pos == std::string::npos)
            label = label + " ";
        else
            label.erase(pos, 2);
        labels.push_back(label);
      }
      t_out(i) = str_util::Join(labels, "");
    }
  }

 private:
  string vocab_filepath_;
  std::vector<string> id_to_string_map_;
};

REGISTER_KERNEL_BUILDER(Name("BpeIdsToWords").Device(DEVICE_CPU),
                        BpeIdsToWordsOp);

class BpeWordsToIdsOp : public OpKernel {
 public:
  explicit BpeWordsToIdsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("append_eos", &append_eos_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("maxlen", &maxlen_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tokenization_filepath",
                                     &tokenization_filepath_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sos_id", &sos_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eos_id", &eos_id_));
    CHECK_GT(maxlen_, 0);
    string contents;
    OP_REQUIRES_OK(ctx, ReadFileToString(Env::Default(), tokenization_filepath_,
                                         &contents));
    std::vector<string> lines = str_util::Split(contents, '\n',
                                               str_util::SkipEmpty());
    for (const string& line : lines) {
      // Each line:
      // string int1,int2,int3,...,intn
      std::vector<string> parts = str_util::Split(line, ' ');
      std::vector<int32> ids;
      str_util::SplitAndParseAsInts(parts[1], ',', &ids);
      string_to_ids_map_[parts[0]] = ids;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* labels;
    OP_REQUIRES_OK(ctx, ctx->input("labels", &labels));
    const auto& t_label = labels->vec<string>();
    const int32 b_size = labels->dim_size(0);
    Tensor* token_ids = nullptr;
    Tensor* target_ids = nullptr;
    Tensor* paddings = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("token_ids", TensorShape({b_size, maxlen_}),
                                  &token_ids));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("target_ids", TensorShape({b_size, maxlen_}),
                                  &target_ids));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("paddings", TensorShape({b_size, maxlen_}),
                                  &paddings));

    token_ids->flat<int32>().setConstant(eos_id_);
    target_ids->flat<int32>().setConstant(eos_id_);
    paddings->flat<float>().setConstant(1.0f);
    auto t_token_ids = token_ids->tensor<int32, 2>();
    auto t_target_ids = target_ids->tensor<int32, 2>();
    auto t_paddings = paddings->tensor<float, 2>();

    for (int i = 0; i < b_size; ++i) {
      t_token_ids(i, 0) = sos_id_;

      string label(t_label(i));
      VLOG(1) << "Label " << label;
      std::vector<string> tokens =
          str_util::Split(label, ' ', str_util::SkipWhitespace());
      VLOG(1) << "#Tokens " << tokens.size() << " "
              << str_util::Join(tokens, "/");
      int cur_char = 0;
      for (const auto& token : tokens) {
        if (cur_char >= maxlen_) {
          break;
        }
        const std::vector<int32> token_ids = string_to_ids_map_[token];
        for (const auto& token_id : token_ids) {
          t_target_ids(i, cur_char) = token_id;
          t_paddings(i, cur_char) = 0.0f;
          // If the number of tokens is longer than the max length - truncate.
          if (cur_char + 1 >= maxlen_) {
            cur_char++;
            int num_token_ids = 0;
            for (const auto& t : tokens) {
              num_token_ids += string_to_ids_map_[t].size();
            }
            LOG(INFO) << "Label: \"" << label << "\" had " << num_token_ids
                      << " tokens, and was truncated to size: " << maxlen_
                      << " (" << num_token_ids - maxlen_ << " tokens ignored).";
            break;
          }
          t_token_ids(i, cur_char + 1) = token_id;
          cur_char++;
        }
      }
      if (cur_char < maxlen_) {
        // There was no truncation, t_token_ids is ahead by 1 over t_target_ids
        // and t_paddings
        t_target_ids(i, cur_char) = eos_id_;
        t_paddings(i, cur_char) = append_eos_ ? 0.0f : 1.0f;
      }
    }
  }

 private:
  string tokenization_filepath_;
  bool append_eos_ = true;
  int maxlen_ = 0;
  int sos_id_ = 1;
  int eos_id_ = 2;
  std::unordered_map<string, std::vector<int32> > string_to_ids_map_;
};

REGISTER_KERNEL_BUILDER(Name("BpeWordsToIds").Device(DEVICE_CPU),
                        BpeWordsToIdsOp);
}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
