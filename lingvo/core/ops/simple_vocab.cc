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

#include "lingvo/core/ops/simple_vocab.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace {

constexpr char kSosToken[] = "<s>";
constexpr char kEosToken[] = "</s>";
constexpr char kUnkToken[] = "<unk>";
constexpr char kSowToken[] = "<sow>";
constexpr char kEowToken[] = "<eow>";
constexpr char kSosTokenUpper[] = "<S>";
constexpr char kEosTokenUpper[] = "</S>";
constexpr char kUnkTokenUpper[] = "<UNK>";

}  // end namespace

namespace tensorflow {
namespace lingvo {

namespace debug {

static Vocab* vocab = nullptr;

void SetUpVocab(const string& vocab_filename) {
  if (vocab == nullptr) {
    vocab = new Vocab();
    TF_CHECK_OK(vocab->Load(vocab_filename));
  }
}

string IdsToStr(const std::vector<int32>& ids) {
  if (vocab != nullptr) {
    const std::vector<string> toks = vocab->IdsToTokens(ids);
    return str_util::Join(toks, " ");
  } else {
    return str_util::Join(ids, " ");
  }
}
}  // namespace debug

Status Vocab::Load(const string& vocab_glob, bool load_token_ids) {
  std::vector<string> vocab_filenames;
  TF_CHECK_OK(Env::Default()->GetMatchingPaths(vocab_glob, &vocab_filenames))
      << "Unable to match vocab pattern: " << vocab_glob;
  CHECK_EQ(vocab_filenames.size(), 1)
      << "Did not match exactly one file with pattern: " << vocab_glob;
  const string& vocab_filename = vocab_filenames[0];

  debug::SetUpVocab(vocab_filename);

  string content;
  TF_RETURN_IF_ERROR(
      ReadFileToString(Env::Default(), vocab_filename, &content));

  return Load(str_util::Split(content, '\n'), load_token_ids);
}

Status Vocab::Load(const std::vector<string>& lines, bool load_token_ids) {
  id_to_token_.clear();
  token_to_id_.clear();
  int32 next_id = 0;
  for (StringPiece line : lines) {
    if (line.empty()) continue;
    const std::vector<string> parts = str_util::Split(line, '\t');
    CHECK_GE(parts.size(), 1);
    const string tok = parts[0];
    if (!load_token_ids) {
      token_to_id_[tok] = next_id;
      id_to_token_[next_id] = tok;
      next_id++;
    } else {
      CHECK_GE(parts.size(), 2);
      const int32 id = std::stoi(parts[1]);
      token_to_id_[tok] = id;
      id_to_token_[id] = tok;
    }
    VLOG(2) << "Vocab " << token_to_id_[tok] << " " << tok;
  }
  use_upper_token_symbols_ = false;
  std::vector<string> expected_tokens = {kSosToken, kEosToken, kUnkToken};
  std::vector<string> unexpected_tokens = {kSosTokenUpper, kEosTokenUpper,
                                           kUnkTokenUpper};
  if (token_to_id_.find(sos_token()) == token_to_id_.end()) {
    use_upper_token_symbols_ = true;
    expected_tokens.swap(unexpected_tokens);
  }
  sos_id_ = token_to_id_[sos_token()];

  for (const auto& token : expected_tokens) {
    if (token_to_id_.find(token) == token_to_id_.end()) {
      return errors::InvalidArgument(token, " is not found in the vocab.");
    }
  }
  for (const auto& token : unexpected_tokens) {
    if (token_to_id_.find(token) != token_to_id_.end()) {
      return errors::InvalidArgument("Invalid token ", token,
                                     " is found in the vocab.");
    }
  }
  unk_id_ = -1;
  sos_id_ = TokenToId(sos_token());
  eos_id_ = TokenToId(eos_token());
  sow_id_ = TokenToId(sow_token());
  eow_id_ = TokenToId(eow_token());
  unk_id_ = TokenToId(unk_token());
  return Status::OK();
}

const char* Vocab::sos_token() const {
  return use_upper_token_symbols_ ? kSosTokenUpper : kSosToken;
}

const char* Vocab::eos_token() const {
  return use_upper_token_symbols_ ? kEosTokenUpper : kEosToken;
}

const char* Vocab::unk_token() const {
  return use_upper_token_symbols_ ? kUnkTokenUpper : kUnkToken;
}

const char* Vocab::sow_token() const { return kSowToken; }

const char* Vocab::eow_token() const { return kEowToken; }

namespace {

class VocabTokenToIdOp : public OpKernel {
 public:
  explicit VocabTokenToIdOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::vector<string> vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab", &vocab));
    bool load_token_ids_from_vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("load_token_ids_from_vocab",
                                     &load_token_ids_from_vocab));
    OP_REQUIRES_OK(ctx, vocab_.Load(vocab, load_token_ids_from_vocab));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* token;
    OP_REQUIRES_OK(ctx, ctx->input("token", &token));
    Tensor* id;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("id", token->shape(), &id));
    if (token->dims() == 0) {
      id->scalar<int32>()() = vocab_.TokenToId(token->scalar<string>()());
    } else {
      OP_REQUIRES(
          ctx, token->dims() == 1,
          errors::InvalidArgument("Input must be a scalar or 1D tensor."));
      for (int i = 0; i < token->dim_size(0); i++) {
        id->vec<int32>()(i) = vocab_.TokenToId(token->vec<string>()(i));
      }
    }
  }

 private:
  Vocab vocab_;
};

REGISTER_KERNEL_BUILDER(Name("VocabTokenToId").Device(DEVICE_CPU),
                        VocabTokenToIdOp);

class VocabIdToTokenOp : public OpKernel {
 public:
  explicit VocabIdToTokenOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::vector<string> vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab", &vocab));
    bool load_token_ids_from_vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("load_token_ids_from_vocab",
                                     &load_token_ids_from_vocab));
    OP_REQUIRES_OK(ctx, vocab_.Load(vocab, load_token_ids_from_vocab));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* id;
    OP_REQUIRES_OK(ctx, ctx->input("id", &id));
    Tensor* token;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("token", id->shape(), &token));
    if (id->dims() == 0) {
      token->scalar<string>()() = vocab_.IdToToken(id->scalar<int32>()());
    } else {
      OP_REQUIRES(
          ctx, id->dims() == 1,
          errors::InvalidArgument("Input must be a scalar or 1D tensor."));
      for (int i = 0; i < id->dim_size(0); i++) {
        token->vec<string>()(i) = vocab_.IdToToken(id->vec<int32>()(i));
      }
    }
  }

 private:
  Vocab vocab_;
};

REGISTER_KERNEL_BUILDER(Name("VocabIdToToken").Device(DEVICE_CPU),
                        VocabIdToTokenOp);

class TokenInVocabOp : public OpKernel {
 public:
  explicit TokenInVocabOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::vector<string> vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab", &vocab));
    bool load_token_ids_from_vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("load_token_ids_from_vocab",
                                     &load_token_ids_from_vocab));
    OP_REQUIRES_OK(ctx, vocab_.Load(vocab, load_token_ids_from_vocab));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* token;
    OP_REQUIRES_OK(ctx, ctx->input("token", &token));
    Tensor* result;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("result", token->shape(), &result));
    if (token->dims() == 0) {
      result->scalar<bool>()() = vocab_.InVocab(token->scalar<string>()());
    } else {
      OP_REQUIRES(
          ctx, token->dims() == 1,
          errors::InvalidArgument("Input must be a scalar or 1D tensor."));
      for (int i = 0; i < token->dim_size(0); i++) {
        result->vec<bool>()(i) = vocab_.InVocab(token->vec<string>()(i));
      }
    }
  }

 private:
  Vocab vocab_;
};

REGISTER_KERNEL_BUILDER(Name("TokenInVocab").Device(DEVICE_CPU),
                        TokenInVocabOp);

}  // namespace

}  // namespace lingvo
}  // namespace tensorflow
