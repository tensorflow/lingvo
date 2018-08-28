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
  id_to_token_.clear();
  token_to_id_.clear();
  for (StringPiece line : str_util::Split(content, '\n')) {
    if (line.empty()) continue;
    const std::vector<string> parts = str_util::Split(line, '\t');
    CHECK_GE(parts.size(), 1);
    const string tok = parts[0];
    VLOG(2) << "Vocab " << id_to_token_.size() << " " << tok;
    if (!load_token_ids) {
      token_to_id_[tok] = id_to_token_.size();
    } else {
      CHECK_GE(parts.size(), 2);
      token_to_id_[tok] = std::stoi(parts[1]);
    }
    id_to_token_.push_back(tok);
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
      return errors::InvalidArgument(
          token, " is not found in the vocab file: ", vocab_filename);
    }
  }
  for (const auto& token : unexpected_tokens) {
    if (token_to_id_.find(token) != token_to_id_.end()) {
      return errors::InvalidArgument(
          "Invalid token ", token,
          " is found in the vocab file: ", vocab_filename);
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

}  // namespace lingvo

}  // namespace tensorflow
