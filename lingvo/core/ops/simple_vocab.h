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

#ifndef LINGVO_CORE_OPS_SIMPLE_VOCAB_H_
#define LINGVO_CORE_OPS_SIMPLE_VOCAB_H_
// TODO(zhifengc): Add comments for this class.

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace lingvo {

class Vocab {
 public:
  Vocab() {}

  ~Vocab() {}

  Status Load(const string& vocab_filename, bool load_token_ids = false);
  Status Load(const std::vector<string>& lines, bool load_token_ids = false);

  int32 sos_id() const { return sos_id_; }
  int32 eos_id() const { return eos_id_; }
  int32 unk_id() const { return unk_id_; }
  int32 sow_id() const { return sow_id_; }
  int32 eow_id() const { return eow_id_; }

  const char* sos_token() const;
  const char* eos_token() const;
  const char* unk_token() const;
  const char* sow_token() const;
  const char* eow_token() const;

  bool InVocab(const string& tok) const { return token_to_id_.count(tok) > 0; }

  int32 TokenToId(const string& tok) const {
    auto it = token_to_id_.find(tok);
    if (it != token_to_id_.end()) return it->second;
    return unk_id_;
  }

  void GreedyMatchStringToTokenId(StringPiece text, int32* token_id,
                                  int* token_size) const {
    // This finds the longest prefix of the "text" in the given list of tokens
    // and returns the ID of the found token (if nothing found, unk_id_ is
    // returned) and the length of the found token through input argument
    // pointers.
    *token_id = unk_id_;
    *token_size = 1;  // For <unk>, the input is of length 1 char, but output is
                      // <unk> (length of 5).
    for (const auto kv : id_to_token_) {
      if (str_util::StartsWith(text, kv.second)) {
        // Find the longest matching token.
        if (*token_id == unk_id_ || *token_size < kv.second.size()) {
          *token_id = kv.first;
          *token_size = kv.second.size();
        }
      }
    }
  }

  std::vector<int32> TokensToIds(const std::vector<string>& toks) const {
    std::vector<int32> ids;
    ids.reserve(toks.size());
    for (const string& tok : toks) {
      ids.push_back(TokenToId(tok));
    }
    return ids;
  }

  const string IdToToken(const int32 id) const {
    const auto it = id_to_token_.find(id);
    if (it != id_to_token_.end()) {
      return it->second;
    } else {
      return unk_token();
    }
  }

  std::vector<string> IdsToTokens(const std::vector<int32>& ids) const {
    std::vector<string> toks;
    toks.reserve(ids.size());
    for (const int32 id : ids) {
      toks.push_back(IdToToken(id));
    }
    return toks;
  }

 private:
  int32 sos_id_ = -1;
  int32 eos_id_ = -1;
  int32 unk_id_ = -1;
  int32 sow_id_ = -1;
  int32 eow_id_ = -1;
  bool use_upper_token_symbols_ = false;
  std::unordered_map<int32, string> id_to_token_;
  std::unordered_map<string, int32> token_to_id_;

  TF_DISALLOW_COPY_AND_ASSIGN(Vocab);
};

namespace debug {
string IdsToStr(const std::vector<int32>& ids);
}  // namespace debug

}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_SIMPLE_VOCAB_H_
