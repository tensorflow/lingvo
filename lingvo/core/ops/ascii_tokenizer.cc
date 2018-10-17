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

#include "lingvo/core/ops/ascii_tokenizer.h"

#include <algorithm>
#include <string>
#include <unordered_map>

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace lingvo {
namespace {

const string& FindOrDie(const std::unordered_map<int32, string>& m, int32 k) {
  const auto it = m.find(k);
  CHECK(it != m.end());
  return it->second;
}

const int32 kUnkId = 0;
const int32 kSOSId = 1;
const int32 kEOSId = 2;
const int32 kEOWId = 3;
const int32 kNoiseId = 4;
const int32 kEpsilonId = 73;
const int32 kTextOnlyId = 74;
const int32 kSORWId = 75;  // indicator for start of rare word
const int32 kMaxTokenId = 75;

struct CharTokenizer {
  std::unordered_map<string, int32> token_to_id;
  std::unordered_map<int32, string> id_to_token;

  string epsilon_token;
  string unk_token;
  string noise_token;
  string eos_token;
  string sos_token;
  string text_only_token;
  string sorw_token;

  string IdToToken(int32 id) const {
    const auto it = id_to_token.find(id);
    if (it != id_to_token.end()) return it->second;
    return unk_token;
  }

  int32 TokenToId(const string& tok) const {
    const auto it = token_to_id.find(tok);
    if (it != token_to_id.end()) return it->second;
    return kUnkId;
  }
};

const CharTokenizer* CreateTokenizer() {
  CharTokenizer* ct = new CharTokenizer();
  ct->id_to_token = {{0, "<unk>"},  {1, "<s>"},        {2, "</s>"},
                     {3, " "},      {4, "<noise>"},    {5, "a"},
                     {6, "b"},      {7, "c"},          {8, "d"},
                     {9, "e"},      {10, "f"},         {11, "g"},
                     {12, "h"},     {13, "i"},         {14, "j"},
                     {15, "k"},     {16, "l"},         {17, "m"},
                     {18, "n"},     {19, "o"},         {20, "p"},
                     {21, "q"},     {22, "r"},         {23, "s"},
                     {24, "t"},     {25, "u"},         {26, "v"},
                     {27, "w"},     {28, "x"},         {29, "y"},
                     {30, "z"},     {31, "."},         {32, "\'"},
                     {33, "-"},     {34, ":"},         {35, "!"},
                     {36, "~"},     {37, "`"},         {38, ";"},
                     {39, "0"},     {40, "1"},         {41, "2"},
                     {42, "3"},     {43, "4"},         {44, "5"},
                     {45, "6"},     {46, "7"},         {47, "8"},
                     {48, "9"},     {49, "\""},        {50, "#"},
                     {51, "$"},     {52, "%"},         {53, "&"},
                     {54, "("},     {55, ")"},         {56, "*"},
                     {57, "+"},     {58, ","},         {59, "/"},
                     {60, "<"},     {61, "="},         {62, ">"},
                     {63, "?"},     {64, "@"},         {65, "["},
                     {66, "\\"},    {67, "]"},         {68, "^"},
                     {69, "_"},     {70, "{"},         {71, "|"},
                     {72, "}"},     {73, "<epsilon>"}, {74, "<text_only>"},
                     {75, "<sorw>"}};
  // kEpsilonWord: end-of-block for neural transducer.
  for (const std::pair<const int32, string>& p : ct->id_to_token) {
    CHECK_LE(p.first, kMaxTokenId);
    CHECK(ct->token_to_id.insert({p.second, p.first}).second);
  }
  ct->unk_token = FindOrDie(ct->id_to_token, kUnkId);
  ct->noise_token = FindOrDie(ct->id_to_token, kNoiseId);
  ct->epsilon_token = FindOrDie(ct->id_to_token, kEpsilonId);
  ct->sos_token = FindOrDie(ct->id_to_token, kSOSId);
  ct->eos_token = FindOrDie(ct->id_to_token, kEOSId);
  ct->sorw_token = FindOrDie(ct->id_to_token, kSORWId);
  ct->text_only_token = FindOrDie(ct->id_to_token, kTextOnlyId);
  return ct;
}

const CharTokenizer* GetTokenizer() {
  static const CharTokenizer* tokenizer = CreateTokenizer();
  return tokenizer;
}

}  // namespace

string AsciiTokenizer::ConvertString(const string& transcript) {
  string result = transcript;
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  return result;
}

int32 AsciiTokenizer::NumTokens() { return kMaxTokenId + 1; }

std::vector<int32> AsciiTokenizer::StringToIds(const string& label) {
  const CharTokenizer* tokenizer = GetTokenizer();
  const string converted = ConvertString(label);
  const StringPiece converted_view(converted);

  std::vector<int32> ids;
  const std::vector<std::pair<string, const int32>> special_token_ids{
      {tokenizer->unk_token, kUnkId},
      {tokenizer->noise_token, kNoiseId},
      {tokenizer->sos_token, kSOSId},
      {tokenizer->eos_token, kEOSId},
      {tokenizer->epsilon_token, kEpsilonId},
      {tokenizer->text_only_token, kTextOnlyId},
      {tokenizer->sorw_token, kSORWId},
  };

  for (int i = 0; i < converted.size(); ++i) {
    bool is_special_token = false;
    for (const auto& token_id : special_token_ids) {
      if (str_util::StartsWith(converted_view.substr(i), token_id.first)) {
        ids.push_back(token_id.second);
        i += token_id.first.size() - 1;
        is_special_token = true;
        break;
      }
    }
    if (!is_special_token) {
      ids.push_back(tokenizer->TokenToId(string(1, converted[i])));
    }
  }
  return ids;
}

std::vector<string> AsciiTokenizer::IdToStrings(const std::vector<int32>& ids) {
  const CharTokenizer* tokenizer = GetTokenizer();
  std::vector<string> out_strings(ids.size());
  for (int i = 0; i < ids.size(); ++i) {
    out_strings[i] = tokenizer->IdToToken(ids[i]);
  }
  return out_strings;
}

string AsciiTokenizer::JoinLabels(const std::vector<string>& labels) {
  return str_util::Join(labels, "");
}

}  // namespace lingvo
}  // namespace tensorflow
