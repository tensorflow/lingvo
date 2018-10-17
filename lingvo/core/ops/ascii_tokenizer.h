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

#ifndef LINGVO_CORE_OPS_ASCII_TOKENIZER_H_
#define LINGVO_CORE_OPS_ASCII_TOKENIZER_H_

#include <string>
#include <vector>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace lingvo {

// A static simple tokenizer that maps a small vocabulary of character
// tokens for (lower case) letters, digits, and punctuation symbols.
class AsciiTokenizer {
 public:
  // Convert string to a set of graphemes.
  static string ConvertString(const string& transcript);

  // Returns the number of tokens.
  static int32 NumTokens();

  // Splits 'label' into tokens and returns their token ids.
  static std::vector<int32> StringToIds(const string& label);

  // Convert 'ids' back into tokens.
  static std::vector<string> IdToStrings(const std::vector<int32>& ids);

  // Joins the token labels into a string.
  static string JoinLabels(const std::vector<string>& labels);
};

}  // namespace lingvo
}  // namespace tensorflow
#endif  // LINGVO_CORE_OPS_ASCII_TOKENIZER_H_
