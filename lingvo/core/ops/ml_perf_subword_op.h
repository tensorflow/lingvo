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

#ifndef THIRD_PARTY_PY_LINGVO_CORE_OPS_ML_PERF_SUBWORD_OP_H_
#define THIRD_PARTY_PY_LINGVO_CORE_OPS_ML_PERF_SUBWORD_OP_H_

#include <string>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace lingvo {

// Subword vocabulary class.
class MlPerfSubword {
 public:
  MlPerfSubword() {}
  ~MlPerfSubword() {}

  Status Load(const string& vocab_glob);
  Status LoadLines(const std::vector<string>& lines);

  void Decode(const std::vector<int32>& ids, string* out);

 private:
  std::vector<string> id_to_token_;
};

}  // namespace lingvo
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_LINGVO_CORE_OPS_MLPERF_SUBWORD_OP_H
