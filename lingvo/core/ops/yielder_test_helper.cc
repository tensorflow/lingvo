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
#include "lingvo/core/ops/yielder_test_helper.h"

#include "absl/flags/flag.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace lingvo {

void GeneratePlainTextTestData(const string& prefix, int n, int m) {
  for (int i = 0; i < n; ++i) {
    std::unique_ptr<WritableFile> file;
    TF_CHECK_OK(Env::Default()->NewWritableFile(
        io::JoinPath("/tmp",
                     strings::StrCat(prefix, ".", i)),
        &file));
    for (int j = 0; j < m; ++j) {
      TF_CHECK_OK(file->Append(
          strings::Printf("%s:%010d\n", prefix.c_str(), m * i + j)));
    }
  }
}

void GenerateCheckpointPlainTextTestData(const string& prefix, int m) {
  std::unique_ptr<WritableFile> ckpt_file;
  std::unique_ptr<WritableFile> data_file;
  TF_CHECK_OK(Env::Default()->NewWritableFile(
      io::JoinPath("/tmp", prefix), &ckpt_file));
  TF_CHECK_OK(ckpt_file->Append(
      strings::Printf("current: {file_pattern:\"data-0.txt\"}")));

  string data_file_path =
      io::JoinPath("/tmp", "data-0.txt");
  TF_CHECK_OK(Env::Default()->NewWritableFile(data_file_path, &data_file));
  for (int j = 0; j < m; ++j) {
    TF_CHECK_OK(data_file->Append(
        strings::Printf("%s:%010d\n", prefix.c_str(), m + j)));
  }
}

void UpdateCheckpointPlainTextTestData(const string& prefix, int m) {
  std::unique_ptr<WritableFile> ckpt_file;
  std::unique_ptr<WritableFile> data_file;
  TF_CHECK_OK(Env::Default()->NewWritableFile(
      io::JoinPath("/tmp", prefix), &ckpt_file));
  TF_CHECK_OK(ckpt_file->Append(
      strings::Printf("current: {file_pattern:\"data-1.txt\"}")));

  string data_file_path =
      io::JoinPath("/tmp", "data-1.txt");
  TF_CHECK_OK(Env::Default()->NewWritableFile(data_file_path, &data_file));
  for (int j = 0; j < m; ++j) {
    TF_CHECK_OK(data_file->Append(
        strings::Printf("%s:%010d\n", prefix.c_str(), 2 * m + j)));
  }
}

std::unordered_map<std::string, float> ComputeInputSourceDistribution(
    const std::vector<string>& vals) {
  std::unordered_map<std::string, float> input_source_distribution;
  for (const string& val : vals) {
    const auto prefix_end = val.find(':');
    if (prefix_end != string::npos) {
      input_source_distribution[val.substr(0, prefix_end)] += 1.0;
    }
  }
  for (auto it = input_source_distribution.begin();
       it != input_source_distribution.end(); ++it) {
    it->second /= vals.size();
  }
  return input_source_distribution;
}

}  // namespace lingvo
}  // namespace tensorflow
