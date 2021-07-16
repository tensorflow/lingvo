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

#include "lingvo/core/ops/input_common.h"

#include <gtest/gtest.h>
#include "lingvo/core/ops/yielder_test_helper.h"
#include "tensorflow/core/lib/io/path.h"

namespace tensorflow {
namespace lingvo {

TEST(CreatePerFileYielderOptionsTest, SourceIdOffset) {
  const int OFFSET = 9;
  const int NUM_FILES = 3;

  BasicRecordYielder::Options opts_tpl;
  opts_tpl.source_id = OFFSET;

  std::vector<string> file_patterns;
  for (int i = 0; i < NUM_FILES; ++i) {
    file_patterns.push_back(strings::StrCat("file_", i));
  }

  std::vector<BasicRecordYielder::Options> per_file_options =
      CreatePerFileYielderOptions(file_patterns, opts_tpl);

  EXPECT_EQ(per_file_options[0].source_id, 0 + OFFSET);
  EXPECT_EQ(per_file_options[1].source_id, 1 + OFFSET);
  EXPECT_EQ(per_file_options[2].source_id, 2 + OFFSET);
}

}  // namespace lingvo
}  // namespace tensorflow
