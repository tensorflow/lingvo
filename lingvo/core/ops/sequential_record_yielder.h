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
#ifndef LINGVO_CORE_OPS_SEQUENTIAL_RECORD_YIELDER_H_
#define LINGVO_CORE_OPS_SEQUENTIAL_RECORD_YIELDER_H_

#include "lingvo/core/ops/record_yielder.h"

namespace tensorflow {
namespace lingvo {

// SequentialRecordYielder processes records in order.
class SequentialRecordYielder : public RecordYielder {
 public:
  ~SequentialRecordYielder() override;
  void Close() override;
  Status Yield(Rope* value, int* source_id) override;

  // Returns a sequential record yielder. The caller is responsible for calling
  // Close when this yielder is no longer required. The caller shouldn't delete
  // the yielder.
  static SequentialRecordYielder* New(const string& file_pattern);

 protected:
  explicit SequentialRecordYielder(const string& file_pattern);

 private:
  const string file_type_;
  std::vector<string> filenames_;
  int cur_file_index_ = 0;
  std::unique_ptr<RecordIterator> record_iterator_;
};

}  // namespace lingvo
}  // namespace tensorflow
#endif  // LINGVO_CORE_OPS_SEQUENTIAL_RECORD_YIELDER_H_
