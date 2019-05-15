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
#ifndef LINGVO_CORE_OPS_CHAIN_RECORD_YIELDER_H_
#define LINGVO_CORE_OPS_CHAIN_RECORD_YIELDER_H_

#include "lingvo/core/ops/record_yielder.h"

namespace tensorflow {
namespace lingvo {

// ChainRecordYielder is a special RecordYielder that consecutively yields
// examples based on a list of yielder options consuming those options in order.
//
// Usage example:
//   BasicRecordYielder::Options opts1;
//   opts1.file_pattern = <file_pattern>;
//   opts1.seed = 301;
//   opts1.bufsize = 1000000;    // A randomized buffer with 1M records.
//   opts1.parallelism = 8;      // Use 8 iterators.
//   BasicRecordYielder::Options opts2;
//   opts2.file_pattern = <file_pattern>;
//   opts2.seed = 301;
//   opts2.bufsize = 1000000;    // A randomized buffer with 1M records.
//   opts2.parallelism = 8;      // Use 8 iterators.
//   // Yield all records defined by opts1, then all records defined by opts2.
//   ChainRecordYielder* yielder = ChainRecordYielder::New({opts1, opts2});
//   Rope val;
//   while (true) {
//     yielder->Yield(&val);
//     // process val.
//   }
//   yielder->Close();
//
// ChainRecordYielder can be accessed by multiple threads concurrently.
class ChainRecordYielder : public RecordYielder {
 public:
  ~ChainRecordYielder() override;
  void Close() override;
  Status Yield(Rope* value, int* source_id) override;

  // Creates new ChainRecordYielder that will be creating child yielders using
  // options provided. Caller is responsible for closing the ChainRecordYielder
  // returned by this function. Caller should not delete the yielder as it will
  // be handled internally.
  static ChainRecordYielder* New(
      const std::vector<BasicRecordYielder::Options>& yielder_options);

 protected:
  ChainRecordYielder(
      const std::vector<BasicRecordYielder::Options>& yielder_options);

 private:
  mutable Mutex mu_;
  int current_yielder_idx_ GUARDED_BY(mu_);
  int64 current_yielder_epoch_ GUARDED_BY(mu_);
  BasicRecordYielder* current_yielder_ GUARDED_BY(mu_);

  std::vector<BasicRecordYielder::Options> yielder_options_;
};

}  // namespace lingvo
}  // namespace tensorflow
#endif  // LINGVO_CORE_OPS_CHAIN_RECORD_YIELDER_H_
