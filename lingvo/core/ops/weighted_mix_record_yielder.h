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
#ifndef LINGVO_CORE_OPS_WEIGHTED_MIX_RECORD_YIELDER_H_
#define LINGVO_CORE_OPS_WEIGHTED_MIX_RECORD_YIELDER_H_

#include "lingvo/core/ops/record_yielder.h"

namespace tensorflow {
namespace lingvo {

// WeightedMixRecordYielder is a special RecordYielder that mixes examples from
// yielders in a random order maintaining the mix ratio specified by
// input_source_weights.
//
// Usage example:
//   BasicRecordYielder::Options opts1;
//   opts1.file_pattern = <file_pattern>;
//   opts1.seed = 301;
//   opts1.bufsize = 1000000;    // A randomized buffer with 1M records.
//   opts1.parallelism = 8;      // Use 8 iterators.
//   RecordYielder* yielder1 = BasicRecordYielder::New(opts1);
//   BasicRecordYielder::Options opts2;
//   opts2.file_pattern = <file_pattern>;
//   opts2.seed = 301;
//   opts2.bufsize = 1000000;    // A randomized buffer with 1M records.
//   opts2.parallelism = 8;      // Use 8 iterators.
//   RecordYielder* yielder2 = BasicRecordYielder::New(opts2);
//   // Use records from yielder1 for 70% of the yields.
//   WeightedMixRecordYielder* yielder = WeightedMixRecordYielder::New(
//       opts1.seed, {yielder1, yielder2}, {0.7, 0.3});
//   Rope val;
//   while (true) {
//     yielder->Yield(&val);
//     // process val.
//   }
//   yielder->Close();
//
// WeightedMixRecordYielder can be accessed by multiple threads concurrently.
class WeightedMixRecordYielder : public RecordYielder {
 public:
  ~WeightedMixRecordYielder() override;
  void Close() override;
  Status Yield(Rope* value, int* source_id) override;

  // Creates new WeightedMixRecordYielder and takes ownership over yielders
  // provided. Those yielders should be properly initialized already and will be
  // closed once WeightedMixRecordYielder is closed. Caller is responsible
  // closing the WeightedMixRecordYielder returned by this function. Caller
  // should not delete the yielder as it will be handled internally.
  static WeightedMixRecordYielder* New(
      const int64 seed,
      const std::vector<RecordYielder*>& yielders,
      const std::vector<float>& input_source_weights);

 protected:
  WeightedMixRecordYielder(
      const int64 seed,
      const std::vector<RecordYielder*>& yielders,
      const std::vector<float>& input_source_weights);


 private:
  mutable Mutex mu_;

  // PRG used for randomization.
  std::mt19937_64 rnd_ GUARDED_BY(mu_);

  std::discrete_distribution<size_t> sample_distribution_;

  // A list of child yielders used as an input to the mixer.
  std::vector<RecordYielder*> yielders_;
};

}  // namespace lingvo
}  // namespace tensorflow
#endif  // LINGVO_CORE_OPS_WEIGHTED_MIX_RECORD_YIELDER_H_
