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

#include "tensorflow/core/platform/env.h"
#include "lingvo/core/ops/mutex.h"
#include "lingvo/core/ops/record_batcher.h"
#include "lingvo/core/ops/record_yielder.h"

namespace tensorflow {
namespace lingvo {

void BasicRecordYielder::WaitForBufEnough() {
  if (!BufEnough()) {
    auto start = Env::Default()->NowMicros();
    mu_.Await(buf_enough_);
    VLOG(1) << "Wait for buf containing enough records: "
            << (Env::Default()->NowMicros() - start) * 1e-6
            << " Hint: Check network condition (e.g., are files in the same "
            << "data center) and/or increase file_parallelism.";
  }
}

void RecordBatcher::WaitForCurrEmpty() {
  if (!CurrEmpty()) {
    auto start = Env::Default()->NowMicros();
    mu_.Await(curr_empty_);
    VLOG(2)
        << "Wait for curr empty: "
        << (Env::Default()->NowMicros() - start) * 1e-6
        << " Hint: Processing is not fast enough to consume example batches.";
  }
}

void RecordBatcher::WaitForCurrNonEmpty() {
  if (!CurrNonEmpty()) {
    auto start = Env::Default()->NowMicros();
    mu_.Await(curr_non_empty_);
    VLOG(1) << "Wait for curr non empty: "
            << (Env::Default()->NowMicros() - start) * 1e-6
            << " Hint: Consider improving Merge() method.";
  }
}

void RecordBatcher::WaitForToFlushEmpty() {
  if (!ToFlushEmpty()) {
    auto start = Env::Default()->NowMicros();
    mu_.Await(to_flush_empty_);
    VLOG(3) << "Wait for to_flush empty: "
            << (Env::Default()->NowMicros() - start) * 1e-6
            << " Hint: Expected to be the common case.";
  }
}

void RecordBatcher::WaitForToFlushNonEmpty() {
  if (!ToFlushNonEmpty()) {
    auto start = Env::Default()->NowMicros();
    mu_.Await(to_flush_non_empty_);
    VLOG(1) << "Wait for to_flush non empty: "
            << (Env::Default()->NowMicros() - start) * 1e-6
            << " Hint: Increase num_batcher_thread.";
  }
}

}  // namespace lingvo
}  // namespace tensorflow
