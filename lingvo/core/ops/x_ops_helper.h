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

#ifndef LINGVO_CORE_OPS_X_OPS_H_
#define LINGVO_CORE_OPS_X_OPS_H_

#define INPUT_ATTRS                                \
  Attr("file_pattern: string")                     \
      .Attr("file_random_seed: int = 301")         \
      .Attr("file_shuffle_shift_ratio: float = 0") \
      .Attr("file_buffer_size: int = 10000")       \
      .Attr("file_parallelism: int = 16")          \
      .Attr("bucket_upper_bound: list(int)")       \
      .Attr("bucket_batch_limit: list(int)")       \
      .Attr("flush_every_n: int = 0")              \
      .Attr("num_threads: int = 1")                \
      .SetIsStateful()

#define INPUT_DOCS \
  R"( \
file_pattern: Glob pattern for the data files.\
file_random_seed: Random seeds used to produce randomized records.\
file_shuffle_shift_ratio: Shifts the list of files after the list is randomly\
    shuffled.\
file_buffer_size: The randomization shuffling buffer.\
file_parallelism: How many sstables are opened and concurrently iterated over.\
bucket_upper_bound: Bucketing scheme. Specifies each bucket's upper bound.\
bucket_batch_limit: Batching scheme. Specifies each bucket's maximum batch\
  size.\
flush_every_n: If non-zero, flushes all batches buffered so far every these\
    many records are yielded.\
num_threads: Number of threads to use for the record batcher. Each thread fills\
    separate batches based on bucket limits.\
)"

#endif  // LINGVO_CORE_OPS_X_OPS_H_
