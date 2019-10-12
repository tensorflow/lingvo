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

#ifndef LINGVO_CORE_OPS_X_OPS_HELPER_H_
#define LINGVO_CORE_OPS_X_OPS_HELPER_H_

#define INPUT_ATTRS                                   \
  Output("bucket_keys: int32")                        \
      .Attr("file_pattern: string")                   \
      .Attr("input_source_weights: list(float) = []") \
      .Attr("file_random_seed: int = 301")            \
      .Attr("file_buffer_size: int = 10000")          \
      .Attr("file_buffer_size_in_seconds: int = 0")   \
      .Attr("file_parallelism: int = 16")             \
      .Attr("bucket_upper_bound: list(int)")          \
      .Attr("bucket_batch_limit: list(int)")          \
      .Attr("bucket_adjust_every_n: int = 0")         \
      .Attr("flush_every_n: int = 0")                 \
      .Attr("num_threads: int = 1")                   \
      .Attr("require_sequential_order: bool = False") \
      .Attr("repeat_count: int = -1")                 \
      .Attr("use_chaining: bool = False")             \
      .SetIsStateful()

#define INPUT_DOCS \
  R"( \
file_pattern: A comma-separated list of glob patterns or sharded file patterns\
  for the data files. A sharded file pattern looks like /path/name@100 or \
  /path/name@*.\
input_source_weights: A list of input sources weights that control the input\
  example mix. The records will be sampled from inputs proportionally to these\
  weights. When empty list is provided, no mix weighting will be done.\
  Defaults to empty list.\
file_random_seed: Random seeds used to produce randomized records.\
file_buffer_size: The randomization shuffling buffer.\
file_buffer_size_in_seconds: Number of records the shuffling buffer should\
  contain, measured in seconds (the number of records demanded by the trainer\
  in this many seconds).\
file_parallelism: How many sstables are opened and concurrently iterated over.\
bucket_upper_bound: Bucketing scheme. Specifies each bucket's upper bound.\
bucket_batch_limit: Batching scheme. Specifies each bucket's maximum batch\
  size.\
bucket_adjust_every_n: If non-zero, optimize the values of bucket_upper_bound\
  except the last one after every N records based on the current input length\
  distribution.\
flush_every_n: If non-zero, flushes all batches buffered so far every these\
  many records are yielded.\
num_threads: Number of threads to use for the record batcher. Each thread\
  fills separate batches based on bucket limits.\
require_sequential_order: If true, the input op is required to process the file\
  glob as well as the contents of each file in a deterministic sequential order.\
  Setting this automatically disables file_random_seed, file_buffer_size,\
  file_parallelism, num_threads, and requires a single file_pattern.\
repeat_count: Number of repetitions of a dataset before throwing OutOfRange\
  error when using require_sequential_order. Must only be set if\
  require_sequential_order is True.)\
use_chaining: If true, the input op is outputing records from file patterns in \
  order. That is, first all records from first file pattern will be yielded, \
  then all records from the second file pattern and so on. If false, the \
  records from different file patterns will be mixed.\
)"

#endif  // LINGVO_CORE_OPS_X_OPS_HELPER_H_
