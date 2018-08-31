#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -eu

. ./lingvo/tasks/asr/tools/librispeech_lib.sh

mkdir -p "${ROOT}/devtest"

for subset in {dev,test}-{clean,other}; do
  set -x
  bazel-bin/lingvo/tools/create_asr_features \
    --logtostderr \
    --input_tarball="${ROOT}/raw/${subset}.tar.gz" --generate_tfrecords \
    --shard_id=0 --num_shards=1 --num_output_shards=1 \
    --output_range_begin=0 --output_range_end=1 \
    --output_template="${ROOT}/devtest/${subset}.tfrecords-%5.5d-of-%5.5d"
  set +x
done
