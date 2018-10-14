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

mkdir -p "${ROOT}/train"

# To save space, we don't unpack to intermediate files. The first pass collects
# all transcription files from the tarball. The second pass unpacks the audio,
# decompresses it, and encodes MFCC frames in memory, then writes a tf.Example
# with the accompanying transcription from the first pass.

# This takes about 10 minutes per set.
for subset in train-clean-100 train-clean-360 train-other-500; do
  echo "=== First pass, collecting transcripts: ${subset}"
  bazel-bin/lingvo/tools/create_asr_features --logtostderr \
    --input_tarball="${ROOT}/raw/${subset}.tar.gz" --dump_transcripts \
    --transcripts_filepath="${ROOT}/train/${subset}.txt"
done

# We are allocating as follows:
#  num utts num_shards  subset
#   28539     10        train-clean-100
#  104014     40        train-clean-360
#  148688     50        train-other-500
#  281241    100        Total
#
# We expect a total of, say, 100GB, so we want 100 shards to get into the 1GB
# range. We use 10 processors, each alloted a range of 10 output shards.

# Second pass: Create tf.Examples. It takes about 90 minutes.

rm -f FAILED
subset=train-clean-100
echo "=== Second pass, parameterization: ${subset}"
for subshard in $(seq 0 9); do
  set -x
  nice -n 20 bazel-bin/lingvo/tools/create_asr_features \
    --logtostderr \
    --input_tarball="${ROOT}/raw/${subset}.tar.gz" --generate_tfrecords \
    --transcripts_filepath="${ROOT}/train/${subset}.txt" \
    --shard_id="${subshard}" --num_shards=10 --num_output_shards=100 \
    --output_range_begin="${subshard}" --output_range_end="$((subshard + 1))" \
    --output_template="${ROOT}/train/train.tfrecords-%5.5d-of-%5.5d" || touch FAILED &
  set +x
done
wait
! [ -f FAILED ]

subset=train-clean-360
echo "=== Second pass, parameterization: ${subset}"
for subshard in $(seq 0 9); do
  set -x
  nice -n 20 bazel-bin/lingvo/tools/create_asr_features \
    --logtostderr \
    --input_tarball="${ROOT}/raw/${subset}.tar.gz" --generate_tfrecords \
    --transcripts_filepath="${ROOT}/train/${subset}.txt" \
    --shard_id="${subshard}" --num_shards=10 --num_output_shards=100 \
    --output_range_begin="$((10 + 4 * subshard))" \
    --output_range_end="$((10 + 4 * subshard + 4))" \
    --output_template="${ROOT}/train/train.tfrecords-%5.5d-of-%5.5d" || touch FAILED &
  set +x
done
wait
! [ -f FAILED ]

subset=train-other-500
echo "=== Second pass, parameterization: ${subset}"
for subshard in $(seq 0 9); do
  set -x
  nice -n 20 bazel-bin/lingvo/tools/create_asr_features \
    --logtostderr \
    --input_tarball="${ROOT}/raw/${subset}.tar.gz" --generate_tfrecords \
    --transcripts_filepath="${ROOT}/train/${subset}.txt" \
    --shard_id="${subshard}" --num_shards=10 --num_output_shards=100 \
    --output_range_begin="$((50 + 5 * subshard))" \
    --output_range_end="$((50 + 5 * subshard + 5))" \
    --output_template="${ROOT}/train/train.tfrecords-%5.5d-of-%5.5d" || touch FAILED &
  set +x
done
wait
! [ -f FAILED ]
