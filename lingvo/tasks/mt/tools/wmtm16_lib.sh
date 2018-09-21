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

ROOT=/tmp/wmtm16

SRC=en
TGT=de

TOKENIZER="${ROOT}/mosesdecoder/scripts/tokenizer"
TRAIN_CLEANER="${ROOT}/mosesdecoder/scripts/training/clean-corpus-n.perl"
INPUT_FROM_SGM="${ROOT}/mosesdecoder/scripts/ems/support/input-from-sgm.perl"

function tokenize {
  local input_prefix="$1"
  local lang="$2"
  local out=$input_prefix
  local output_base="${ROOT}/tokenized/${out}"

  echo "$input_prefix.$lang -> ${out}.${lang}"
  zcat -f "${ROOT}/unpacked/${input_prefix}.$lang" | \
  "${TOKENIZER}"/tokenizer.perl \
    -threads 20 \
    -l ${lang} > \
    "${output_base}.${lang}"
}

WPM_BINARY=./bazel-bin/lingvo/tools/wpm_encode_file
WPM_VOC=./lingvo/tasks/mt/wpm-${SRC}${TGT}-2k.voc

function wpm_encode {
  local source_files="$1"
  local target_files="$2"
  local max_len="$3"
  local num_shards="$4"
  local output_template="$5"

  rm -f convert.FAILED
  for n in $(seq "${num_shards}"); do
    local shard_id=$((n - 1))
    local output_filepath=$(printf ${output_template} ${shard_id} ${num_shards})
    set -x
    nice -n 20 ${WPM_BINARY} --wpm_filepath=${WPM_VOC} --source_filepaths="${source_files}" \
      --target_filepaths="${target_files}" --num_shards="${num_shards}" --shard_id="${shard_id}" \
      --max_len="$max_len" \
      --output_filepath="${output_filepath}" --logtostderr || touch convert.FAILED &
    set +x
  done
  wait
  ! [ -f convert.FAILED ]
}

test -x ${WPM_BINARY} || (echo 1>&2 "Please build wpm_encode_file binary."; exit 1)
test -f ${WPM_VOC} || (echo 1>&2 "Could not locate wpm file: ${WPM_VOC}"; exit 2)
