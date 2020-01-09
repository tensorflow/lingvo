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

. wmt14_lib.sh

mkdir -p ${ROOT}/wpm

function make_input_files_list {
  local lang=$1
  local indir="${ROOT}/tokenized/train"
  echo "${indir}"/{news-commentary-v9,commoncrawl,europarl-v7}.clean.${lang} \
    | tr ' ' ','
}


SRC_FILES=$(make_input_files_list ${SRC})
TGT_FILES=$(make_input_files_list ${TGT})

TRAIN_TFRECORDS="${ROOT}/wpm/train.tfrecords-%5.5d-of-%5.5d"

# The encoding goes at about 200 sentences per second.
# So, 4.4M sentences will take about 6 hours to run on a single core.
wpm_encode "${SRC_FILES}" "${TGT_FILES}" 200 16 ${TRAIN_TFRECORDS}

cp -f "${WPM_VOC}" "${ROOT}/wpm"
