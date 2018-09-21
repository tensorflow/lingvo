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

. ./lingvo/tasks/mt/tools/wmtm16_lib.sh

mkdir -p ${ROOT}/wpm

# The encoding goes at about 200 sentences per second.
for BASE in "train" "val" "test"
do
  echo "Encoding ${BASE}"
  TFRECORDS="${ROOT}/wpm/${BASE}.tfrecords"
  SRC_FILE="${ROOT}/tokenized/${BASE}.clean.${SRC}"
  TGT_FILE="${ROOT}/tokenized/${BASE}.clean.${TGT}"
  wpm_encode "${SRC_FILE}" "${TGT_FILE}" 200 1 ${TFRECORDS}
done

cp -f "${WPM_VOC}" "${ROOT}/wpm"
