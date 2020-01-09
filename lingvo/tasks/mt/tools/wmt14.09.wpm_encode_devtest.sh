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

DEV_SRC_FILES="${ROOT}"/tokenized/dev/newstest2013.${SRC}
TEST_SRC_FILES="${ROOT}"/tokenized/test/newstest2014.${SRC}
DEV_TGT_FILES="${ROOT}"/tokenized/dev/newstest2013.${TGT}
TEST_TGT_FILES="${ROOT}"/tokenized/test/newstest2014.${TGT}
DEV_TFRECORDS="${ROOT}/wpm/dev.tfrecords"
TEST_TFRECORDS="${ROOT}/wpm/test.tfrecords"

wpm_encode "${DEV_SRC_FILES}" "${DEV_TGT_FILES}" 0 1 ${DEV_TFRECORDS}
wpm_encode "${TEST_SRC_FILES}" "${TEST_TGT_FILES}" 0 1 ${TEST_TFRECORDS}

cp -f "${WPM_VOC}" "${ROOT}/wpm"
