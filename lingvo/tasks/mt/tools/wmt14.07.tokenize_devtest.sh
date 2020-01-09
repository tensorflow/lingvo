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

# --- TEST SET.
mkdir -p "${ROOT}/tokenized/test"
function sgm_to_input {
  local file=$1
  local output="$2"

  cat ${ROOT}/unpacked/$file | ${INPUT_FROM_SGM} > "${ROOT}/unpacked/test/$2"
}

# TODO(drpng): eliminate en/de patterns.
sgm_to_input test/newstest2014-deen-src.${SRC}.sgm newstest2014.${SRC}
sgm_to_input test/newstest2014-deen-ref.${TGT}.sgm newstest2014.${TGT}

tokenize test/newstest2014.${SRC} ${SRC} test newstest2014
tokenize test/newstest2014.${TGT} ${TGT} test newstest2014

# --- DEV SET.
mkdir -p "${ROOT}/tokenized/dev"
# While 2008 onwards are available, we only use 2013 as the dev set. The "input"
# file is already in the tar file, so no need to pull from the SGM.
tokenize dev/newstest2013.${SRC} ${SRC} dev newstest2013
tokenize dev/newstest2013.${TGT} ${TGT} dev newstest2013
