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

mkdir -p "${ROOT}/tokenized/"

function clean_train {
  local base=$1
  local output_base="${ROOT}/tokenized/${base}"

  # Requires both directions to run and will output both.
  "${TRAIN_CLEANER}" "${output_base}" ${SRC} ${TGT} \
    "${output_base}.clean" 1 100
  "${TRAIN_CLEANER}" "${output_base}" ${TGT} ${SRC} \
    "${output_base}.clean" 1 100
}

tokenize train ${SRC}
tokenize train ${TGT}
tokenize val ${SRC}
tokenize val ${TGT}
tokenize test ${SRC}
tokenize test ${TGT}

clean_train train
clean_train val
clean_train test
