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

./wmt14.01.download_moses_scripts.sh
./wmt14.02.download_train.sh
./wmt14.03.download_devtest.sh
./wmt14.04.unpack_train.sh
./wmt14.05.unpack_devtest.sh
./wmt14.06.tokenize_train.sh
./wmt14.07.tokenize_devtest.sh
./wmt14.08.wpm_encode_train.sh
./wmt14.09.wpm_encode_devtest.sh
