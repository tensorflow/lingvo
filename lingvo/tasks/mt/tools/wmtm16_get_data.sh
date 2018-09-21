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

./lingvo/tasks/mt/tools/wmtm16.01.download_moses_scripts.sh
./lingvo/tasks/mt/tools/wmtm16.02.download_data.sh
./lingvo/tasks/mt/tools/wmtm16.03.unpack_data.sh
./lingvo/tasks/mt/tools/wmtm16.04.tokenize_data.sh
./lingvo/tasks/mt/tools/wmtm16.05.wpm_encode_data.sh
