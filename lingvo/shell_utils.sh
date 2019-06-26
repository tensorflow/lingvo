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
# Common shell utils.

wheres-the-bin() {
  local target=${1:?} query
  query=$(bazel query --output=build -- "$target") || return
  printf "%s/%s\n" \
    "$(bazel info -c opt bazel-bin \
      $(grep python_version <<<"$query" \
        | sed -e 's/.*python_version.*\(PY[23]\).*/--python_version=\1/'))" \
    "$(bazel query "$target" | sed -e 's!:!/!g' -e 's!^//!!' )"
}
