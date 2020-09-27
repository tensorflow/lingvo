#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

cd "${0%/*}"  # Change to script directory.

if ! command -v sphinx-build &> /dev/null
then
    echo "sphinx could not be found. Please install with 'pip3 install sphinx sphinx_rtd_theme recommonmark.'"
    exit
fi

cleanup() {
  rm -rf single.py* _build __pycache__
  exit
}
trap cleanup ERR INT TERM

echo 'Paste the docstring to check including the """ (end with ctrl+D).'
cat > single.py

sphinx-build -b singlehtml -q -T . _build
cleanup
