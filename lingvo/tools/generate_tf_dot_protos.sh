#!/bin/bash

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

# Generates the tensorflow .proto files. This is a thin wrapper around
# generate_proto_def.

binary=$1
dest=$2

set -e
set -u
set +x

rm -f ${dest}/.generated

mkdir -p ${dest}/tensorflow/core/framework
mkdir -p ${dest}/tensorflow/core/protobuf

${binary} ${dest}

# genrule requires statically determined outputs, so we package all
# into a single file.
tar -C ${dest} -cf ${dest}/tf_protos.tar tensorflow/core/{framework,protobuf}
