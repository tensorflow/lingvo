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

set -e
OUTDIR=/tmp/lingvo_apidoc

# Run this from inside docker.
bazel build -c opt //lingvo/core/ops:x_ops.so \
    //lingvo/core/ops:hyps_pb2.py \
    //lingvo/core:inference_graph_pb2.py \
    //lingvo/core:hyperparams_pb2.py \
    //lingvo/tasks/mt:text_input_pb2.py \
    //lingvo/tasks/car/ops \
    //lingvo/core/ops:record_py_pb2 2>&1
cp -f bazel-bin/lingvo/core/ops/x_ops.so lingvo/core/ops
cp -f bazel-bin/lingvo/tasks/car/ops/car_ops.so lingvo/tasks/car/ops
cp -f bazel-genfiles/lingvo/core/ops/record_pb2.py lingvo/core/ops
cp -f bazel-genfiles/lingvo/core/ops/hyps_pb2.py lingvo/core/ops
cp -f bazel-genfiles/lingvo/core/inference_graph_pb2.py lingvo/core
cp -f bazel-genfiles/lingvo/core/hyperparams_pb2.py lingvo/core
cp -f bazel-genfiles/lingvo/tasks/mt/text_input_pb2.py lingvo/tasks/mt
sphinx-apidoc -o "$OUTDIR" -efPM --implicit-namespaces lingvo/ $(find . -name '*_test.py')
cp docs/apidoc/{conf.py,index.rst} "$OUTDIR"
(export PYTHONPATH="$(pwd)" && cd "$OUTDIR" && sphinx-build -b html -T -j auto . build)
rm -f lingvo/core/{inference_graph_pb2.py,hyperparams_pb2.py} lingvo/core/ops/{x_ops.so,hyps_pb2.py,record_pb2.py} lingvo/tasks/car/ops/car_ops.so
rm -f lingvo/tasks/mt/text_input_pb2.py
