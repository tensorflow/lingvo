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
bazel build -c opt \
    //lingvo/core/ops:x_ops.so \
    //lingvo/tasks/car/ops:car_ops.so \
    //lingvo/core:inference_graph_pb2.py \
    //lingvo/core:hyperparams_pb2.py \
    //lingvo/core/ops:hyps_pb2.py \
    //lingvo/core/ops:record_py_pb2 \
    //lingvo/core/ops:versioned_file_set_py_pb2 \
    //lingvo/tasks/mt:text_input_pb2.py \
    2>&1
cp -f bazel-bin/lingvo/core/ops/x_ops.so lingvo/core/ops
cp -f bazel-bin/lingvo/tasks/car/ops/car_ops.so lingvo/tasks/car/ops
cp -f bazel-bin/lingvo/core/inference_graph_pb2.py lingvo/core
cp -f bazel-bin/lingvo/core/hyperparams_pb2.py lingvo/core
cp -f bazel-bin/lingvo/core/ops/hyps_pb2.py lingvo/core/ops
cp -f bazel-bin/lingvo/core/ops/record_pb2.py lingvo/core/ops
cp -f bazel-bin/lingvo/core/ops/versioned_file_set_pb2.py lingvo/core/ops
cp -f bazel-bin/lingvo/tasks/mt/text_input_pb2.py lingvo/tasks/mt
rm -rf lingvo/tasks/car  # TODO(b/179168646): generate APIdocs for car.
sphinx-apidoc -o "$OUTDIR" -efPM --implicit-namespaces lingvo/ $(find . -name '*_test.py')
cp docs/apidoc/{conf.py,index.rst} "$OUTDIR"
(export PYTHONPATH="$(pwd)" && cd "$OUTDIR" && sphinx-build -b html -T -j auto . build)
rm -f lingvo/core/{inference_graph_pb2.py,hyperparams_pb2.py}
rm -f lingvo/core/ops/{x_ops.so,hyps_pb2.py,record_pb2.py,versioned_file_set_pb2.py}
rm -f lingvo/tasks/car/ops/car_ops.so
rm -f lingvo/tasks/mt/text_input_pb2.py
