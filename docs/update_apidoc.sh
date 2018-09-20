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

# Run this from inside docker.
bazel build -c opt lingvo:trainer lingvo/core/ops:hyps_py_pb2
cp -f bazel-bin/lingvo/core/ops/x_ops.so lingvo/core/ops
cp -f bazel-genfiles/lingvo/core/ops/hyps_pb2.py lingvo/core/ops
rm -f docs/apidoc/lingvo.*.rst
sphinx-apidoc -o docs/apidoc -efPM --implicit-namespaces lingvo/ $(find . -name '*_test.py')
(cd docs/apidoc && PYTHONPATH=../.. sphinx-build -M html . _build)
rm -f lingvo/core/ops/{x_ops.so,hyps_pb2.py}
