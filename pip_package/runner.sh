# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#!/bin/bash

set -e -x

# Use the new docker BuildKit subsystem.
export DOCKER_BUILDKIT=1

# Build wheel-building environment, with four python interpreters and access to
# the Tensorflow toolchains.
docker build --tag tensorflow:lingvo_wheelhouse --no-cache \
  -f pip_package/build.Dockerfile .

docker run --rm -it \
  -v /tmp/lingvo:/tmp/lingvo \
  -w /tmp/lingvo \
  tensorflow:lingvo_wheelhouse \
  bash

# Run invoke_build_per_interpreter.sh in the wheelhouse environment.
