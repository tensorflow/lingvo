#!/bin/bash
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

# This script is invoked once per Python version to produce lingvo pip wheel.
# The chain of scripts is:
#    runner.sh (main)
#      -> invoke_build_per_interpreter.sh
#      -> this script
#      -> build_pip_pkg.sh
#
# Extra arguments get passed through to the bazel commands.
#
# See README.md for more usage instructions.

set -e -x

# Override the following env variables if necessary.
export PYTHON_MINOR_VERSION="${PYTHON_MINOR_VERSION?python minor version required}"
PYTHON="python3.${PYTHON_MINOR_VERSION}"
update-alternatives --install /usr/bin/python3 python3 "/usr/bin/$PYTHON" 1

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

# Remove .bazelrc if it already exists
[ -e .bazelrc ] && rm .bazelrc

write_to_bazelrc "build -c opt"
write_to_bazelrc 'build --copt=-mavx --host_copt=-mavx'
write_to_bazelrc 'build --auto_output_filter=subpackages'
write_to_bazelrc 'build --copt="-Wall" --copt="-Wno-sign-compare"'
write_to_bazelrc 'build --linkopt="-lrt -lm"'
write_to_bazelrc 'build --experimental_repo_remote_exec'
write_to_bazelrc 'test --test_summary=short'

write_action_env_to_bazelrc PYTHON_BIN_PATH "/usr/bin/${PYTHON}"
write_action_env_to_bazelrc PYTHON_LIB_PATH "/usr/lib/${PYTHON}"

TF_NEED_CUDA=0
echo 'Using installed tensorflow'
TF_CFLAGS=( $(${PYTHON} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS="$(${PYTHON} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)
if ! [[ $TF_LFLAGS =~ .*:.* ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
  else
    SHARED_LIBRARY_NAME="libtensorflow_framework.so"
  fi
fi
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_NEED_CUDA" ${TF_NEED_CUDA}

# Exclude lingvo Jax from the pip package.
# TODO(b/203463351): Add lingvo jax into a pip package.
rm -rf lingvo/jax/

# It is expected that you have git cloned this repo at the branch you want,
# ideally in our docker.

echo 'Using .bazelrc:\n'
batcat .bazelrc -l sh

bazel clean
# Add -s for verbose logging of all bazel subcommands
bazel build $@ ...
if ! [[ $SKIP_TESTS ]]; then
  # Just test the core for the purposes of the pip package.
  bazel test $@ lingvo/core/...
fi

DST_DIR="/tmp/lingvo/dist"
./pip_package/build_pip_pkg.sh "${DST_DIR}" "3"

# Note: constraining our release to plat==manylinux2014_x86_64 to match TF.
# This corresponds to our use of the devtoolset-9 toolchain.
find "$DST_DIR" -name "*cp3${PYTHON_MINOR_VERSION}*.whl" |\
  xargs -n1 ./third_party/auditwheel.sh repair --plat manylinux2014_x86_64 -w "$DST_DIR"

rm .bazelrc

