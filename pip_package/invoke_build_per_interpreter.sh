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

# Grab patchelf >= 16; takes care of the following error which popped up:
#  'auditwheel repair requires patchelf >= 0.14'
pip install patchelf

PYTHON_MINOR_VERSION=10 pip_package/build.sh \
	--crosstool_top=@sigbuild-r2.9-python3.10_config_cuda//crosstool:toolchain

PYTHON_MINOR_VERSION=9 pip_package/build.sh \
	--crosstool_top=@sigbuild-r2.9-python3.9_config_cuda//crosstool:toolchain

PYTHON_MINOR_VERSION=8 pip_package/build.sh \
	--crosstool_top=@sigbuild-r2.9-python3.8_config_cuda//crosstool:toolchain
