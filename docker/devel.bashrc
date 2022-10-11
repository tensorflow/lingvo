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
#
# ==============================================================================


## Similar to
## https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/tf_sig_build_dockerfiles

# Do not print anything if this is not being used interactively
[ -z "$PS1" ] && return

# Set up attractive prompt
export _TF_VERSION=$(pip3 show tensorflow | grep Version | cut -f 2 -d' ')
export _PY_VERSION=$(python --version | cut -f 2 -d' ')
export PS1="\[\e[31m\]lingvo-dev (TF ${_TF_VERSION}; PY ${_PY_VERSION})\[\e[m\] \[\e[33m\]\w\[\e[m\] > "
export TERM=xterm-256color
alias grep="grep --color=auto"
alias ls="ls --color=auto"
alias l="ls -l"
# Fix nvidia-docker
ldconfig
