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
"""Starts an IPython kernel with Lingvo deps.

jupyter_http_over_ws must be installed and activated. See
https://research.google.com/colaboratory/local-runtimes.html for more details.

To use:
  bazel run -c opt //lingvo:ipython_kernel
"""

from absl import app
from IPython.html.notebookapp import NotebookApp


def main(_):
  notebookapp = NotebookApp.instance()
  notebookapp.open_browser = False
  notebookapp.ip = "0.0.0.0"
  notebookapp.port = 8888
  notebookapp.allow_origin_pat = "https://colab\\.[^.]+\\.google.com"
  notebookapp.allow_root = True
  notebookapp.token = ""
  notebookapp.disable_check_xsrf = True
  notebookapp.initialize()
  notebookapp.start()


if __name__ == "__main__":
  app.run(main)
