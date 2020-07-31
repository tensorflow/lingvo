# Lint as: python3
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
"""Configuration file for the Sphinx documentation builder."""

import os
import sys
from recommonmark.parser import CommonMarkParser

# enable autodoc to load local modules
sys.path.insert(0, os.path.abspath('.'))

project = 'Lingvo'
copyright = '2020'  # pylint: disable=redefined-builtin
author = ''
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx.ext.mathjax',
    'sphinx.ext.napoleon', 'sphinx.ext.todo', 'sphinx.ext.viewcode'
]
autodoc_default_flags = [
    'members', 'undoc-members', 'private-members', 'show-inheritance'
]
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
default_role = 'py:obj'
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.7', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
}
templates_path = ['_templates']
source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
html_static_path = []
html_theme_options = {'nosidebar': True}
todo_include_todos = True
