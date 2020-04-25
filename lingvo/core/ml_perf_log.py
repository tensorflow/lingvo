# Lint as: python3

# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
"""Convenience function for logging compliance tags to stdout."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import json
import logging
import os
import re
import sys
import time

PATTERN = re.compile('[a-zA-Z0-9]+')

LOG_FILE = os.getenv('COMPLIANCE_FILE')
# create logger with 'spam_application'
LOGGER = logging.getLogger('mlperf_compliance')
LOGGER.setLevel(logging.DEBUG)

_STREAM_HANDLER = logging.StreamHandler(stream=sys.stdout)
_STREAM_HANDLER.setLevel(logging.INFO)
LOGGER.addHandler(_STREAM_HANDLER)

if LOG_FILE:
  _FILE_HANDLER = logging.FileHandler(LOG_FILE)
  _FILE_HANDLER.setLevel(logging.DEBUG)
  LOGGER.addHandler(_FILE_HANDLER)
else:
  _STREAM_HANDLER.setLevel(logging.DEBUG)


def get_caller(stack_index=2, root_dir=None):
  caller = inspect.getframeinfo(inspect.stack()[stack_index][0])

  # Trim the filenames for readability.
  filename = caller.filename
  if root_dir is not None:
    filename = re.sub('^' + root_dir + '/', '', filename)
  return (filename, caller.lineno)


# :::MLL 1556733699.71 run_start: {"value": null,
# "metadata": {"lineno": 77, "file": main.py}}
LOG_TEMPLATE = ':::MLL {:.3f} {}: {{"value": {}, "metadata": {}}}'


def mlperf_format(key, value, now, stack_offset=0, metadata=None):
  """Format a message for MLPerf."""
  if metadata is None:
    metadata = {}

  if 'lineno' not in metadata:
    filename, lineno = get_caller(2 + stack_offset, root_dir=None)
    metadata['lineno'] = lineno
    metadata['file'] = filename

  msg = LOG_TEMPLATE.format(now, key, json.dumps(value), json.dumps(metadata))
  return msg


def mlperf_print(key, value, stack_offset=0, metadata=None):
  now = time.time()
  LOGGER.info(
      mlperf_format(
          key, value, now, stack_offset=stack_offset + 1, metadata=metadata))
  return now
