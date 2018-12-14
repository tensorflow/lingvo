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
"""Retry on exception."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools
import time
import traceback
import random
import sys
import tensorflow as tf


def Retry(retry_value=Exception,
          max_retries=None,
          initial_delay_sec=1.0,
          delay_growth_factor=1.5,
          delay_growth_fuzz=0.1,
          max_delay_sec=60):
  """Returns a retry decorator."""
  if max_retries is None:
    max_retries = 2**30  # Effectively forever.

  if delay_growth_factor < 1.0:
    raise ValueError("Invalid delay_growth_factor: %f" % delay_growth_factor)

  def _Retry(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      delay = initial_delay_sec
      for retries in itertools.count(0):
        try:
          return func(*args, **kwargs)
        except retry_value as e:
          if retries >= max_retries:
            raise
          time.sleep(delay)
          fuzz_factor = 1.0 + random.random() * delay_growth_fuzz
          delay += delay * (delay_growth_factor - 1) * fuzz_factor
          delay = min(delay, max_delay_sec)

          e_desc_str = "".join(traceback.format_exception_only(e.__class__, e))
          stack_traceback_str = "".join(traceback.format_stack()[:-2])
          e_traceback = sys.exc_info()[2]
          e_traceback_str = "".join(traceback.format_tb(e_traceback))
          tf.logging.info(
              "Retry: caught exception: %s while running %s. "
              "Call failed at (most recent call last):\n%s"
              "Traceback for above exception (most recent call last):\n%s"
              "Waiting for %.2f seconds before retrying.", func.__name__,
              e_desc_str, stack_traceback_str, e_traceback_str, delay)

    return wrapper

  return _Retry
