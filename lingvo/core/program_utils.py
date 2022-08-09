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
"""Utils function for program.py."""

import os

import lingvo.compat as tf


def SummaryToCsv(summaries):
  """Convert summary (Dict[str, tf.Summary]) to csv format."""
  res = ''
  for k, s in summaries.items():
    res += f'{k},{s.value[0].simple_value}\n'
  return res


def CsvToSummary(csv):
  """Convert csv format to summary (Dict[str, tf.Summary])."""
  summaries = {}
  for l in csv.split('\n'):
    row = l.split(',')
    if len(row) != 2:
      tf.logging.warn(f'Failed to parse csv line: {l}, will ignore it.')
      continue
    s = tf.Summary()
    v = s.value.add()
    v.tag, v.simple_value = row[0], float(row[1])
    summaries.update({v.tag: s})
  return summaries


class DecodeStatusCache:
  """Maintain status file to keep decoding datasets status.

  Status file should have following format:
  - 1st line is checkpoint key, e.g. ckpt-123
  - the rest lines are dataset names that has been decoded.
  Here's an example:
  ckpt-123
  Dev
  Test
  """

  def __init__(self, program_dir):
    self.ckpt_key = ''
    self.decoded_datasets = []
    self.status_file = os.path.join(program_dir, 'decoded_datasets.txt')
    # TODO(xingwu): Consider add a TTL.
    self.cache_dir = os.path.join(program_dir, 'cache')
    tf.io.gfile.makedirs(self.cache_dir)
    if tf.io.gfile.exists(self.status_file):
      with tf.io.gfile.GFile(self.status_file, 'r') as f:
        content = list(l.strip() for l in f.readlines())
        if content:
          self.ckpt_key = content[0]
        if len(content) > 1:
          self.decoded_datasets = content[1:]

  def UpdateCkpt(self, ckpt_key):
    """Update checkpoint key in the status."""
    if ckpt_key != self.ckpt_key:
      self.ckpt_key = ckpt_key
      self.decoded_datasets = []
      with tf.io.gfile.GFile(self.status_file, 'w') as f:
        f.write(self.ckpt_key)

  def UpdateDataset(self, dataset_name, summaries):
    """Update decoded dataset in the status."""
    cache_file = os.path.join(self.cache_dir, f'{dataset_name}.csv')
    with tf.io.gfile.GFile(cache_file, 'w') as f:
      f.write(SummaryToCsv(summaries))
    with tf.io.gfile.GFile(self.status_file, 'w+') as f:
      f.write(f.read().strip() + '\n' + dataset_name)

  def TryLoadCache(self, ckpt_key, dataset_name):
    """Try load summary cache for ckpt_key, dataset_name.

    Args:
      ckpt_key: str, checkpoint key, e.g. ckpt-123
      dataset_name: str, the dataset name, e.g. Test

    Returns:
      summaries if load successful, otherwise, return None
    """
    if ckpt_key == self.ckpt_key and dataset_name in self.decoded_datasets:
      cache_file = os.path.join(self.cache_dir, f'{dataset_name}.csv')
      if not tf.io.gfile.exists(cache_file):
        tf.logging.warn(f'cached summary {cache_file} is gone!')
        return None
      with tf.io.gfile.GFile(cache_file, 'r') as f:
        summaries = CsvToSummary(f.read())
      with tf.io.gfile.GFile(self.status_file, 'w+') as f:
        f.write(f.read().strip() + '\n' + dataset_name)
      return summaries
    return None


class TriggerScheduler:
  """A trigger scheduler with offset, and interval.

  Maintains an counter, incremented when Trigger() called. ShouldRun() only
  returns True when (counter - offset) % interval == 0.
  """

  def __init__(self, offset, interval):
    self.offset = offset
    self.interval = interval
    self.counter = -offset

  def Trigger(self):
    self.counter += 1
    if self.counter >= self.interval:
      self.counter = 0

  def ShouldRun(self):
    return self.counter == 0
