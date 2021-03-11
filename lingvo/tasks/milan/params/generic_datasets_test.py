# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for lingvo.tasks.milan.params.generic_datasets."""

import json
import os

from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.milan.params import generic_datasets


class ImageTextTFRecordsTest(test_utils.TestCase):

  def testFromEnv(self):
    env_var_name = 'MILAN_DATASET_CONFIG_JSON'
    # For hermeticity: Ensure this variable isn't set, and unset it afterward.
    self.assertNotIn(env_var_name, os.environ)
    self.addCleanup(os.environ.pop, env_var_name, None)

    params_in_env = dict(
        split_paths=dict(train='foo', dev='bar', test='/abs/path/baz'),
        bert_max_length=17,
        data_dir='/base/dir')
    os.environ[env_var_name] = json.dumps(params_in_env)

    dataset_spec = generic_datasets.ImageTextTFRecords.ParamsFromEnv(
        env_var_name).Instantiate()

    actual_params_as_dict = dict(dataset_spec.params.IterParams())
    actual_params_as_dict.pop('cls')
    self.assertSameStructure(actual_params_as_dict, params_in_env)
    self.assertEqual(
        dataset_spec._split_paths,
        dict(train='/base/dir/foo', dev='/base/dir/bar', test='/abs/path/baz'))
    self.assertEqual(
        dataset_spec.meta.features['text/bert/token_features'].shape[-2:],
        [params_in_env['bert_max_length'], 768])


if __name__ == '__main__':
  tf.test.main()
