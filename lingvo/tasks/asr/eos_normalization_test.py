# Lint as: python3
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
"""Tests for eos_normalization."""
from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.asr import eos_normalization
import numpy as np


class NormalizeTrailingEosTest(test_utils.TestCase, parameterized.TestCase):

  def _assert_label_equivalence(self, ids, id_len, new_ids, new_id_len):
    """Assert label sequence is equivalent after eos normalization."""

    batch_size = ids.shape[0]
    for b in range(batch_size):
      min_no_eos_len = min(id_len[b], new_id_len[b])
      msg = f'Assertion failed for input (ids={ids[b]}, len={id_len[b]})'
      self.assertAllEqual(
          ids[b, :min_no_eos_len], new_ids[b, :min_no_eos_len], msg=msg)

  def _assert_implementation_equivalence(self, new_ids_tf: np.array,
                                         new_ids_np: np.array,
                                         new_id_lens_tf: np.array,
                                         new_id_lens_np: np.array,
                                         ids: np.array, id_lens: np.array,
                                         need_trailing_eos: bool):
    """Assert two impl.

    (numpy and tf) gives exactly the same results.

    Args:
      new_ids_tf: tf-impl. returned `new_ids`.
      new_ids_np: np-impl. returned `new_ids`.
      new_id_lens_tf: tf-impl. returned `new_id_lens`.
      new_id_lens_np: np-impl. returned `new_id_lens`.
      ids: ids array as the input to eos_normalization function.
      id_lens: id_lens array as the input to eos_normalization function.
      need_trailing_eos: whether trailing eos is needed.
    """
    self.assertEqual(new_ids_np.shape, new_ids_tf.shape)
    self.assertEqual(new_id_lens_np.shape, new_id_lens_np.shape)

    batch_size = new_ids_np.shape[0]
    for b in range(batch_size):
      if (new_id_lens_np[b] != new_id_lens_tf[b] or
          not np.array_equal(new_ids_np[b, :], new_ids_tf[b, :])):
        msg = (
            f"For input (ids={ids[b, :]}, id_len={id_lens[b]}), tf's results "
            f'(ids={new_ids_tf[b, :]}, id_len={new_id_lens_tf[b, :]}) '
            "differs from numpy's results "
            f'(ids={new_ids_np[b, :]}, id_len={new_id_lens_np[b, :]}). '
            'Note: need_trailing_eos = {}'.format(need_trailing_eos))
        self.fail(msg=msg)

  @parameterized.named_parameters(
      {
          'testcase_name': 'remove_eos',
          'need_trailing_eos': False,
          'results': [3, 4],
          'data': ([[5, 4, 3, 2, 2], [6, 7, 8, 9, 2]], [4, 4]),
      },
      {
          'testcase_name': 'keep_eos',
          'need_trailing_eos': True,
          'results': [4, 5],
          'data': ([[5, 4, 3, 2, 2], [6, 7, 8, 9, 2]], [4, 4]),
      },
      {
          'testcase_name': 'keep_eos_overflow',
          'need_trailing_eos': True,
          'data': ([[5, 4, 3, 3, 3], [5, 4, 3, 3, 2]], [5, 4]),
          'results': [5, 5],
      },
      {
          'testcase_name': 'keep_eos_overflow_modify_id',
          'need_trailing_eos': True,
          'data': ([[5, 4, 3, 3, 3], [5, 4, 3, 3, 2]], [5, 3]),
          'results': [5, 4],
          # in this test case, the ids will be modifed to
          # [[5, 4, 3, 3, 3],
          #  [5, 4, 3, 2, 2]]
      },
      {
          'testcase_name': 'keep_eos_len',
          'need_trailing_eos': True,
          'data': ([[1, 7, 2, 2, 2], [1, 2, 2, 2, 2], [5, 2, 2, 2, 2],
                    [1, 2, 2, 2, 2], [1, 7, 2, 2, 2]], [2, 1, 1, 1, 2]),
          'results': [3, 2, 2, 2, 3],
          # in this case, the input len does not include eos, but the
          # returned len will include eos
      },
      {
          'testcase_name': 'keep_eos_len0',
          'need_trailing_eos': True,
          'data': ([[1, 7, 2, 2, 2], [2, 2, 2, 2, 2], [5, 2, 2, 2, 2],
                    [1, 2, 2, 2, 2], [1, 7, 2, 2, 2]], [2, 0, 1, 1, 2]),
          'results': [3, 0, 2, 2, 3],
          # in this case, the input len does not include eos, and one of the
          # length is 0
      })
  def test_normalize_trailing_eos(self, need_trailing_eos, results, data):
    ids = tf.convert_to_tensor(data[0], dtype=tf.int32)
    id_lens = tf.convert_to_tensor(data[1], dtype=tf.int32)

    with self.session(use_gpu=False) as sess:
      new_ids, new_id_lens = eos_normalization.NormalizeTrailingEos(
          ids, id_lens, need_trailing_eos=need_trailing_eos, eos_id=2)
      new_ids_np, ids_np, new_id_lens_np, id_lens_np = sess.run(
          [new_ids, ids, new_id_lens, id_lens])
      self.assertAllEqual(new_id_lens_np, results)
      self._assert_label_equivalence(ids_np, id_lens_np, new_ids_np,
                                     new_id_lens_np)

  def _generate_random_test_data(self, fill_eos=True, eos_id=2):
    batch_size = np.random.randint(low=1, high=10, size=1).tolist()[0]
    max_label_len = np.random.randint(low=5, high=10, size=1).tolist()[0]
    ids = np.random.randint(low=0, high=40, size=(batch_size, max_label_len))
    id_len = np.random.randint(low=1, high=max_label_len, size=(batch_size,))
    if fill_eos:
      for b in range(batch_size):
        ids[b, id_len[b]:] = eos_id
    return ids, id_len

  def _test_equivalent_to_reference_implementation(self, need_trailing_eos):
    for _ in range(10):
      ids, id_lens = self._generate_random_test_data(fill_eos=True)
      new_ids_v, new_id_lens_v = eos_normalization.NormalizeTrailingEos(
          tf.convert_to_tensor(ids, dtype=tf.int32),
          tf.convert_to_tensor(id_lens, dtype=tf.int32),
          need_trailing_eos=need_trailing_eos,
          eos_id=2)
      with self.session(use_gpu=False) as sess:
        new_ids, new_id_lens = sess.run([new_ids_v, new_id_lens_v])

      (new_ids_ref, new_id_lens_ref) = (
          eos_normalization.NumpyNormalizeTrailingEos(
              ids, id_lens, need_trailing_eos=need_trailing_eos, eos_id=2))

      self._assert_implementation_equivalence(
          new_ids,
          new_ids_ref,
          new_id_lens,
          new_id_lens_ref,
          ids,
          id_lens,
          need_trailing_eos=need_trailing_eos)

  @parameterized.named_parameters(
      {
          'testcase_name': 'need_eos',
          'need_eos': True
      }, {
          'testcase_name': 'not_need_eos',
          'need_eos': False
      })
  def test_equivalent_to_reference_implementation(self, need_eos):
    self._test_equivalent_to_reference_implementation(need_eos)


if __name__ == '__main__':
  tf.test.main()
