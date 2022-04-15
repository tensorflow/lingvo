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
"""Unit tests for model."""

from typing import Any, Tuple

from absl.testing import absltest
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import base_model
from lingvo.jax import py_utils
from lingvo.jax import test_utils
import numpy as np

NestedMap = py_utils.NestedMap
InstantiableParams = py_utils.InstantiableParams


class MockLM(base_layer.BaseLayer):

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'logits', None,
        'results returned by extend_step(), shape [max step, batch size, '
        'vocab size].')
    return p

  def __init__(self, p: InstantiableParams) -> None:
    super().__init__(p)
    self.logits = jnp.array(p.logits, dtype=jnp.float32)

  def init_states(self, *args: Any, **kwargs: Any) -> NestedMap:
    return NestedMap(step=0)

  def extend_step(
      self,
      states: NestedMap,
      inputs: Any,
  ) -> Tuple[Any, NestedMap]:
    del inputs
    ret = NestedMap()
    ret.logits = self.logits.at[states.step].get()
    states.step = states.step + 1
    return states, ret


class LanguageModelTest(test_utils.TestCase):

  def _run_decode(self, decoder_p, logits, input_batch):
    p = base_model.LanguageModel.Params()
    p.name = 'mock_lm'
    p.decoder = decoder_p.Copy()
    p.lm = MockLM.Params()
    p.lm.logits = logits
    lang_model = p.Instantiate()
    theta = NestedMap(lm=NestedMap())
    # We fix seed to 1027 to get the desired prefix lengths below.
    _, results = test_utils.apply(
        lang_model, theta, lang_model.decode, input_batch, seed=1027)
    return results

  def test_base_case(self):
    p = base_model.LanguageModel.Params().decoder
    p.seqlen = 3
    p.min_prefix_len = 0
    logits = [
        [
            [0, 1, 0, 0],
        ],
        [
            [0, 0, 0, 1],
        ],
    ]
    # We use full paddings to force prefix lengths to be 0 (since it is capped
    # at the lengths of input ids.
    input_batch = NestedMap(
        ids=jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.ones(shape=(1, 5), dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    self.assertArraysEqual(results.prefix_lengths, np.array([0],
                                                            dtype=np.int32))
    # Decoding starts at 1 from input.ids, then each step uses argmax from the
    # provided logits, which are 1 and 3.
    self.assertArraysEqual(results.output_ids,
                           np.array([[11, 1, 3]], dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths, np.array([3],
                                                            dtype=np.int32))

  def test_prefix(self):
    p = base_model.LanguageModel.Params().decoder
    p.seqlen = 5
    p.min_prefix_len = 2
    logits = [
        [
            [0, 1, 0, 0, 0, 0],  # argmax=1
        ],
        [
            [0, 0, 0, 1, 0, 0],  # argmax=3
        ],
        [
            [0, 0, 0, 0, 1, 0],  # argmax=4
        ],
        [
            [0, 0, 0, 0, 0, 1],  # argmax=5
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 12, 13, 14, 15]], dtype=jnp.int32),
        paddings=jnp.array([[0., 0., 1., 1., 1.]], dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    self.assertArraysEqual(results.prefix_lengths, np.array([2],
                                                            dtype=np.int32))
    # We copy prefix of length 2 from input.ids, so the first argmax
    # from logits is unused. Remaining 3 ids are from argmax.
    self.assertArraysEqual(results.output_ids,
                           np.array([[11, 12, 3, 4, 5]], dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths, np.array([5],
                                                            dtype=np.int32))

  def test_eos_terminate(self):
    p = base_model.LanguageModel.Params().decoder
    p.seqlen = 6
    p.min_prefix_len = 0
    p.eos_id = 2
    logits = [
        [
            [0, 0, 0, 0, 1],  # argmax=4
        ],
        [
            [0, 0, 1, 0, 0],  # argmax=2
        ],
        [
            [0, 0, 0, 1, 0],  # argmax=3
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 13]], dtype=jnp.int32),
        paddings=jnp.ones(shape=(1, 2), dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    self.assertArraysEqual(results.prefix_lengths, np.array([0],
                                                            dtype=np.int32))
    # Decoding terminates after step 2 when eos_id=2 is encountered.
    self.assertArraysEqual(results.output_ids,
                           np.array([[11, 4, 2, 0, 0, 0]], dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths, np.array([3],
                                                            dtype=np.int32))

  def test_eos_independent(self):
    p = base_model.LanguageModel.Params().decoder
    p.seqlen = 5
    p.min_prefix_len = 0
    p.eos_id = 2
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[4, 3]
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],  # argmax=[2, 4]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],  # argmax=[3, 2]
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 13], [12, 14]], dtype=jnp.int32),
        paddings=jnp.ones(shape=(2, 2), dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([0, 0], dtype=np.int32))
    # EOS termination are row independent: row 0 terminates at step 2 while
    # row 1 terminates at step 3.
    self.assertArraysEqual(
        results.output_ids,
        np.array([[11, 4, 2, 0, 0], [12, 3, 4, 2, 0]], dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths,
                           np.array([3, 4], dtype=np.int32))

  def test_prefix_and_eos(self):
    p = base_model.LanguageModel.Params().decoder
    p.seqlen = 5
    p.min_prefix_len = 0
    p.eos_id = 2
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],  # argmax=[4, 3, 3]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],  # argmax=[3, 4, 2]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],  # argmax=[3, 2, 3]
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[4, 4, 3]
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 13, 15], [12, 14, 16], [20, 30, 40]],
                      dtype=jnp.int32),
        paddings=jnp.zeros(shape=(3, 3), dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    # This is fixed by the prng seed provided.
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([2, 1, 0], dtype=np.int32))
    # Row 0 copies 2 ids from the input as prefix, and continues without
    # ever hitting EOS. Row 1 and 2 only copies the first id from the input,
    # and continues until EOS is found.
    self.assertArraysEqual(
        results.output_ids,
        np.array([[11, 13, 3, 3, 4], [12, 3, 4, 2, 0], [20, 3, 2, 0, 0]],
                 dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths,
                           np.array([5, 4, 3], dtype=np.int32))

  def test_max_decode_steps(self):
    p = base_model.LanguageModel.Params().decoder
    p.seqlen = 5
    p.min_prefix_len = 0
    p.eos_id = 2
    p.max_decode_steps = 2
    logits = [
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],  # argmax=[4, 3, 3]
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],  # argmax=[2, 4, 4]
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],  # argmax=[3, 4, 3]
        ],
    ]
    input_batch = NestedMap(
        ids=jnp.array([[11, 13, 15], [12, 14, 16], [20, 30, 40]],
                      dtype=jnp.int32),
        paddings=jnp.zeros(shape=(3, 3), dtype=jnp.float32),
    )
    results = self._run_decode(p, logits, input_batch)
    # This is fixed by the prng seed provided.
    self.assertArraysEqual(results.prefix_lengths,
                           np.array([2, 1, 0], dtype=np.int32))
    # Row 0 has prefix length 2, and hit EOS after decode for one step, so it
    # stops. Row 1 has prefix length 1, and hit max decode steps of 2, so it
    # stops at 3 decoded ids. Row 2 has prefix length 0, and stops after
    # hitting the max decode step of 2, ending with 2 decoded ids.
    # Note that logically prefix length 1 and 0 are equivalent, because
    # decoding always starts with the fixed first ids (BOS in practice), the
    # only difference is how they affect the counting of max_decode_steps.
    self.assertArraysEqual(
        results.output_ids,
        np.array([[11, 13, 2, 0, 0], [12, 3, 4, 0, 0], [20, 3, 0, 0, 0]],
                 dtype=np.int32))
    self.assertArraysEqual(results.decode_lengths,
                           np.array([3, 3, 2], dtype=np.int32))
    # softmax on logits of [0, 0, 0, 0, 1] reproduces:
    # [-1.904833   -1.904833   -1.904833   -1.904833   -0.90483296]
    self.assertAllClose(
        results.logprobs,
        np.array(
            [[1., -0.904832, -0.904832, 1., 1.],
             [1., -0.904832, -0.904832, 1., 1.], [1., -0.904832, 1., 1., 1.]],
            dtype=np.float32))


if __name__ == '__main__':
  absltest.main()
