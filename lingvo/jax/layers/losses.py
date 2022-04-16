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
"""Loss functions."""

from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes

from jax_bitempered_loss import loss

InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


class BiTemperedLoss(base_layer.BaseLayer):
  """Bi-tempered logitstic loss.

  Bi-Tempered logistic loss is a generalized softmax cross-entropy loss function
  with a bounded loss value per sample and a heavy-tail softmax probability
  function. Temperature t1 < 1.0 controls the boundedness and t2 > 1.0 controls
  the tail heaviness of the softmax probabilities.

  Source: https://bit.ly/3jSol8T
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('t1', 1.0, 'Temperature 1 (log).')
    p.Define('t2', 1.0, 'Temperature 2 (exp).')
    p.Define('label_smoothing', 0.0, 'Label smoothing.')
    return p

  def fprop(self, logits: JTensor, labels: JTensor) -> JTensor:
    """Applies bi-tempered loss.

    Args:
      logits: The logits JTensor.  Shaped [..., num_classes].
      labels: The one-hot labels JTensor.  Shaped [..., num_classes].

    Returns:
      Loss values. Shaped either [...] or same as logits/labels but without the
      last dimension of size `num_classes`.
    """
    p = self.params
    loss_vals = loss.bi_tempered_logistic_loss(
        logits, labels, p.t1, p.t2, label_smoothing=p.label_smoothing)
    return loss_vals
