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
"""Multitask models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import py_utils


class SharedEncoderModel(base_model.MultiTaskModel):
  """Multitask model that shares encoder between tasks."""

  @classmethod
  def Params(cls):
    p = super(SharedEncoderModel, cls).Params()
    p.Define('encoder_to_share', None,
             'The task name whose encoder should be shared.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(SharedEncoderModel, self).__init__(params)
    p = self.params
    assert p.encoder_to_share in self.task_names

    # Assign the encoder from p.encoder_to_share task to all other tasks.
    # The assumption here is that all of the variables are initialized in the
    # __init__ function of the Task, and that they are connected together later
    # in the FProp function. So, here, after all of the Tasks have been
    # initialized, we can change references around to share what we need, and
    # later when FProp is called everything will be connected properly.
    encoder = self.GetTask(p.encoder_to_share).encoder
    for name in self.task_names:
      if name != p.encoder_to_share:
        task = self.GetTask(name)
        assert 'encoder' not in task.children
        task.AddChild('encoder', encoder)


class SharedEncoderDecoderModel(base_model.MultiTaskModel):
  """Multitask model that shares both encoder and decoder between tasks."""

  @classmethod
  def Params(cls):
    p = super(SharedEncoderDecoderModel, cls).Params()
    p.Define('encoder_to_share', None,
             'The task name whose encoder should be shared.')
    p.Define('decoder_to_share', None,
             'The task name whose decoder should be shared.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(SharedEncoderDecoderModel, self).__init__(params)
    p = self.params
    assert p.encoder_to_share in self.task_names
    assert p.decoder_to_share in self.task_names

    # Assign the encoder from p.encoder_to_share task to all other tasks, and
    # assign the decoder from p.decoder_to_share task to all other tasks.
    encoder = self.GetTask(p.encoder_to_share).encoder
    decoder = self.GetTask(p.decoder_to_share).decoder
    for name in self.task_names:

      if name != p.encoder_to_share:
        task = self.GetTask(name)
        assert 'encoder' not in task.children
        task.AddChild('encoder', encoder)

      if name != p.decoder_to_share:
        task = self.GetTask(name)
        assert 'decoder' not in task.children
        task.AddChild('decoder', decoder)


class RegExSharedVariableModel(base_model.MultiTaskModel):
  """Multitask models that share variables across different tasks.

  Note, do NOT use this model unless you know exactly what you are trying to do
  and you have verified that it indeed achieves what you would have expected.
  """

  @classmethod
  def Params(cls):
    p = super(RegExSharedVariableModel, cls).Params()
    p.Define(
        'variable_renaming_rules', None,
        'A list/tuple of variable renaming rules. Each element in the'
        ' list is a pair of strings. The first element is a regex'
        ' expression while the second element is a python format string.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    # Enable variable sharing.
    p = params
    with py_utils.OpportunisticVariableReuseScope():
      with py_utils.VariableRenameScope(p.variable_renaming_rules):
        super(RegExSharedVariableModel, self).__init__(params)
