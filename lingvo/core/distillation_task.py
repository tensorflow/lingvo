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
"""Base class for tasks that implement knowledge distillation."""
from lingvo import compat as tf
from lingvo.core import base_model
from lingvo.core import py_utils
from lingvo.core import schedule


class DistillationTask(base_model.BaseTask):
  """A task to distill knowledge from a teacher task to a student task.

  The training parameters (e.g., learning rate) are determined only by
  `DistillationTask.params.train`. Teacher and student task's training and eval
  parameters must be set to None.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('teacher', None, 'The teacher task params.')
    p.Define('student', None, 'The student task params.')
    p.Define(
        'distillation_loss_weight',
        # Only uses distillation loss by default.
        schedule.ConstantOne.Params(),
        'A schedule of distillation loss weight. '
        'The weight determines the fraction of total loss contributed by '
        'distillation loss, while the rest loss will be computed against '
        'the ground truth. '
        'A weight of 0 means to only use ground-truth and ignore teacher '
        'predictions, while a weight 1 means to only use teacher '
        'predictions and ignore ground truth. '
        'The weight is specified as a schedule to allow it to change '
        'during training.')
    p.Define(
        'teacher_target_type', 'truth', 'The target type for the teacher. '
        'Choices are: '
        ' "truth": using the ground-truth target labels '
        ' "beam": using the 1-best hypothesis from the beam search.')
    p.Define(
        'beam_search_temperature', 1.0, 'The temperature to scale the'
        'log-prob of each beam search hypothesis. This is used in '
        'training only')
    p.Define(
        'train_teacher', False, 'Adds the teacher\'s loss (w.r.t the ground '
        'truth labels) to the overall ground truth loss. This can be used for '
        'instance when the teacher is trained in parallel to the student.')
    return p

  def __init__(self, params):
    assert issubclass(params.cls, DistillationTask)
    super().__init__(params)

    p = self.params

    # This seems to be root cause of teacher/student input ops being
    # incorrectly placed on TPU.
    if not py_utils.use_tpu():
      # While student does not need its own input generator for training, it
      # needs an input generator for inference graphs.
      p.student.input = p.input
      # Teacher also might need an input generator, eg. for waveform_processor.
      p.teacher.input = p.input
    for child in ('teacher', 'student'):
      child_p = getattr(p, child)
      assert issubclass(child_p.cls, base_model.BaseTask)
      assert child_p.train is None
      assert child_p.eval is None
      # In theory it's ok for teacher to be a DistillationTask. In practice
      # it probably won't happen.
      assert not issubclass(child_p.cls, DistillationTask)
      child_p.name = child
      self.CreateChild(child, child_p)
    self.CreateChild('distillation_loss_weight', p.distillation_loss_weight)

  def ComputePredictions(self, theta, input_batch):
    p = self.params
    with tf.name_scope(p.name):
      if p.teacher_target_type == 'truth':
        teacher_predictions = self.teacher.ComputePredictions(
            theta.teacher, input_batch)
        student_predictions = self.student.ComputePredictions(
            theta.student, input_batch)
        return py_utils.NestedMap(
            teacher=teacher_predictions, student=student_predictions)
      elif p.teacher_target_type == 'beam':
        (teacher_predictions, teacher_input_batch,
         teacher_beam_prob) = self.teacher.ComputeBeamPredictions(
             theta.teacher, input_batch, p.beam_search_temperature)
        # We use 'teacher_input_batch' instead of 'input_batch' for 'student'
        # because the training of student network uses target transcripts for
        # the "teacher forcing" mode and here the target transcripts should come
        # from the teacher's beam search.
        student_predictions = self.student.ComputePredictions(
            theta.student, teacher_input_batch)
        return py_utils.NestedMap(
            teacher=teacher_predictions,
            student=student_predictions,
            teacher_beam_prob=teacher_beam_prob)
      else:
        raise ValueError('teacher target type not defined properly: %s' %
                         self.p.teacher_target_type)

  def ComputeLoss(self, theta, predictions, input_batch):
    p = self.params
    per_example = {}
    with tf.name_scope('groundtruth_loss'):
      student_groundtruth_loss, student_groundtruth_per_example = (
          self.student.ComputeLoss(theta.student, predictions.student,
                                   input_batch))
      groundtruth_loss = student_groundtruth_loss
      groundtruth_loss['student_groundtruth_loss'] = (
          student_groundtruth_loss['loss'])
      per_example.update(student_groundtruth_per_example)

      if p.train_teacher:
        teacher_groundtruth_loss, _ = self.teacher.ComputeLoss(
            theta.teacher, predictions.teacher, input_batch)
        groundtruth_loss['teacher_groundtruth_loss'] = (
            teacher_groundtruth_loss['loss'])
        # The new loss is the wighted sum of the teacher and student losses.
        groundtruth_loss['loss'] = py_utils.WeightedAvg(*zip(
            teacher_groundtruth_loss['loss'], student_groundtruth_loss['loss']))

    with tf.name_scope('distillation_loss'):
      distillation_loss, distill_per_example = self.ComputeDistillationLoss(
          theta, predictions, input_batch)
      distillation_loss['distillation_loss'] = distillation_loss['loss']
      per_example.update(distill_per_example)

    distillation_loss_weight = self.distillation_loss_weight.Value()
    metrics = py_utils.CombineMetrics([
        (groundtruth_loss, 1 - distillation_loss_weight),
        (distillation_loss, distillation_loss_weight),
    ])
    return metrics, per_example

  def ComputeDistillationLoss(self, theta, predictions, input_batch):
    raise NotImplementedError('Abstract method')

  def BProp(self):
    p = self.params
    if p.train_teacher:
      return super().BProp()
    else:
      # Only bprop on student variables.
      with py_utils.TaskCallScope(self):  # Support bprop for TPU embedding.
        self._BPropForVariables(self.student.vars)

  def Decode(self, input_batch):
    return self.student.Decode(input_batch)

  def Inference(self):
    return self.student.Inference()

  def CreateDecoderMetrics(self):
    return self.student.CreateDecoderMetrics()

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    return self.student.PostProcessDecodeOut(dec_out_dict, dec_metrics_dict)
