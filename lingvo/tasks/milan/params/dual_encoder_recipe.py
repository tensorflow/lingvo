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
"""Helpers for defining Milan dual-encoder models."""

import functools

from lingvo.core import base_model_params
from lingvo.core import layers as lingvo_layers
from lingvo.core import optimizer
from lingvo.core import schedule
from lingvo.tasks.milan import constants
from lingvo.tasks.milan import dataset_spec
from lingvo.tasks.milan import dual_encoder
from lingvo.tasks.milan import input_generator


class RecipeError(Exception):
  pass


class DualEncoderRecipe(base_model_params.SingleTaskModelParams):
  """Base class that simplifies configuration of Milan dual encoder models.

  `DualEncoderRecipe` is a `SingleTaskModelParams` with extra builder-like
  methods for configuring the dual encoder (the `Task()` params) and input
  generators (advertised through `GetAllDatasetParams()`).

  In typical usage, model definitions subclass `DualEncoderRecipe`, call helper
  methods in the constructor to configure the dual encoder, and specify a
  `default_dataset` for the model to run on. For example::

    @model_registry.RegisterSingleTaskModel
    class MyExperiment(DualEncoderRecipe):
      def __init__(self):
        super().__init__()
        self.AddModality(
            'TEXT',
            input_feature='text_feature',
            id_feature='text_id',
            encoder=MyTextEncoder.Params(),
            encoder_output_dim=42)
        # Preprocess the raw 'image_feature' input prior to encoding.
        self.AddPreprocessor('image_feature', ImagePreprocessor.Params())
        self.AddModality(
            'IMAGE',
            input_feature='image_feature',
            id_feature='image_id',
            encoder=MyImageEncoder.Params(),
            encoder_output_dim=67)

      @property
      def default_dataset(self) -> DatasetSpec:
        # Point to your dataset of choice
        ...
  """

  def __init__(self):
    # Define these members here to make pytype happy.
    self.dataset = None
    self.input_params = None
    self.task_params = None

    self.dataset = self._ChooseDatasetSpec()
    # Base input params, be shared by both train and eval sets.
    self.input_params = input_generator.MilanInputGenerator.Params().Set(
        batch_size=64,
        # Run input pipeline on each TPU host (vs. one for all hosts) to
        # avoid input-boundedness.
        use_per_host_infeed=True)

    # Default optimization and checkpointer settings.
    self.task_params = dual_encoder.MilanTask.Params()
    self.task_params.train.Set(
        clip_gradient_norm_to_value=1.0,
        grad_norm_tracker=lingvo_layers.GradNormTracker.Params().Set(
            name='grad_norm_tracker',
            # Don't clip if the grad norm is already smaller than this.
            grad_norm_clip_cap_min=0.1),
        save_max_to_keep=2000,
        save_keep_checkpoint_every_n_hours=0.1667,  # At most every 10 min.
        optimizer=optimizer.Adam.Params().Set(
            beta1=0.9, beta2=0.999, epsilon=1e-8),
        learning_rate=0.0001,
        lr_schedule=schedule.StepwiseExponentialSchedule.Params().Set(
            decay=0.999, num_steps_per_decay=1000),
        tpu_steps_per_loop=100,
        max_steps=40000)

  def _ChooseDatasetSpec(self):
    """Returns the `DatasetSpec` to be used by the recipe."""
    return self.default_dataset

  @property
  def default_dataset(self) -> dataset_spec.DatasetSpec:
    """Returns a default dataset for the recipe to use.

    Subclasses should override this method to specify a dataset, or add logic
    (elsewhere) to choose the dataset at runtime, falling back to this one
    as the default.
    """
    raise NotImplementedError()

  @property
  def encoder_configs(self):
    return self.task_params.dual_encoder.encoder_configs

  def AddModality(self, name: str, **kwargs):
    config = dual_encoder.EncoderConfig().Set(**kwargs)
    self.encoder_configs[name] = config
    return config

  def AddPreprocessor(self, input_feature, preprocessor):
    self.input_params.preprocessors[input_feature] = preprocessor.Copy()

  def StartFromCheckpoint(self, checkpoint_path: str):
    """Configures the recipe to start training from the given model checkpoint.

    This is intended to be used in fine-tuning recipes. All variables, including
    Adam accumulators, are loaded from the checkpoint except for global step
    (so that it resets to 0 in new experiment) and grad norm tracker stats
    (since gradients are likely to have different moments in the new
    experiment).

    Args:
      checkpoint_path: Path of the checkpoint to start training from.
    """
    self.task_params.train.init_from_checkpoint_rules = {
        checkpoint_path: (
            [('(.*)', '%s')],
            # Don't load vars matching these regexes.
            ['.*grad_norm_tracker/.*', 'global_step'])
    }

  # Methods below implement the lingvo SingleTaskModelParams interface, allowing
  # the recipe to be registered with `RegisterSingleTaskModel()`.

  def GetAllDatasetParams(self):
    return {
        'Train':
            self.input_params.Copy().Set(
                name='Train',
                dataset_fn=functools.partial(
                    self.dataset.Read,
                    split=constants.Split.TRAIN,
                    shuffle_buffer_size=1024)),
        'Dev':
            self.input_params.Copy().Set(
                name='Dev',
                dataset_fn=functools.partial(
                    self.dataset.Read,
                    split=constants.Split.DEV,
                    shuffle_buffer_size=0)),
        'Test':
            self.input_params.Copy().Set(
                name='Test',
                dataset_fn=functools.partial(
                    self.dataset.Read,
                    split=constants.Split.TEST,
                    shuffle_buffer_size=0)),
    }

  def Task(self):
    task_params = self.task_params.Copy()
    if not task_params.dual_encoder.encoder_configs:
      raise RecipeError('Must configure at least one encoder.')

    assert task_params.dual_encoder.label_fn is None
    task_params.dual_encoder.label_fn = self.dataset.Label
    return task_params
