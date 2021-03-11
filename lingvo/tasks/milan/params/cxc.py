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
"""Image-text dual encoders for Crisscrossed Captions (CxC).

From the paper:
Parekh et al., "Crisscrossed Captions: Extended Intramodal and Intermodal
Semantic Similarity Judgments for MS-COCO". To appear in EACL 2021
(https://arxiv.org/abs/2004.15020).
"""

from lingvo import model_registry
from lingvo.tasks.milan import constants
from lingvo.tasks.milan import image_preprocessor
from lingvo.tasks.milan import tf_hub_layers
from lingvo.tasks.milan import transformers
from lingvo.tasks.milan.params import dual_encoder_recipe
from lingvo.tasks.milan.params import generic_datasets

IMAGE = constants.Modality.IMAGE
TEXT = constants.Modality.TEXT


class _BaseImageTextRecipe(dual_encoder_recipe.DualEncoderRecipe):
  """Common base for image-text recipes."""

  def __init__(self):
    super().__init__()
    self.task_params.dual_encoder.loss_weights = {
        (IMAGE, TEXT): 0.5,
        (TEXT, IMAGE): 0.5
    }

  def AddEfficientNetB4ImageEncoder(self,
                                    image_feature='image/encoded',
                                    id_feature='image/id'):
    self.input_params.features_to_read += [image_feature, id_feature]
    self.AddPreprocessor(
        image_feature,
        image_preprocessor.ImagePreprocessor.Params().Set(
            output_image_size=tf_hub_layers.EFFICIENTNET_B4_INPUT_SHAPE))

    self.AddModality(
        IMAGE,
        input_features=image_feature,
        id_feature=id_feature,
        encoder=tf_hub_layers.EfficientNetB4Params(),
        output_dim=tf_hub_layers.EFFICIENTNET_B4_OUTPUT_FEATURE_DIM)

  def AddBertAdapterTextEncoder(
      self,
      bert_embeddings_feature='text/bert/token_features',
      lengths_feature='text/bert/lengths',
      id_feature='text/id',
      output_dim=768):

    self.input_params.features_to_read += [
        bert_embeddings_feature, lengths_feature, id_feature
    ]
    input_embedding_dim = self.dataset.meta.features[
        bert_embeddings_feature].shape[-1]

    self.AddModality(
        TEXT,
        input_features=(bert_embeddings_feature, lengths_feature),
        id_feature=id_feature,
        encoder=transformers.GetTransformerStackWithEmbeddingInput(
            input_dim=input_embedding_dim,
            num_layers=3,
            hidden_dim=3072,
            num_attention_heads=8,
            output_dim=output_dim),
        output_dim=output_dim)


@model_registry.RegisterSingleTaskModel
class EfficientNetB4BertAdapter(_BaseImageTextRecipe):
  """EfficientNet-B4 + adapter on top of precomputed BERT features."""

  def __init__(self):
    super().__init__()
    self.AddEfficientNetB4ImageEncoder()
    self.AddBertAdapterTextEncoder()

  @property
  def default_dataset(self):
    return generic_datasets.ImageTextTFRecords.ParamsFromEnv().Instantiate()
