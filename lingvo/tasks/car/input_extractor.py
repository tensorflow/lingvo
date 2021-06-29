# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Input extractors.

Input extractors are an API for parsing and processing a set of fields from
serialized records.
"""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils

from lingvo.tasks.car import base_extractor

BaseExtractor = base_extractor._BaseExtractor  # pylint:disable=protected-access
BUCKET_UPPER_BOUND = base_extractor.BUCKET_UPPER_BOUND


################################################################################
# Extractors for car data.
################################################################################
class FieldsExtractor(base_layer.BaseLayer):
  """An API for parsing and processing a set of fields from serialized records.

  Input generators often need to parse several fields from a serialized record.
  This involves two stages: specifying the name and type of the fields to
  extract from serialized records (tf.Example or tf.SequenceExample), and then
  processing the raw output into a form to be consumed by higher-level callers.

  This class attempts to modularize this processing within the Minecraft input
  generators, so that users can easily create input generator pipelines that mix
  and match the composition of different fields from the same dataset.

  A descendant of this class will implement three functions:

    1) FeatureMap(): returning a dictionary of field names to field types, e.g.,
       'images' to tf.io.VarLenFeature(tf.string).  For PlainTextIterator
       datasets, FeatureMap() should be empty.

    2) _Extract(features): Given a 'features' dictionary containing the result
       from calling tf.io.parse_example or tf.parse_sequence_example on all
       extractors' features, produce a NestedMap of Tensors.

       NOTE: The return of the overall pipeline is a NestedMap of batched
       Tensors. However, the names and associations of the fields of each
       extractor are lost on the boundary of the map fn.  At the moment, one
       must implement _Extract() such that the names of the fields returned in
       the NestedMap matches self.Shape()'s keys; this is checked during the
       parent's Extract() call.

    3) Shape(): A NestedMap mapping names of outputs to their static shape,
       without the batch dimension.  In _InputBatch, this shape will be used to
       ensure that every output has a statically known shape.

  The caller of Extractors calls each extractor's FeatureMap() to populate the
  schema passed to tf.io.parse_example() or tf.parse_sequence_example(). The
  resulting dicationary of Tensors is then passed to each extractor's _Extract()
  function (via FieldsExtractor.Extract()) to return each extractor's output.

  It is the responsibility of the caller to maintain orders of outputs, since
  NestedMaps do not have any inherent ordering during iteration.
  """

  @classmethod
  def Params(cls):
    """Defaults params."""
    p = super().Params()
    p.name = cls.__name__
    return p

  def __init__(self, params):
    super().__init__(params)
    self.SetVariableFree()

  def FeatureMap(self):
    """Return a dictionary from tf.Example feature names to Features."""
    raise NotImplementedError()

  def ContextMap(self):
    """Return a dict mapping tf.SequenceExample context names to Features."""
    return {}

  def Extract(self, features):
    """Given 'feature' (Sparse)Tensors, output Tensors for consumption.

    NOTE: Implementation provided by subclasses's _Extract() method.

    Args:
      features: A dictionary of (Sparse)Tensors which includes tensors from all
        extractors.

    Returns:
      A NestedMap of output Tensors.
    """
    outputs = self._Extract(features)
    shapes = self.Shape()
    assert outputs.IsCompatible(shapes), '{} vs. {}'.format(
        outputs.DebugString(), shapes.DebugString())
    return outputs

  def ExtractBatch(self, features):
    """Given 'features' batched Tensors, output Tensors for consumption.

    NOTE: Implementation provided by subclasses's _ExtractBatch() method.

    Args:
      features: A dictionary of Tensors which includes tensors from this
        extractor.

    Returns:
      A NestedMap of batched output Tensors.
    """
    shapes = self.Shape()
    outputs = self._ExtractBatch(features)
    # Add batch dimension to shape.
    shapes = shapes.Transform(lambda x: [None] + x)
    assert outputs.IsCompatible(shapes), '{} vs. {}'.format(
        outputs.DebugString(), shapes.DebugString())
    return outputs

  def Filter(self, outputs):
    """Return the bucket based on the result of Extract().

    This function should return 1 if the example should pass through without
    being dropped, and a value in [BUCKET_UPPER_BOUND, inf) if the example
    should be dropped.  Currently no other bucketing strategies are supported.

    Args:
      outputs: The NestedMap returned by this extractor's _Extract() function.
        This is useful to implement filtering based on the values of the
        extracted example.

    Returns:
      A scalar bucket id.
    """
    del outputs
    return 1

  def FilterBatch(self, outputs):
    """Like Filter but runs over batches of outputs.

    This function should be called to decide whether the entire batch should be
    dropped.  Downstream implementations that do not run within an input
    pipeline must figure out how to handle these outputs, if filtering at the
    batch level is desired.

    Args:
      outputs: A NestedMap of preprocessed Tensors.

    Returns:
      A scalar bucket id.
    """
    del outputs
    return 1

  def Shape(self):
    """Return a NestedMap of un-batched fully-specified tf.TensorShapes."""
    raise NotImplementedError()

  def DType(self):
    """Return a NestedMap mapping names to tf.DType."""
    raise NotImplementedError()

  def _Extract(self, features):
    """The subclass-defined implementation of Extract().

    Args:
      features: A dictionary of (Sparse)Tensors which includes tensors from this
        extractor.

    Returns:
      A NestedMap of output Tensors whose key names match self.Shape()'s keys.
    """
    raise NotImplementedError()

  def _ExtractBatch(self, features):
    """The subclass-defined implementation of ExtractBatch().

    Args:
      features: A dictionary of batched Tensors including tensors from this
        extractor.

    Returns:
      A NestedMap of output Tensors whose key names match self.Shape()'s keys.
    """
    # Default implementation uses map_fn.
    result = tf.map_fn(
        self._Extract, elems=features, dtype=self.DType(), back_prop=False)
    return py_utils.NestedMap(result)


class NestedFieldsExtractor(FieldsExtractor):
  """A nested fields extractor that contains multiple sub fields extractors."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'extractors', py_utils.NestedMap(),
        'A map of sub-extractors that are FieldsExtractor. The output of '
        'the sub-extractors will be nested under corresponding key.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self.CreateChildren('extractors', self.params.extractors)

  def FeatureMap(self):
    feature_map = {}
    for extractor in self.extractors.Flatten():
      feature_map.update(extractor.FeatureMap())
    return feature_map

  def _Extract(self, features):
    ret = py_utils.NestedMap()
    for key, extractor in self.extractors.FlattenItems():
      ret.Set(key, extractor.Extract(features))
    return ret

  def Shape(self):
    shapes = py_utils.NestedMap()
    for key, extractor in self.extractors.FlattenItems():
      shapes.Set(key, extractor.Shape())
    return shapes

  def DType(self):
    dtypes = py_utils.NestedMap()
    for key, extractor in self.extractors.FlattenItems():
      dtypes.Set(key, extractor.DType())
    return dtypes


class LaserExtractor(FieldsExtractor):
  """Interface for extracting laser data.

  Must produce:
    points_xyz: [max_num_points, 3] - XYZ coordinates of laser points.

    points_feature: [max_num_points, num_features] - Features for each point in
      points_xyz.

    points_padding: [max_num_points]: Padding for points.  0 means the
      corresponding point is the original, and 1 means there is no point
      (xyz or feature) present.  Only present if max_num_points is not
      None.

  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('max_num_points', None, 'The number of points per spin.')
    p.Define('num_features', 1, 'Number of features per laser point.')
    return p

  def Shape(self):
    p = self.params
    ret = py_utils.NestedMap(
        points_xyz=tf.TensorShape([p.max_num_points, 3]),
        points_feature=tf.TensorShape([p.max_num_points, p.num_features]))
    if p.max_num_points is not None:
      ret.points_padding = tf.TensorShape([p.max_num_points])
    return ret

  def DType(self):
    ret = py_utils.NestedMap(points_xyz=tf.float32, points_feature=tf.float32)
    if self.params.max_num_points is not None:
      ret.points_padding = tf.float32
    return ret
