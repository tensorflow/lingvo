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
"""Base classes for defining datasets."""

import abc
import copy
import functools

from typing import Callable, Dict, Optional, Union

from absl import logging
import attr
from lingvo import compat as tf
from lingvo.tasks.milan import constants
from lingvo.tasks.milan import labels as label_lib

_default_items_per_example = {
    constants.Modality.AUDIO: 1,
    constants.Modality.IMAGE: 1,
    constants.Modality.TEXT: 1
}


@attr.s(kw_only=True)
class Metadata:
  """Describes the `tf.data.Dataset`s that a `DatasetSpec` instance produces."""

  # Names, shapes, and dtypes of features in the resulting dataset.
  # Shapes reflect those of a single, unbatched example.
  features: Optional[Dict[str, tf.TensorSpec]] = attr.ib(default=None)

  items_per_example = attr.ib(factory=_default_items_per_example.copy)
  # Map of modality name to batch shape.
  modality_batch_shapes = attr.ib()

  @modality_batch_shapes.default
  def _InferDefaultBatchShapes(self):
    return {
        modality: tf.TensorShape([None] if n == 1 else [None, n])
        for modality, n in self.items_per_example.items()
        if n > 0
    }


class DatasetSpec(metaclass=abc.ABCMeta):
  """Abstract base class for dataset definitions."""

  @abc.abstractmethod
  def Read(self,
           split: str,
           batch_size: Optional[int] = None,
           **kwargs) -> 'tf.data.Dataset':
    """Reads the specified split of the dataset as a `tf.data.Dataset`."""
    pass

  @abc.abstractmethod
  def Label(self, pairs: label_lib.ExamplePairs) -> tf.Tensor:
    """Computes labels for the given query-result pairs.

    Args:
      pairs: The examples containing the queries and results to label.

    Returns:
      A Tensor containing a label for every query-result pair.
    """
    pass

  @property
  @abc.abstractmethod
  def meta(self) -> Metadata:
    """Returns `Metadata` describing the features in this dataset."""
    pass


class FileBasedDatasetSpec(DatasetSpec):
  """Template for file-based datasets.

  `Read()` implements a typical tf.data pipeline that
    - loads serialized records from input files,
    - extracts dict-format examples from them, and
    - applies an example-level transformation, e.g. to rename or reshape
      features.

  Constructor arguments specify where the files are located (`split_paths`)
  and define how to read records from the files (`file_reader`), extract
  examples from the records (`parser`), and transform the resulting examples
  (`transform`).
  """

  def __init__(self,
               *,
               split_paths,
               file_reader: Callable,
               parser: Callable,
               transform: Optional[Callable] = None,
               label_fn: label_lib.LabelFnType,
               metadata: Optional[Metadata] = None):
    """Creates an instance that reads examples from files at `split_paths`.

    Args:
      split_paths: File paths for each split, keyed by split name. Each split
        can be specified as a file pattern or list of file patterns, e.g.
        '/foo/ba*' or ['/foo/bar-*', '/foo/baz*'].
      file_reader: Callable that takes filenames => tf.data.Dataset (e.g.
        tf.data.TFRecordDataset).
      parser: Callable that converts raw file records into parsed examples
        (e.g. `lambda t: tf.io.parse_example(t, features=...)`).
      transform: Optional transformation to apply to the parsed examples. If
        given, should be a function Dict[str, Tensor] -> Dict[str, Tensor].
      label_fn: Function that generates pairwise labels between batches of
        queries and results read from this dataset.
      metadata: `Metadata` object that describes examples in this dataset.
    """
    self._split_paths = dict(split_paths)
    self._file_reader = file_reader
    self._parser = parser
    self._label_fn = label_fn
    self._transform = transform
    self._meta = metadata or Metadata()

  def Read(self,
           split: str,
           batch_size: Optional[int] = None,
           input_filepattern=None,
           shuffle_buffer_size: Optional[int] = None,
           num_examples: int = -1,
           num_epochs: int = -1,
           read_parallelism: int = 16) -> 'tf.data.Dataset':
    """Reads the specified `split` as a `tf.data.Dataset`.

    Reads the files at the path configured for `split`, parses the examples,
    and optionally shuffles and batches them.

    By default, the file paths are taken from the `split_paths` passed to
    the constructor. Callers can optionally override this path by passing
    `input_filepattern` explicitly.

    If `shuffle_buffer_size` is set, examples are randomly shuffled with a
    buffer of the given size prior to batching. Shuffling is on by default for
    the `TRAIN` split, and can be disabled by setting shuffle_buffer_size=0.

    Args:
      split: Name of the split to read.
      batch_size: If set, the number of examples to batch together.
      input_filepattern: If given, read the tfrecords at this path instead of
        the one specified in `split_paths`.
      shuffle_buffer_size: If > 0, size of the buffer used to shuffle examples.
        Defaults to 4096 for the `TRAIN` split and 0 otherwise.
      num_examples: Number of examples to read from the underlying tfrecords.
        Defaults to -1, meaning all examples are read.
      num_epochs: Number of epochs (full passes) through the dataset to make.
        Defaults to -1, meaning the dataset repeats indefinitely. If set to n >
        0, `tf.errors.OutOfRangeError` at the end of the n'th epoch.
      read_parallelism: Number of input files to read in parallel.

    Returns:
      The split as a `tf.data.Dataset` object.
    """
    if input_filepattern is None:
      input_filepattern = self._split_paths.get(split)
      assert input_filepattern, 'Unsupported split {}'.format(split)

    if shuffle_buffer_size is None:
      shuffle_buffer_size = 4096 if split == constants.Split.TRAIN else 0
    shuffle_files = shuffle_buffer_size > 0

    logging.info(
        'Reading inputs %s with batch_size=%s, shuffle_buffer_size=%s,'
        'num_examples=%d, num_epochs=%d', input_filepattern, batch_size,
        shuffle_buffer_size, num_examples, num_epochs)

    dataset = (
        tf.data.Dataset.list_files(input_filepattern,
                                   shuffle=shuffle_files).apply(
                                       tf.data.experimental.parallel_interleave(
                                           self._file_reader,
                                           cycle_length=read_parallelism,
                                           sloppy=shuffle_files)))

    if shuffle_buffer_size > 0:
      dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.take(num_examples).repeat(num_epochs)
    if batch_size is not None:
      dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.map(self._parser)
    if self._transform is not None:
      dataset = dataset.map(self._transform)
    return dataset

  def Label(self, pairs: label_lib.ExamplePairs) -> tf.Tensor:
    return self._label_fn(pairs)

  @property
  def meta(self):
    if self._meta.features is None:
      self._meta.features = self.Read(
          constants.Split.TRAIN, shuffle_buffer_size=0,
          read_parallelism=1).element_spec
    return self._meta


# Pytype alias.
Schema = Dict[str, Union[tf.io.FixedLenFeature, tf.io.FixedLenSequenceFeature]]

# Size of TFRecord reader buffer, per file. (The TFRecordDataset default, 256KB,
# is too small and makes TPU trainers input-bound.)
_TFRECORD_READER_BUFFER_SIZE_BYTES = 64 * (1 << 20)  # 64 MiB


class TFRecordDatasetSpec(FileBasedDatasetSpec):
  """Base class for datasets stored in TFRecord files.

  This is a specialization of `FileBasedDatasetSpec` for the case where input
  files are TFRecords of `tf.train.Example` protos.
  """

  def __init__(self,
               *,
               split_paths,
               schema: Schema,
               transform: Optional[Callable] = None,
               label_fn: label_lib.LabelFnType,
               metadata: Optional[Metadata] = None):
    """Creates an instance that reads examples from tfrecords at `split_paths`.

    Args:
      split_paths: Paths of the tfrecord files for each split, keyed by split
        name. Each split can be specified as a file pattern or list of file
        patterns (e.g. '/foo/ba*' or ['/foo/bar-*', '/foo/baz*']).
      schema: Feature definitions for the `tf.train.Example`s in this dataset,
        in the format expected by `tf.io.parse_example`.
      transform: Optional transformation to apply to the parsed examples. If
        given, should be a function Dict[str, Tensor] -> Dict[str, Tensor].
      label_fn: Function that generates pairwise labels between batches of
        queries and results read from this dataset.
      metadata: `Metadata` object that describes examples in this dataset.
    """
    super().__init__(
        split_paths=split_paths,
        file_reader=functools.partial(
            tf.data.TFRecordDataset,
            buffer_size=_TFRECORD_READER_BUFFER_SIZE_BYTES),
        parser=functools.partial(
            tf.io.parse_example, features=copy.deepcopy(schema)),
        transform=transform,
        label_fn=label_fn,
        metadata=metadata)
