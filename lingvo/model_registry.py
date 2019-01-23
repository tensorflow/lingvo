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
"""Convenience helpers for managing Params for datasets and models.

Typical usage will be to define and register a subclass of ModelParams
for each dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import tensorflow as tf
from lingvo.core import base_model_params

tf.flags.DEFINE_string(
    'model_params_override', '', 'Optional text specifying'
    ' model_class.Model() params to be overridden in'
    ' Params.FromText format.  Each param must be separated by'
    ' a newline or a semicolon.  This is only intended for'
    ' quick experimentation.  Only one of'
    ' --model_params_override and --model_params_file_override'
    ' may be specified.')
tf.flags.DEFINE_string(
    'model_params_file_override', '', 'Optional text file for'
    ' overwriting model_class.Model() params in Params.FromText'
    ' format. Each param must occur on a single line.  Only one'
    ' of --model_params_override and'
    ' --model_params_file_override may be specified.')

FLAGS = tf.flags.FLAGS


def _MaybeUpdateParamsFromFlags(cfg):
  """Updates Model() Params from flags if set."""
  if FLAGS.model_params_override and FLAGS.model_params_file_override:
    raise ValueError('Only one of --model_params_override and'
                     ' --model_params_file_override may be specified.')

  if FLAGS.model_params_override:
    params_override = FLAGS.model_params_override.replace(';', '\n')
    tf.logging.info('Applying params overrides:\n%s\nTo:\n%s', params_override,
                    cfg.ToText())
    cfg.FromText(params_override)
  if (FLAGS.model_params_file_override and
      tf.gfile.Exists(FLAGS.model_params_file_override)):
    params_override = tf.gfile.FastGFile(FLAGS.model_params_file_override,
                                         'r').read()
    tf.logging.info('Applying params overrides from file %s:\n%s\nTo:\n%s',
                    FLAGS.model_params_file_override, params_override,
                    cfg.ToText())
    cfg.FromText(params_override)


class _ModelRegistryHelper(object):
  # Global dictionary mapping subclass name to registered ModelParam subclass.
  _MODEL_PARAMS = {}
  # Global set of modules from which ModelParam subclasses have been registered.
  _REGISTERED_MODULES = set()

  @classmethod
  def _ClassPathPrefix(cls):
    return 'lingvo.tasks.'

  @classmethod
  def _ModelParamsClassKey(cls, src_cls):
    """Returns a string key used for `src_cls` in the model registry.

    The returned key is a period separated string. E.g., image.mnist.LeNet5. It
    roughly reflects how params files are organized. We put some of the
    directory information into the key to avoid future model name conflicts.

    Args:
      src_cls: A subclass of `~.base_model.BaseModel`.
    """
    path = src_cls.__module__
    # Removes the prefix.
    path_prefix = cls._ClassPathPrefix()
    path = path.replace(path_prefix, '')

    # Removes '.params' if exists.
    if '.params' not in path:
      # Sometimes, we define a param class in a unittest.
      if not inspect.getfile(src_cls).endswith('test.py'):
        raise ValueError('Model params being registered must be '
                         'in a params subfolder or a test.')
      return 'test.{}'.format(src_cls.__name__)
    path = path.replace('.params', '')
    return '{}.{}'.format(path, src_cls.__name__)

  @classmethod
  def _GetSourceInfo(cls, src_cls):
    """Gets a source info string given a source class."""
    return '%s@%s:%d' % (cls._ModelParamsClassKey(src_cls),
                         inspect.getsourcefile(src_cls),
                         inspect.getsourcelines(src_cls)[-1])

  @classmethod
  def _RegisterModel(cls, wrapper_cls, src_cls):
    """Registers a ModelParams subclass in the global registry."""
    key = cls._ModelParamsClassKey(src_cls)
    module = src_cls.__module__
    if key in cls._MODEL_PARAMS:
      raise ValueError('Duplicate model registered for key {}: {}.{}'.format(
          key, module, src_cls.__name__))

    tf.logging.debug('Registering model %s', key)
    # Log less frequently (once per module) but at a higher verbosity level.
    if module not in cls._REGISTERED_MODULES:
      tf.logging.info('Registering models from module: %s', module)
      cls._REGISTERED_MODULES.add(module)

    # Decorate param methods to add source info metadata.
    cls._MODEL_PARAMS[key] = wrapper_cls
    return key

  @classmethod
  def _CreateWrapperClass(cls, src_cls):
    """Creates a wrapper class for model params that adds source info."""

    # Python2 class decorators that return a different class are fundamentally
    # broken (technically, they are fine but canonical use of super() is
    # broken). Also, fallback mechanisms don't exist in python3. So, we only
    # decorate the version of the class that we register, but any decorators
    # should return the original class for maximum compatibility.
    # When the python3 super() is used, it should be possible to return this
    # from the decorators too.

    class Registered(src_cls):
      REGISTERED_SOURCE_INFO = cls._GetSourceInfo(src_cls)

      # Extend model to annotate source information.
      @classmethod
      def Model(cls):
        """Wraps BaseTask params into SingleTaskModel params."""
        p = super(Registered, cls).Model()
        p.model = cls.REGISTERED_SOURCE_INFO
        return p

    # So things show up in messages well.
    Registered.__name__ = src_cls.__name__
    return Registered

  @classmethod
  def RegisterSingleTaskModel(cls, src_cls):
    """Class decorator that registers a `.SingleTaskModelParams` subclass."""
    if not issubclass(src_cls, base_model_params.SingleTaskModelParams):
      raise TypeError(
          'src_cls %s is not a SingleTaskModelParams!' % src_cls.__name__)
    cls._RegisterModel(cls._CreateWrapperClass(src_cls), src_cls)
    return src_cls

  @classmethod
  def RegisterMultiTaskModel(cls, src_cls):
    """Class decorator that registers a `.MultiTaskModelParams` subclass."""
    if not issubclass(src_cls, base_model_params.MultiTaskModelParams):
      raise TypeError(
          'src_cls %s is not a MultiTaskModelParams!' % src_cls.__name__)
    cls._RegisterModel(cls._CreateWrapperClass(src_cls), src_cls)
    return src_cls

  @staticmethod
  def GetAllRegisteredClasses():
    """Returns global registry map from model names to their param classes."""
    all_params = _ModelRegistryHelper._MODEL_PARAMS
    if not all_params:
      tf.logging.warning('No classes registered.')
    return all_params

  @classmethod
  def GetClass(cls, class_key):
    """Returns a ModelParams subclass with the given `class_key`.

    Args:
      class_key: string key of the ModelParams subclass to return.

    Returns:
      A subclass of `~.base_model_params._BaseModelParams`.

    Raises:
      LookupError: If no class with the given key has been registered.
    """
    all_params = cls.GetAllRegisteredClasses()
    if class_key not in all_params:
      raise LookupError('Model %s not found. Known models:\n%s' %
                        (class_key, '\n'.join(sorted(all_params.keys()))))
    return all_params[class_key]

  @classmethod
  def GetParams(cls, class_key, dataset_name):
    """Constructs a `Params` object for given model and dataset, obeying flags.

    In case of default model, params may be updated based on the flags
    `--model_params_override` or `--model_params_file_override`.

    Args:
      class_key: String class key (i.e. `image.mnist.LeNet5`).
      dataset_name: Method to generate dataset params (i.e. 'Test').

    Returns:
      Full `~.hyperparams.Params` for the model class.
    """
    model_params_cls = cls.GetClass(class_key)
    cfg = model_params_cls.Model()
    cfg.input = model_params_cls.GetDatasetParams(dataset_name)

    _MaybeUpdateParamsFromFlags(cfg)
    return cfg


# pyformat: disable
# pylint: disable=invalid-name
RegisterSingleTaskModel = _ModelRegistryHelper.RegisterSingleTaskModel
RegisterMultiTaskModel = _ModelRegistryHelper.RegisterMultiTaskModel
GetAllRegisteredClasses = _ModelRegistryHelper.GetAllRegisteredClasses
GetClass = _ModelRegistryHelper.GetClass
GetParams = _ModelRegistryHelper.GetParams
# pylint: enable=invalid-name
# pyformat: enable
