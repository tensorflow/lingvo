# Lint as: python2, python3
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
"""Utilities to bind function signatures to params."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect


def DefineParamsFromArgs(func, params, ignore=None):
  """Defines params for each parameter of func.

  This allows you to define the parameters necessary to call a function
  or instantiate a class without having to type the Define statements yourself.
  Default values for the function parameters will be copied into the params
  object as well.

  To use this with a class constructor, use MyClass.__init__ as the func
  parameter.

  Args:
    func: A callable. Parameters for this function will be defined in `params`.
    params: A Params object to be updated. New parameters will be defined here.
    ignore: A collection of parameter names in `func` to be ignored from
      defining resulting entries in `params`.
  """
  # Get the call signature of the function.
  init_signature = inspect.signature(func)
  for parameter in init_signature.parameters.values():
    if parameter.name in ['self', 'args', 'kwargs']:
      continue
    if ignore and parameter.name in ignore:
      continue
    # Add each parameter of the constructor to the params object.
    # If the parameter has a default, add that too.
    params.Define(parameter.name, parameter.default, 'Function parameter.')


def _ExtractCallParams(func, params, **kwargs):
  """Extracts parameters from params that should be used to call func.

  Args:
    func: A function, constructor, or method.
    params: A hyperparams object containing arguments for func.
    **kwargs: Argument/value pairs that should override params.

  Returns:
    A dict containing function parameters.
  """
  # Read the list of parameters that func requires.
  init_signature = inspect.signature(func)
  func_params = {}
  for parameter in init_signature.parameters.values():
    key = parameter.name
    # Anything in kwargs overrides parameters.
    if key in kwargs:
      func_params[key] = kwargs[key]
      continue
    # If there's something in the function signature that's not in params,
    # skip it.
    if key not in params:
      continue
    # These are special function parameters that we should skip.
    if key in ['self', 'args', 'kwargs']:
      continue
    # If the value in params is the same as the function default, we will
    # let the function signature fill in this parameter for us.
    if init_signature.parameters[key].default == params.Get(key):
      continue
    func_params[key] = params.Get(key)
  return func_params


def CallWithParams(func, params, **kwargs):
  """Call a function or method with a hyperparams object.

  Args:
    func: A function or method.
    params: A hyperparams object with parameters to pass to func.
    **kwargs: Argument/value pairs that should override params.

  Returns:
    The return values from func.
  """
  return func(**_ExtractCallParams(func, params, **kwargs))


def ConstructWithParams(class_type, params, **kwargs):
  """Construct and object with a hyperparams object.

  Args:
    class_type: A class type.
    params: A hyperparams object with parameters to pass to the constructor.
    **kwargs: Argument/value pairs that should override params.

  Returns:
    The constructed object.
  """
  return class_type(**_ExtractCallParams(class_type.__init__, params, **kwargs))
