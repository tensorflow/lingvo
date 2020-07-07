# Lint as: python3
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

import inspect

# Groups of parameter kinds.
DEFINABLE_PARAMETER_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                             inspect.Parameter.KEYWORD_ONLY)
IGNORABLE_PARAMETER_KINDS = (inspect.Parameter.VAR_POSITIONAL,
                             inspect.Parameter.VAR_KEYWORD)


def _IsDefinableParameter(parameter):
  """Checks if the parameter can be defined in `Params`.

  Args:
    parameter: inspect.Parameter to be checked.

  Returns:
    True if the `parameter`'s kind is either `POSITIONAL_OR_KEYWORD` or
    `KEYWORD_ONLY` which are definable in `Params`, False if it is either
    `VAR_POSITIONAL` or `VAR_KEYWORD` which are ignorable.

  Raises:
    ValueError: The `parameter` has another kind which are possibly not
      supported, e.g., `POSITIONAL_ONLY` parameters.
  """
  if parameter.kind in DEFINABLE_PARAMETER_KINDS:
    return True
  elif parameter.kind in IGNORABLE_PARAMETER_KINDS:
    return False
  else:
    raise ValueError('Unsupported parameter signature `%s` with kind `%s`.' %
                     (parameter.name, parameter.kind))


def _ExtractParameters(func, ignore, bound):
  """Extracts parameters of func which can be defined in Params.

  Args:
    func: A callable to be analysed.
    ignore: A collection of parameter names in `func` to be ignored.
    bound: Whether the `func` is used as a bound function (an object method or a
      class method) or not. If True, the first parameter of the `func` will be
      ignored.

  Returns:
    A generator of `inspect.Parameter` representing definable parameters.
  """
  ignore = set(ignore if ignore is not None else ())

  # Obtains parameter signatures.
  parameters = tuple(inspect.signature(func).parameters.values())
  # Ignores the bound parameter: typically `self` or `cls`.
  parameters = parameters[(1 if bound else 0):]
  # Filters unnecessary parameters.
  parameters = filter(_IsDefinableParameter, parameters)
  parameters = (p for p in parameters if p.name not in ignore)

  return parameters


def DefineParams(func, params, ignore=None, bound=False):
  """Defines params for each parameter of given callable.

  This allows you to define the parameters necessary to call a callable without
  having to type the Define statements yourself.
  Default values for the function parameters will be copied into the params
  object as well.

  To use this function for analysing a class instantiation, users usually can
  pass the class type as the `func`. If it does not work correctly, pass the
  `__init__` method of the class with `bound=True` instead.

  Args:
    func: A callable to be analysed. Parameters of this function will be defined
      in `params`. This function expects that `func` maintains the explicit
      signature of its parameters. Implicit parameters that are stored in
      `*args` or `**kwargs` could not be analysed correctly.
    params: A `Params` object to be updated. New parameters will be defined
      here.
    ignore: A collection of parameter names in `func` to be ignored from
      defining corresponding entries in `params`.
    bound: Whether `func` will be used as a bound function (an object method or
      a class method) or not. If True, the first parameter of `func` (typically
      `self` or `cls`) will be ignored.
  """
  for p in _ExtractParameters(func, ignore, bound):
    default = p.default
    if default is inspect.Parameter.empty:
      # TODO(oday): If Params supported required fields, change this behavior to
      # set the "required" flag.
      default = None

    params.Define(p.name, default, 'Function parameter.')


def _MakeArgs(func, params, **kwargs):
  """Generates an argument list to call func.

  Args:
    func: A callable to be called.
    params: A Params object containing arguments for `func`.
    **kwargs: Argument/value pairs that should override params.

  Returns:
    A dict containing function parameters to be used as `**kwargs` of `func`.
  """
  out_kwargs = {}

  # Here we set bound=False so the `func` is expected to be already a bound
  # function.
  for p in _ExtractParameters(func, ignore=None, bound=False):
    key = p.name

    # We will collect only args defined in at least either `kwargs` or `params`.
    # Args in `func`'s signature but in neither both will be skipped.

    if key in kwargs:
      # Anything in kwargs overrides parameters.
      out_kwargs[key] = kwargs[key]
    elif key in params:
      value = params.Get(key)
      # If the value in params is the same as the function default, we do not
      # set the arg so that we will let the function signature fill in this
      # parameter by itself.
      if value != p.default:
        out_kwargs[key] = value

  return out_kwargs


def CallWithParams(func, params, **kwargs):
  """Call a function or method with a Params object.

  Args:
    func: A callable to be called.
    params: A Params object containing parameters of `func`.
    **kwargs: Argument/value pairs that should override `params`.

  Returns:
    The return values from func.
  """
  return func(**_MakeArgs(func, params, **kwargs))


# TODO(oday): Remove this function and replace it with CallWithParams when the
# bug on the initializer of Keras layers has been resolved.
def ConstructWithParams(cls, params, **kwargs):
  """Constructs a class object with a Params object.

  Args:
    cls: A class type to be constructed.
    params: A Params object containing parameters of `cls.__init__`.
    **kwargs: Argument/value pairs that should override `params`.

  Returns:
    The constructed object.
  """
  return cls(**_MakeArgs(cls.__init__, params, **kwargs))
