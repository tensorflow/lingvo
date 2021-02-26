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
"""Alternative to tf.summary for TPU without host call and outside compilation.

To add a summary, in layer implementation code call::

  tpu_summary.scalar('name', value)

Then in trainer code::

  my_model = my_model_params.Instantiate()
  with tpu_summary.context():
    my_model.ConstructFPropBPropGraph()
    summaries = tpu_summary.merge_all()

If some summaries are created inside a while loop::

  my_model = my_model_params.Instantiate()
  with tpu_summary.context(rewrite_while_loop=True):
    output = my_model.GreedyDecodeIds()
    summaries = tpu_summary.merge_all()

Note that you must call tpu_summary.merge_all() inside tpu_summary.context(),
or else it will return nothing.

Note also that merge_all() returns a dict of tensors, not a serialized proto.
As such, merge_all() can be used inside tpu rewrite context.
It is left up to the caller to decide how to return summary tensors from TPU
and what to do next.
"""

import contextlib

from lingvo import compat as tf


class TpuSummaryScalar:
  """Plain data object. Much better than namedtuple."""
  name = None
  value = None
  name_scope = None
  while_loop_reduce = None


class PwTpuSummaryTensor:
  """A summary tensor with the first dim being the batch."""
  name = None
  value = None
  name_scope = None
  # TODO(yonghui): Deal with while_loop context.


class TpuSummaryContext:
  """Non-reentrant context that holds the list of summary tensors."""

  _global_stack = []

  @classmethod
  def current(cls):
    """Returns current context or None."""
    if cls._global_stack:
      return cls._global_stack[-1]
    else:
      return None

  def __init__(self):
    self.summary_tensors = []

  def __enter__(self):
    assert not self._global_stack, 'no re-entry'
    self._global_stack.append(self)

  def __exit__(self, *args):
    self._global_stack.pop()


class RewriteLoopContext:
  """Context manager. Rewrites tf.while_loop to propagate summary tensors."""

  def __init__(self, max_loop_vars=16, dtype=tf.float32):
    self.max_loop_vars = max_loop_vars
    self.dtype = dtype
    self.tf_while_loop = None

  def __enter__(self):
    self.tf_while_loop = tf.while_loop
    tf.while_loop = self.while_loop

  def __exit__(self, *args):
    tf.while_loop = self.tf_while_loop
    self.tf_while_loop = None

  def while_loop(self, cond, body, loop_vars, **kwargs):
    """Wrapper for tf.while_loop that adds summary_tensors to loop vars."""
    ctx = TpuSummaryContext.current()
    assert ctx is not None, 'must be inside TpuSummaryContext'
    assert self.tf_while_loop, 'must be inside self RewriteLoopContext'

    outer_summary_tensors = ctx.summary_tensors
    ctx.summary_tensors = []

    accumulators = [
        tf.constant(0, dtype=self.dtype) for _ in range(self.max_loop_vars)
    ]
    size = kwargs['maximum_iterations']  # as if not None
    arrays = [
        tf.TensorArray(self.dtype, size) for _ in range(self.max_loop_vars)
    ]
    loop_count = tf.constant(0, dtype=tf.int32)

    def loop_body(loop_vars, accumulators, arrays, loop_count):
      loop_vars = body(*loop_vars)
      del ctx.summary_tensors[self.max_loop_vars:]
      for i, x in enumerate(ctx.summary_tensors):
        if x.while_loop_reduce == 'stack':
          arrays[i] = arrays[i].write(loop_count, tf.cast(x.value, self.dtype))
        else:
          accumulators[i] += tf.cast(x.value, self.dtype)
      loop_count += 1
      return loop_vars, accumulators, arrays, loop_count

    def loop_cond(loop_vars, accumulators, arrays, loop_count):
      del accumulators, arrays, loop_count
      return cond(*loop_vars)

    loop_vars, accumulators, arrays, loop_count = self.tf_while_loop(
        cond=loop_cond,
        body=loop_body,
        loop_vars=(loop_vars, accumulators, arrays, loop_count),
        **kwargs)

    for i, x in enumerate(ctx.summary_tensors):
      if x.while_loop_reduce == 'stack':
        x.value = arrays[i].stack()
      elif x.while_loop_reduce == 'mean':
        denominator = tf.cast(tf.math.maximum(1, loop_count), self.dtype)
        x.value = accumulators[i] / denominator
      else:
        x.value = accumulators[i]

    ctx.summary_tensors = outer_summary_tensors + ctx.summary_tensors
    return loop_vars


def scalar(name, value, while_loop_reduce='mean'):
  """Adds summary scalar.

  Outside of tpu_summary.context() does nothing.

  Args:
    name: string name
    value: scalar tensor value
    while_loop_reduce: optional argument, determines what to do when this
      summary appears inside a tf.while_loop. Can be 'mean' or 'sum'.
  """
  assert while_loop_reduce in ('mean', 'sum')
  ctx = TpuSummaryContext.current()
  if ctx is None:
    return
  x = TpuSummaryScalar()
  x.name = str(name)
  x.value = tf.convert_to_tensor(value)
  if x.value.shape != ():  # pylint: disable=g-explicit-bool-comparison
    raise ValueError('use tpu_summary.tensor() instead: %r' % value)
  x.name_scope = tf.get_default_graph().get_name_scope()
  x.while_loop_reduce = while_loop_reduce
  ctx.summary_tensors.append(x)


def tensor(name, value):
  """Adds summary tensor. Similar to scalar() but allows other shapes."""
  ctx = TpuSummaryContext.current()
  if ctx is None:
    return
  x = TpuSummaryScalar()
  x.name = str(name)
  x.value = tf.convert_to_tensor(value)
  x.name_scope = tf.get_default_graph().get_name_scope()
  x.while_loop_reduce = 'stack'
  ctx.summary_tensors.append(x)


def pw_tensor(name, value):
  """Adds summary tensor."""
  ctx = TpuSummaryContext.current()
  if ctx is None:
    return
  x = PwTpuSummaryTensor()
  x.name = str(name)
  x.value = tf.convert_to_tensor(value)
  x.name_scope = tf.get_default_graph().get_name_scope()
  ctx.summary_tensors.append(x)


def merge_all():
  """Returns all summary tensors as a dict of {name: tensor}.

  Note that this is not the same return type as tf.summary.merge_all
  which returns a serialized proto string.

  Outside of tpu_summary.context() returns {}
  """
  ctx = TpuSummaryContext.current()
  if ctx is None:
    return {}
  g = tf.get_default_graph()
  ret = {}
  for x in ctx.summary_tensors:
    if x.value.graph is not g:
      raise ValueError('Tensor %r %r is not an element of this graph.' %
                       (x.name, x.value))
    ret['%s/%s' % (x.name, x.name_scope)] = x.value
  return ret


def merge_all_pw_tensor():
  """Returns all summary tensors as a dict of {name: tensor}.

  Note this function only returns summary tensors of type PwTpuSummaryTensor.

  Outside of tpu_summary.context() returns {}
  """
  ctx = TpuSummaryContext.current()
  if ctx is None:
    return {}
  g = tf.get_default_graph()
  ret = {}
  for x in ctx.summary_tensors:
    if isinstance(x, PwTpuSummaryTensor):
      # Only keep summaries of the desired type.
      if x.value.graph is not g:
        raise ValueError('Tensor %r %r is not an element of this graph.' %
                         (x.name, x.value))
      name = ('%s/%s' % (x.name_scope, x.name)).replace('/', '__')
      ret[name] = x.value
  return ret


# pylint: disable=g-doc-return-or-yield
@contextlib.contextmanager
def context(rewrite_while_loop=False,
            max_loop_vars=16,
            loop_vars_dtype=tf.float32):
  """TPU summary context.

  Args:
    rewrite_while_loop: rewrite tf.while_loop to propagate summaries
    max_loop_vars: number of loop vars added by rewrite_while_loop
    loop_vars_dtype: dtype of loop vars added by rewrite_while_loop
  """
  if not rewrite_while_loop:
    with TpuSummaryContext():
      yield
  else:
    with TpuSummaryContext(), RewriteLoopContext(max_loop_vars,
                                                 loop_vars_dtype):
      yield
