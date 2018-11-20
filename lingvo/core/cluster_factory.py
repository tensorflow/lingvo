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
"""Cluster factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo.core import cluster

Cluster = cluster._Cluster


def Current():
  """Returns the current cluster specification.

  E.g.::

    with Cluster(cluster_params) as foo:
      model = p.cls(p)
      model.FProp()  # FProp can access foo through cluster_factory.Current().
  """
  stack = Cluster._cluster_stack().stack
  if not stack:
    return Cluster(Cluster.Params())
  else:
    return stack[-1]


def ForTestingWorker(mode=None,
                     job=None,
                     gpus=None,
                     split_size=None,
                     tpus=None,
                     add_summary=None):
  """Returns a Cluster for unittesting with a worker."""
  p = Cluster.Params()
  if mode:
    p.mode = mode
  if job:
    p.job = job
  if gpus:
    p.worker.gpus_per_replica = gpus
  if tpus:
    p.worker.tpus_per_replica = tpus
  if split_size:
    p.worker.devices_per_split = split_size
  if add_summary:
    p.add_summary = add_summary
  return p.cls(p)
