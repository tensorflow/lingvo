#!/usr/bin/env python
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
"""Runs code on a fleet of machines.

This runs the lingvo code on a fleet of docker for demonstration and testing
purposes. We assume the following:
* There is a running container
* There is a shared volume in /sharedfs. In reality, this would be something
like an NFS or HDFS mount.

The script is run on the host and only requires python and the docker binary
to be installed.

We run two "clusters": one for training, and one for decoding. The trainer
jobs (controller, trainer_client/worker or trainer/ps) are connected to
each other, whereas the decoder jobs are independent, only reading from
the shared filesystem. The trainer jobs are configured via a cluster spec
flag, whereas the decoder jobs are configured with individual flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pipes import quote as shell_quote
import shutil
import six
import subprocess
import sys


_SYNC_TRAIN_CLUSTER_SPEC = {
    "worker": [
        "worker0:43222",
        "worker1:43222",
        "worker2:43222",
    ],
    "controller": ["controller:43214",],
    "trainer_client": ["trainer_client:24601"],
}

_ASYNC_TRAIN_CLUSTER_SPEC = {
    "trainer": [
        "trainer0:43222",
        "trainer1:43222",
        "trainer2:43222",
    ],
    "ps": [
        "ps0:43221",
        "ps1:43221",
    ],
    "controller": ["controller:43214",],
}

DECODE_CLUSTER_SPEC = {
    "evaler_test": ["evaler_test:23487"],
    "decoder_test": ["decoder_test:24679"],
}

MODEL = "image.mnist.LeNet5"
DATADIR = "/tmp/mnist"
TRAIN_MODE = "sync"

TRAIN_CLUSTER_SPEC = (
    _SYNC_TRAIN_CLUSTER_SPEC
    if TRAIN_MODE == "sync" else _ASYNC_TRAIN_CLUSTER_SPEC)

DOCKER_BIN = "/usr/bin/docker"
# All that is required is that we have pip installed tensorflow.
DOCKER_IMAGE_NAME = "tensorflow:lingvo"
# This was created using
# bazel build -c opt //lingvo:trainer.par
# cp bazel-bin/lingvo/trainer.par .
# Since /tmp/lingvo is mounted, we can see it.
# TODO(drpng): hard-wiring below.
TRAINER_PACKAGE = "/tmp/lingvo/trainer.par"
DRY_RUN = False
NETWORK_NAME = "tf-net"

SHARED_FS_MOUNTPOINT = "/tmp/sharedfs"


def _RunDocker(args):
  print("Running: docker %s" % args)
  if DRY_RUN:
    return 0
  ret = subprocess.call([DOCKER_BIN] + args)
  return ret


def _RunDockerOrDie(args):
  ret = _RunDocker(args)
  if ret != 0:
    sys.stderr.write("Failed to run: %s\n" % ret)
    sys.stderr.flush()
    sys.exit(ret)


def _ExecInDocker(container_name,
                  cmd_array,
                  workdir=None,
                  logfile=None,
                  detach=False):
  """Execute in docker container."""
  if not workdir:
    workdir = "/tmp"
  opts = ["-t", "-w", workdir]
  if detach:
    opts += ["-d"]
  # TODO(drpng): avoid quoting hell.
  base_cmd = ["exec"] + opts + [container_name]
  if logfile:
    # The logfile is in the container.
    cmd = " ".join(shell_quote(x) for x in cmd_array)
    cmd += " >& %s" % logfile
    full_cmd = base_cmd + ["bash", "-c", cmd]
  else:
    full_cmd = base_cmd + cmd_array
  ret = _RunDocker(full_cmd)
  if ret != 0:
    sys.stderr.write(
        "Failed to exec within %s: %s" % (container_name, cmd_array))
    sys.exit(ret)


def _Machine(machine_port):
  # From host:port to host.
  return machine_port[:machine_port.index(":")]


def Cleanup():
  specs = TRAIN_CLUSTER_SPEC.values() + DECODE_CLUSTER_SPEC.values()
  for job_machines in specs:
    machines = [_Machine(x) for x in job_machines]
    _RunDocker(["stop", "-t", "0"] + machines)
  _RunDocker(["network", "rm", NETWORK_NAME])
  shutil.rmtree(SHARED_FS_MOUNTPOINT, ignore_errors=True)


def InitFiles():
  os.mkdir(SHARED_FS_MOUNTPOINT, 01777)
  # Create these directories so that we own them, not root.
  os.mkdir(SHARED_FS_MOUNTPOINT + "/log", 01777)
  os.mkdir(SHARED_FS_MOUNTPOINT + "/log/train", 01777)
  os.mkdir(SHARED_FS_MOUNTPOINT + "/log/decoder_test", 01777)
  os.mkdir(SHARED_FS_MOUNTPOINT + "/log/eval_test", 01777)


def InitNetwork():
  _RunDockerOrDie(["network", "create", "--driver", "bridge", NETWORK_NAME])


def StartFleet():
  specs = TRAIN_CLUSTER_SPEC.values() + DECODE_CLUSTER_SPEC.values()
  for job_machines in specs:
    for machine_port in job_machines:
      machine_name = _Machine(machine_port)
      _RunDockerOrDie([
          "run", "--rm", "--name", machine_name, "-dit", "--network",
          NETWORK_NAME, "-v", ":".join([SHARED_FS_MOUNTPOINT] * 2), "-v",
          ":".join([DATADIR] * 2 + ["ro"]), DOCKER_IMAGE_NAME, "bash"
      ])


def MakeFlagClusterSpec(cluster_spec):
  job_specs = []
  for job_name in sorted(cluster_spec.keys()):
    job_specs += [job_name + "=" + ",".join(cluster_spec[job_name])]
  flag_spec = "@".join(job_specs)
  return flag_spec


def CopyTrainerToSharedMount():
  shutil.copy(TRAINER_PACKAGE, SHARED_FS_MOUNTPOINT + "/trainer.par")


def InstallAndStartProcess(cluster_spec):
  """Unpacks the trainer and kick off training."""
  cluster_spec_flag = MakeFlagClusterSpec(cluster_spec)
  for job_name, machines in six.iteritems(cluster_spec):
    task_idx = 0
    for machine_port in machines:
      machine_name = _Machine(machine_port)
      _ExecInDocker(
          machine_name, [
              os.path.join(SHARED_FS_MOUNTPOINT, "trainer.par"),
              "--cluster_spec=%s" % cluster_spec_flag,
              "--job=%s" % job_name,
              "--task=%d" % task_idx,
              "--mode=%s" % TRAIN_MODE,
              "--logtostderr",
              "--model=%s" % MODEL,
              "--logdir=%s/log" % SHARED_FS_MOUNTPOINT,
          ],
          workdir="/tmp",
          logfile="%s/%s.%d.log" % (SHARED_FS_MOUNTPOINT, job_name, task_idx),
          detach=True)
      task_idx += 1


def main():
  Cleanup()
  InitFiles()
  InitNetwork()
  StartFleet()
  CopyTrainerToSharedMount()
  InstallAndStartProcess(TRAIN_CLUSTER_SPEC)
  for role in sorted(DECODE_CLUSTER_SPEC.keys()):
    # Each decode process is its own spec.
    machine_spec = DECODE_CLUSTER_SPEC[role]
    InstallAndStartProcess({role: machine_spec})


if __name__ == "__main__":
  main()
