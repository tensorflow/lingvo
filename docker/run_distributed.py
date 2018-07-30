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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import six
import subprocess
import sys


SCRIPT = "lingvo/trainer.py"

CLUSTER_SPEC = {
    "worker": [
        "worker0:43222",
        "worker1:43222",
        "worker2:43222",
    ],
    "ps": [
        "ps0:43221",
        "ps1:43221",
    ],
}

DOCKER_BIN = "/usr/bin/docker"
# All that is required is that we have pip installed tensorflow.
DOCKER_IMAGE_NAME = "tensorflow:lingvo"
# This was created using
# bazel build -c opt //lingvo:lingvo_trainer_pkg
# cp bazel-bin/lingvo/lingvo_trainer_pkg.tar .
# Since /tmp/lingvo is mounted, we can see it.
# TODO(drpng): hard-wiring below.
TRAINER_PACKAGE = "/tmp/lingvo/lingvo_trainer_pkg.tar"
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


def _ExecInDocker(container_name, cmd_array, workdir=None, detach=False):
  """Execute in docker container."""
  if not workdir:
    workdir = "/tmp"
  opts = ["-t", "-w", workdir]
  if detach:
    opts += ["-d"]
  full_cmd = ["exec"] + opts + [container_name] + cmd_array
  ret = _RunDocker(full_cmd)
  if ret != 0:
    sys.stderr.write(
        "Failed to exec within %s: %s" % (container_name, cmd_array))
    sys.exit(ret)


def _Machine(machine_port):
  # From host:port to host.
  return machine_port[:machine_port.index(":")]


def Cleanup():
  for job_machines in CLUSTER_SPEC.values():
    machines = [_Machine(x) for x in job_machines]
    _RunDocker(["stop", "-t", "0"] + machines)
  _RunDocker(["network", "rm", NETWORK_NAME])
  shutil.rmtree(SHARED_FS_MOUNTPOINT, ignore_errors=True)


def InitFiles():
  os.mkdir(SHARED_FS_MOUNTPOINT, 01777)


def InitNetwork():
  _RunDockerOrDie(["network", "create", "--driver", "bridge", NETWORK_NAME])


def StartFleet():
  for job_machines in CLUSTER_SPEC.values():
    for machine_port in job_machines:
      machine_name = _Machine(machine_port)
      _RunDockerOrDie(["run", "--rm", "--name", machine_name, "-dit",
                       "--network", NETWORK_NAME,
                       "-v", ":".join([SHARED_FS_MOUNTPOINT] * 2),
                       DOCKER_IMAGE_NAME, "bash"])


def InstallAndStartTrainer():
  """Unpacks the trainer and kick off training."""
  shutil.copy(TRAINER_PACKAGE, SHARED_FS_MOUNTPOINT + "/lingvo_trainer_pkg.tar")
  ps_hosts = ",".join(CLUSTER_SPEC["ps"])
  worker_hosts = ",".join(CLUSTER_SPEC["worker"])
  for job_name, machines in six.iteritems(CLUSTER_SPEC):
    task_idx = 0
    for machine_port in machines:
      machine_name = _Machine(machine_port)
      _ExecInDocker(machine_name,
                    ["tar", "-xf",
                     SHARED_FS_MOUNTPOINT + "/lingvo_trainer_pkg.tar"],
                    workdir="/tmp")
      _ExecInDocker(machine_name,
                    ["python", SCRIPT,
                     "--shared_fs", SHARED_FS_MOUNTPOINT,
                     "--ps_hosts", ps_hosts,
                     "--worker_hosts", worker_hosts,
                     "--job_name", job_name,
                     "--task_index", str(task_idx)],
                    workdir="/tmp",
                    detach=True)
      task_idx += 1


def main():
  Cleanup()
  InitFiles()
  InitNetwork()
  StartFleet()
  InstallAndStartTrainer()


if __name__ == "__main__":
  main()
