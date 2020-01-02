# Lint as: python2, python3
# -*- coding: utf-8 -*-
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
r"""Launch script for GKE jobs.

This script generates the GKE deployment configs for TPU training, GPU/CPU
decoding, and TensorBoard jobs.

It assumes you have:

a) Copied any input data to GCS.
b) Docker and GKE/GCP tools installed locally.
c) Created the TPU and GPU clusters using `gcloud containers create`.

This script launches jobs by:

1) Building a lingvo docker image built from --base_image, copying
   the directory pointed to by --build, writing that image to --image
   (with an automatically generated date-based tag for versioning).

2) Identifying the full name of the GKE cluster that each accelerator
   job runs in based on --trainer_cell and --decoder_cell

3) Writing out .yaml configuration files based on --name, --model, --logdir,
   that will launch the docker images for each job type.

Usage looks something like:

  python3 lingvo/tools/gke_launch.py \
    --model=$MODEL \
    --base_image=tensorflow:lingvo_lib_gpu \
    --image=$DOCKER_IMAGE \
    --logdir=$LOGDIR \
    --tpu_type=$TPU_TYPE \
    --trainer_cell=$TPU_CLUSTER_NAME \
    --decoder_cell=$GPU_CLUSTER_NAME \
    --decoder_gpus=1 \
    --gpu_type=$GPU_TYPE \
    --decoder=dev \
    --extra_envs=KITTI_DIR=$GCS_PATH \
    --name=$EXP_NAME \
    --build=$YOUR_CODE_DIR \
    $ACTION $TARGETS

ACTION specifies whether to start (up), stop (down) or reload the target jobs.
One can also specify "print" to just print out the .yaml configuration files.

TARGETS specifies whether the action affects all jobs ("all") or just an
individual job ("trainer", "decoder", "tensorboard").

See the flags definition below for details on the arguments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import subprocess
import tempfile

from absl import app
from absl import flags
import six
import yaml

FLAGS = flags.FLAGS

# General flags for all jobs.
flags.DEFINE_string("name", None, "Base name of experiment.")
flags.DEFINE_string("model", None, "Name of registered model.")
flags.DEFINE_string(
    "image", None, "Name of docker image to use.  If tag is not specified "
    "and --build is set, a time-based tag will be used.")
flags.DEFINE_string("logdir", None, "GCS location of base logdir.")
flags.DEFINE_string("gpu_type", "",
                    "Type of GPU to use, e.g., 'p100' or 'v100'.")

# Cluster flags
flags.DEFINE_string(
    "trainer_cell", "", "Query name for the GKE cluster to use for training. "
    "Must uniquely identify among all active clusters.")
flags.DEFINE_string(
    "decoder_cell", "", "Query name for the GKE cluster to use for decoding. "
    "Must uniquely identify among all active clusters.")

# TPU training flags.
flags.DEFINE_string("tpu_type", "v3-8",
                    "Type of TPU to use for training, e.g., 'v3-8' for a 2x2.")

# Decoder flags.
flags.DEFINE_string("decoder_ram", "24G",
                    "Amount of CPU memory for the decoder.")
# TODO(vrv): Support launching multiple decoder jobs when this is
# comma-separated.
flags.DEFINE_string("decoder", "dev", "Split(s) of dataset for decoding.")
flags.DEFINE_integer("decoder_gpus", 0,
                     "Number of GPUs per machine for decoding.")

# Docker-based flags.
flags.DEFINE_string(
    "build", None, "If set, builds the docker image "
    "for --image including all code from this directory.")
flags.DEFINE_string(
    "base_image", "tensorflow:lingvo_lib",
    "Base Lingvo docker image to use. If using GPU, make "
    "sure you have built a GPU-enabled image.")
flags.DEFINE_string(
    "extra_envs", "", "Extra comma-separated list of key=value environment "
    "variables to set when building the docker image.")


def _get_or_add(cfg, name):
  """Gets cfg[name], or adds 'name' with an empty dict if not present."""
  if name not in cfg:
    cfg.update({name: {}})
  return cfg[name]


def add_gpu_to_pod(cfg, gpu_type, num_gpus):
  """Sets the appropriate GPU fields to cfg.

  Args:
    cfg: The YAML-based dictionary to update.
    gpu_type: The type of GPU to launch on GKE.
    num_gpus: The number of GPUs to launch in the task.
  """
  if num_gpus == 0:
    return

  if gpu_type == "p100":
    gpu_str = "nvidia-tesla-p100"
  elif gpu_type == "v100":
    gpu_str = "nvidia-tesla-v100"
  else:
    raise ValueError("Invalid gpu type: ", gpu_type)
  cfg["spec"].update(
      {"nodeSelector": {
          "cloud.google.com/gke-accelerator": gpu_str
      }})

  containers = cfg["spec"]["containers"][0]
  resources = _get_or_add(containers, "resources")
  resources.update({"limits": {"nvidia.com/gpu": num_gpus}})


def set_pod_cpu_memory(cfg, cpu_memory):
  """Sets the amount of CPU memory to request in the container."""
  containers = cfg["spec"]["containers"][0]
  resources = _get_or_add(containers, "resources")
  resources.update({"requests": {"memory": cpu_memory}})


def decoder_template(job_name, model, image, logdir, decoder_type,
                     decoder_gpus):
  """Constructs the base yaml config for the decoder."""
  name = six.ensure_str(job_name) + ".decoder"
  container_name = name.replace(".", "-")
  job = "decoder_" + six.ensure_str(decoder_type)
  return """
apiVersion: v1
kind: Pod
metadata:
  name: {job_name}
spec:
  restartPolicy: Never
  containers:
    - name: {container_name}
      image: {image}
      command: ["/usr/bin/python3"]
      args: ["-m", "lingvo.trainer", "--mode=sync", "--alsologtostderr", "--model={model}", "--logdir={logdir}", "--job={job}", "--decoder_gpus={decoder_gpus}", "--decoder_replicas=1", "--cluster_spec={job}=localhost:0"]
  """.format(
      job_name=name,
      container_name=container_name,
      image=image,
      model=model,
      logdir=logdir,
      job=job,
      decoder_gpus=decoder_gpus)


def _tpu_resource(tpu_type):
  version, num_chips = six.ensure_str(tpu_type).split("-")
  return "cloud-tpus.google.com/{}: {}".format(version, num_chips)


def tpu_training_template(job_name, model, image, logdir, tpu_type):
  """Constructs the base yaml config for the TPU trainer."""
  name = six.ensure_str(job_name) + ".trainer"
  container_name = name.replace(".", "-")
  tpu_string = _tpu_resource(tpu_type)
  return """apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
spec:
  template:
    metadata:
      annotations:
        tf-version.cloud-tpus.google.com: "nightly"
    spec:
      restartPolicy: Never
      containers:
      - name: {container_name}
        image: {image}
        command: ["/usr/bin/python3"]
        args: ["-m", "lingvo.trainer", "--mode=sync", "--alsologtostderr", "--model={model}", "--logdir={logdir}", "--tpu=$(KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS)"]
        resources:
          limits:
            {tpu_string}
  """.format(
      name=name,
      container_name=container_name,
      image=image,
      model=model,
      logdir=logdir,
      tpu_string=tpu_string)


def tensorboard_template(job_name, logdir, port):
  """Constructs the tensorboard YAML template."""
  job_name = six.ensure_str(job_name) + ".tensorboard"
  container_name = job_name.replace(".", "-")

  print("To poll for tensorboard address, run: $ kubectl get service %s -w" %
        (container_name + "-service"))
  return """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {job_name}
spec:
  replicas: 1
  selector:
    matchLabels:
      name: {job_name}
  template:
    metadata:
      labels:
        name: {job_name}
    spec:
      restartPolicy: Always
      containers:
      - name: {container_name}
        image: gcr.io/tensorflow/tpu-util:r1.11
        command:
        - tensorboard
        - --logdir=$(MODEL_BUCKET)
        env:
        - name: MODEL_BUCKET
          value: {logdir}
        ports:
        - containerPort: {port}
---
apiVersion: v1
kind: Service
metadata:
  name: {container_name}-service
spec:
  type: LoadBalancer
  selector:
    name: {job_name}
  ports:
  - port: {port}
    targetPort: {port}
""".format(
    job_name=job_name, container_name=container_name, logdir=logdir, port=port)


def build_docker_image(image, base_image, code_directory, extra_envs):
  """Build a docker image and push it to the location specified by image.

  Args:
    image: String name of tag to use, e.g., 'gcr.io/foo/bar:version'
    base_image: String name of base lingvo image to build from.
    code_directory: Location of directory whose contents will be copied into the
      image.
    extra_envs: A comma-separated list of key=value environment variables to be
      built into the docker.
  """
  preamble = [
      "FROM %s AS lingvo" % base_image,
  ]
  envs = [
      "ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/kubernetes/bin/nvidia/lib64",
      "ENV PATH=${PATH}:/home/kubernetes/bin/nvidia/bin",
  ]
  for env_pairs in six.ensure_str(extra_envs).split(","):
    envs += ["ENV %s" % env_pairs]

  copy_code = ["WORKDIR /tmp/lingvo", "COPY . ."]

  gpu_docker_file = preamble + envs + copy_code
  tmp_dockerfile = tempfile.mkstemp(suffix=".dockerfile")[1]
  with open(tmp_dockerfile, "w") as f:
    f.write("\n".join(gpu_docker_file))
    print("Writing Dockerfile to", tmp_dockerfile)

  os.system("docker build --tag %s --no-cache -f- %s < %s " %
            (image, code_directory, tmp_dockerfile))
  os.system("docker push %s" % image)


def get_gke_cluster(gke_cluster_spec):
  """Get the full name of the GKE cluster given shorthand `gke_cluster_spec`.

  For example, gcloud container cluster list produces:

  NAME                      LOCATION        ...
  p100-europe-west4-a-nh16  europe-west4-a  ...
  test-df-europe            europe-west4-a  ...

  Then a gke_cluster_spec of 'p100' or 'df' will produce the fully-qualified
  cluster.

  A gke_cluster_spec of 'europe' will raise a ValueError because there are two
  active clusters that have the string 'europe' in them.

  Args:
    gke_cluster_spec: A string specifying a filter on active clusters.

  Returns:
    The fully qualified GKE cluster name, or None if not found.

  Raises:
    ValueError: If gke_cluster_spec does not uniquely identify an
      active cluster.
  """
  if not gke_cluster_spec:
    return ""

  active_clusters = subprocess.check_output(
      ["gcloud", "container", "clusters", "list"])
  cluster_names = [
      c.split(b" ")[0] for c in active_clusters.split(b"\n")[1:] if c
  ]
  filtered = [
      c for c in cluster_names if six.ensure_binary(gke_cluster_spec) in c
  ]

  if len(filtered) > 1:
    raise ValueError("Cluster filter was not precise enough.")

  if not filtered:
    return None

  result = subprocess.check_output(["kubectl", "config", "view"])
  result = result.split(b"\n")
  clusters = [
      r.lstrip() for r in result if "    cluster: " in r.decode("utf-8")
  ]
  cluster = [r for r in clusters if filtered[0] in r]
  assert len(cluster) == 1
  cluster = cluster[0].split(b" ")[1]
  return cluster


# Not yet used, but illustrates one what needs to do.
def create_tpu_cluster(cluster_name, zone):
  _ = subprocess.check_output([
      "gcloud", "container", "clusters", "create", cluster_name,
      "--cluster_version=1.13", "--scopes=cloud-platform,gke-default",
      "--enable-ip-alias", "--enable-tpu", "--async",
      "--zone=%s" % zone
  ])


def validate_args(argv):
  """Validates the input arguments. Raises a UsageError if invalid."""
  valid_commands = ["up", "down", "print", "reload"]
  error = ""
  if len(argv) < 2:
    error = ("Command not provided. "
             "Command must be one of %s" % valid_commands)
  elif argv[1] not in valid_commands:
    error = "Command must be one of %s" % valid_commands

  if len(argv) > 3:
    print("Too many arguments.")

  if len(argv) == 3:
    valid_targets = ["trainer", "decoder", "tensorboard", "all"]
    if argv[2] not in valid_targets:
      error = "Target must be one of %s" % valid_targets

  if error:
    raise app.UsageError(error)


def main(argv):
  validate_args(argv)
  action = argv[1]
  target = argv[2] if len(argv) == 3 else "all"

  if target == "all":
    targets = ["trainer", "decoder", "tensorboard"]
  else:
    targets = [target]

  image = FLAGS.image

  if action == "reload":
    actions = ["down", "up"]
  else:
    actions = [action]

  # Maybe build.
  if "up" in actions and FLAGS.build:
    # If image does not specify a tag, create a temporary one.
    if ":" not in image:
      image += ":%s" % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    build_docker_image(image, FLAGS.base_image, FLAGS.build, FLAGS.extra_envs)

  # Write out the YAML deployment files to a temporary directory
  # for post-hoc inspection.
  root = tempfile.mkdtemp(prefix=FLAGS.name)
  print("Writing out yaml configs to %s" % root)

  action_to_cmd = {"up": "create", "down": "delete"}

  # Start or stop the jobs as requested.
  if "tensorboard" in targets:
    # Create tensorboard config.  Template consists of two yaml configs, so skip
    # parsing.
    tb_tmpl = tensorboard_template(FLAGS.name, FLAGS.logdir, 6006)
    tb_path = os.path.join(root, "tensorboard.yaml")
    with open(tb_path, "w") as f:
      f.write(tb_tmpl)
    if "print" in actions:
      print("TB yaml: %s" % tb_tmpl)
    else:
      for action in actions:
        cmd = "kubectl %s -f %s" % (action_to_cmd[action], tb_path)
        print("Running: %s" % cmd)
        os.system(cmd)

  if "trainer" in targets:
    tpu_cluster = get_gke_cluster(FLAGS.trainer_cell)
    assert tpu_cluster
    tpu_tmpl = tpu_training_template(FLAGS.name, FLAGS.model, image,
                                     FLAGS.logdir, FLAGS.tpu_type)
    # Check that it can be loaded.
    tpu_cfg = yaml.safe_load(tpu_tmpl)

    trainer_path = os.path.join(root, "tpu_train.yaml")
    loaded = yaml.safe_dump(tpu_cfg)
    with open(trainer_path, "w") as f:
      f.write(six.ensure_str(loaded))

    if "print" in actions:
      print("\n\nTPU yaml: \n%s" % loaded)
    else:
      for action in actions:
        cmd = "kubectl %s -f %s --cluster %s" % (action_to_cmd[action],
                                                 trainer_path, tpu_cluster)
        print("Running: %s" % cmd)
        os.system(cmd)

  if "decoder" in targets:
    decoder_cluster = get_gke_cluster(FLAGS.decoder_cell)
    decoder_tmpl = decoder_template(FLAGS.name, FLAGS.model, image,
                                    FLAGS.logdir, FLAGS.decoder,
                                    FLAGS.decoder_gpus)

    # Parse into YAML
    decoder_cfg = yaml.safe_load(decoder_tmpl)
    # Set decoder config values.
    add_gpu_to_pod(decoder_cfg, FLAGS.gpu_type, FLAGS.decoder_gpus)
    set_pod_cpu_memory(decoder_cfg, FLAGS.decoder_ram)

    assert decoder_cluster
    decoder_path = os.path.join(root, "decoder.yaml")
    with open(decoder_path, "w") as f:
      f.write(six.ensure_str(yaml.safe_dump(decoder_cfg)))

    if "print" in actions:
      print("Decoder yaml: \n%s" % yaml.safe_dump(decoder_cfg))
    else:
      for action in actions:
        cmd = "kubectl %s -f %s --cluster %s" % (action_to_cmd[action],
                                                 decoder_path, decoder_cluster)
        print("Running: %s" % cmd)
        os.system(cmd)


if __name__ == "__main__":
  flags.mark_flags_as_required(["name", "model", "image", "logdir"])
  app.run(main)
