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
r"""Trainer.

To run locally:

.. code-block:: bash

  $ bazel build -c opt //lingvo:trainer
  $ bazel-bin/lingvo/trainer --logtostderr \
      --model=image.mnist.LeNet5 --mode=sync --logdir=/tmp/lenet5 \
      --run_locally=cpu

To use GPU, add `--config=cuda` to build command and set `--run_locally=gpu`.
"""
import os
import re
import sys
import threading

from lingvo import base_trial
from lingvo import datasets
from lingvo import eager_runners
from lingvo import executor
from lingvo import model_imports
from lingvo import model_registry
from lingvo import runners
from lingvo import trainer_utils  # pylint: disable=unused-import
import lingvo.compat as tf
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import cluster_factory
from lingvo.core import inference_graph_exporter
from lingvo.core import py_utils
from lingvo.core import summary_utils

from google.protobuf import text_format

tf.flags.DEFINE_bool(
    'interactive', False,
    'If True, enter interactive IPython for the controller job.')

tf.flags.DEFINE_string(
    'run_locally', '',
    'Can be empty, cpu, or gpu. If not empty, ignores cluster configuration '
    'flags and runs controller and trainer in a single local process.')

tf.flags.DEFINE_string(
    'cluster_spec', '', 'A tf.train.ClusterSpec to override the master. '
    'The dict is specified as: job=host1:port1,host2:port2,'
    'host3:port3@job2=host3:port4,...')

tf.flags.DEFINE_string(
    'mode',
    'async', 'How this trainer binary is used. '
    'async: used in an async training setup; '
    'sync: used in a sync training setup; '
    'shell: an interactive shell for development; '
    'inspect_evaler: print evaler dataset names; '
    'inspect_decoder: print decoder dataset names; '
    'inspect_model: print the names and shapes of variables for this model; '
    'inspect_params: print the model params corresponding to each dataset; '
    'write_inference_graph: write inference graphs to logdir.',
    allow_hide_cpp=True)

tf.flags.DEFINE_multi_string(
    'inspect_model_part_regex', None,
    'This argument is used to check the number of params in different part '
    'of the model. (e.g. encoder or decoder or any specific layers of '
    'encoder/decoder.) The value should be in the name:regex format. '
    'For example, --inspect_model_part_regex=encoder:^.+conformer_encoder.+ '
    'means any tensor\'s name matched with regex `^.+conformer_encoder.+` will '
    'be counted as `encoder`, and the number of params in `encoder` will be '
    'printed out when `inspect_model`. ')

tf.flags.DEFINE_integer('inspect_model_topn', 0,
                        'print `topn` tensors when inspec_model')

tf.flags.DEFINE_string('controller_job', '/job:controller', 'Job name.')
tf.flags.DEFINE_integer('controller_gpus', 0, 'Number of controller GPUs.')

tf.flags.DEFINE_integer('worker_replicas', 1, 'Number of replicas.')
tf.flags.DEFINE_integer('worker_gpus', 0, 'Number of gpus to use per replica.')
tf.flags.DEFINE_integer('worker_split_size', 1,
                        'Number of devices for one split.')

tf.flags.DEFINE_string('ps_job', '/job:ps', 'Job name')
tf.flags.DEFINE_integer('ps_replicas', 1, 'Number of replicas.')
tf.flags.DEFINE_integer('ps_gpus', 0, 'Number of gpus to use per replica.')

tf.flags.DEFINE_string('input_job', '/job:input', 'Job name')
tf.flags.DEFINE_integer('input_replicas', 0, 'Number of replicas.')
tf.flags.DEFINE_string(
    'input_targets', '', 'Target network addresses for the '
    'input job. E.g., a single ip:port, or a list of '
    'comma-separated grpc://ip:port, etc.')

tf.flags.DEFINE_string('tf_data_service_address', '',
                       'The address of the tf.data service.')

tf.flags.DEFINE_string(
    'inference_graph_filename', None,
    'Output inference graph filename. If unspecified, output two inference '
    'graphs, one for CPU and one for TPU using the default settings.')
tf.flags.DEFINE_string(
    'inference_graph_device', None,
    'Type of device the output inference graph is for. This flag is applicable '
    'only when FLAGS.inference_graph_filename is specified.')
tf.flags.DEFINE_integer(
    'inference_graph_random_seed', None,
    'Random seed to fix when exporting inference graph. '
    'Not fixed when set to None.')
tf.flags.DEFINE_list(
    'graph_def_filename', [],
    'Output inference graph_def filenames. Defaults to CPU graph if '
    'inference_graph_filename and inference_graph_device are not specified.')
tf.flags.DEFINE_list(
    'inspect_params_dataset_name', None,
    'Which dataset params to inspect. If None, all the dataset params will be '
    'output')
tf.flags.DEFINE_string(
    'inference_dataset_name', 'Test',
    'Name of the dataset whose params to be extracted inference graph with.')
tf.flags.DEFINE_bool(
    'inference_gen_tpu_init_op', True,
    'Whether the tpu_init_op subgraph is generated for TPU inference graph.')

tf.flags.DEFINE_bool(
    'evaler_in_same_address_as_controller', False,
    'Whether or not evaler is in the same address space as '
    'controller. This flag is meant for unittest only.')

tf.flags.DEFINE_string(
    'vizier_reporting_job', 'evaler',
    'Job responsible for reporting metrics. This specifies a '
    'job prefix, evaler will match all evaler jobs, while '
    'evaler_dev and decoder_dev will only match the corresponding '
    'jobs that are on the dev set.')

tf.flags.DEFINE_bool(
    'add_summary', None,
    'Whether we should output summaries. The default value "None", enables '
    'summaries based on the job type.')
tf.flags.DEFINE_bool('disable_tf2', False,
                     'Whether run on Tensorflow without V2 behaviors.')
tf.flags.DEFINE_bool('use_eager', False, 'Whether to use eager mode.')


@tf.flags.validator('vizier_reporting_job')
def _ValidateVizierReportingJob(value):
  if value in ['evaler', 'decoder']:
    return True
  if value.startswith('evaler_') or value.startswith('decoder_'):
    return True
  tf.logging.info('vizier_reporting_job should usually start with evaler or '
                  'decoder, unless in executor/program mode. '
                  f'vizier_reporting_job={value}')
  return True


tf.flags.DEFINE_bool(
    'checkpoint_in_trainer_tpu', False,
    'Whether to enable checkpointing in TrainerTpu, allowing for '
    'operation without a separate Controller task.'
    'This flag also disables checkpointing from the Controller, '
    'but still allows it to write summaries.')


tf.flags.DEFINE_string(
    'tpu', None,
    'The Cloud TPU on GCP to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url. If set, other cluster parameters (such as --cluster_spec) will be '
    'configured automatically with TPUClusterResolver.')
tf.flags.DEFINE_boolean(
    'infer_tpu_cluster', False,
    'If set, infer tpu cluster parameters by querying the tensorflow cluster '
    'directly instead of reading it from commandline arguments. This is '
    'helpful, for example, when platforms cannot be pre-determined before '
    'scheduling.')
tf.flags.DEFINE_string(
    'gcp_project', None,
    'Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
tf.flags.DEFINE_string(
    'tpu_zone', None,
    'GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Please consider adding model params instead of adding flags.

FLAGS = tf.flags.FLAGS


def _GetClusterSpecDict():
  """Parses the cluster_spec flag and returns a dict."""
  job_specs = FLAGS.cluster_spec.split('@')
  cluster_spec_dict = {}
  for job_spec in job_specs:
    # ps_host=worker1:1231,worker2:1234
    job_machines = job_spec.split('=')
    if len(job_machines) != 2:
      raise ValueError(f'Invalid job specification: {job_spec}')
    cluster_spec_dict[job_machines[0]] = job_machines[1].split(',')

  return cluster_spec_dict


class RunnerManager:
  """Helper class for managing runners."""

  # This is a hack so these classes can be overridded with internal
  # non-public implementations.
  # pylint: disable=invalid-name
  inference_graph_exporter = inference_graph_exporter
  model_registry = model_registry
  Controller = runners.Controller
  TrainerTpu = runners.TrainerTpu
  ExecutorTpu = executor.ExecutorTpu
  TrainSummaries = eager_runners.TrainSummaries

  @property
  def Trainer(self):
    return eager_runners.Trainer if py_utils.IsEagerMode() else runners.Trainer

  @property
  def Evaler(self):
    return eager_runners.Evaler if py_utils.IsEagerMode() else runners.Evaler

  @property
  def Decoder(self):
    return eager_runners.Decoder if py_utils.IsEagerMode() else runners.Decoder

  # pylint: enable=invalid-name

  def __init__(self, model):
    self._model_name = model

  def MaybeLaunchTensorFlow(self):
    """Starts TF machinery in this process."""
    if FLAGS.run_locally or FLAGS.tpu or FLAGS.use_eager:
      return

    tf.logging.info('Launching tensorflow.')

    target = FLAGS.tf_master
    if not target.startswith('localhost'):
      # E.g., trainer_client is configured w/ FLAGS.tf_master pointing to
      # another job. In that case, start a local server.
      cluster_spec_dict = _GetClusterSpecDict()
      self._tf_server = tf.distribute.Server(
          tf.train.ClusterSpec(cluster_spec_dict),
          job_name=FLAGS.job,
          task_index=FLAGS.task)
      target = self._tf_server.target
    if not FLAGS.tf_master:
      FLAGS.tf_master = target
    with tf.Session(target).as_default():
      value = (tf.constant(1.) + tf.constant(1.)).eval()
    assert value == 2.0, 'Something is really wrong.'
    tf.logging.info('Launched tensorflow.')

  def GetExecutorParams(self):
    """Get the params needed to instantiate the ExecutorTpu.

    Returns:
       Tuple (dict, params):

         - ps_params_dict: high_level task_name -> ProgramScheduleParams
         - train_cfg: Either a SingleTaskModelParams or MultiTaskModelParams.
    """
    cluster = cluster_factory.Current()
    self.UpdateClusterParamsFromFlags(cluster.params, 'executor_tpu')
    ps_params_dict, train_cfg = executor.GetExecutorParams(
        self._model_name, cluster.params, self.model_registry)

    return ps_params_dict, train_cfg

  def GetParamsForDataset(self, job_name, dataset_name):
    """Returns params for job `job_name` on the dataset `dataset_name`."""
    # Get the current cluster and update its params from flags.
    cluster = cluster_factory.Current()
    self.UpdateClusterParamsFromFlags(cluster.params, job_name)
    with cluster_factory.Cluster(cluster.params):
      try:
        cfg = self.model_registry.GetParams(self._model_name, dataset_name)
      except base_model_params.DatasetError as e:
        dataset_name_retry = dataset_name.title()
        tf.logging.warning(
            'Exception configuring dataset %s, retrying as %s: %s',
            dataset_name, dataset_name_retry, e)
        cfg = self.model_registry.GetParams(self._model_name,
                                            dataset_name_retry)
        tf.logging.warning('Succeeded after retrying as %s.' %
                           dataset_name_retry)
    cfg.cluster = cluster.params

    # Updates a few params based on flags.
    if FLAGS.enqueue_max_steps is not None:
      cfg.train.enqueue_max_steps = FLAGS.enqueue_max_steps
    if FLAGS.saver_max_to_keep is not None:
      cfg.train.save_max_to_keep = FLAGS.saver_max_to_keep
    if FLAGS.saver_keep_checkpoint_every_n_hours is not None:
      cfg.train.save_keep_checkpoint_every_n_hours = FLAGS.saver_keep_checkpoint_every_n_hours
    return cfg

  def MaybeConfigRunDistributed(self):
    """If given a `FLAGS.cluster_spec`, update flags for running distributed."""
    if not FLAGS.cluster_spec:
      return
    job_specs = FLAGS.cluster_spec.split('@')
    cluster_spec_dict = _GetClusterSpecDict()
    if FLAGS.job == 'trainer_client':
      FLAGS.tf_master = 'grpc://%s' % cluster_spec_dict['worker'][FLAGS.task]
    for job in cluster_spec_dict:
      if job.startswith('decoder_'):
        assert len(job_specs) == 1, 'Decoder jobs must run on their own'
        assert ',' not in job_specs[0], 'Only single machine supported'
        FLAGS.decoder_job = '/job:%s' % job
        FLAGS.decoder_replicas = 1
      if job.startswith('evaler_'):
        assert len(job_specs) == 1, 'Evaler jobs must run on their own'
        assert ',' not in job_specs[0], 'Only single machine supported'
        FLAGS.evaler_job = '/job:%s' % job
        FLAGS.evaler_replicas = 1
    if FLAGS.mode == 'sync' and FLAGS.job in ('controller', 'trainer_client',
                                              'worker', 'executor_tpu'):
      FLAGS.worker_job = '/job:worker'
      FLAGS.worker_replicas = len(cluster_spec_dict['worker'])
      FLAGS.ps_job = '/job:worker'
      FLAGS.ps_replicas = FLAGS.worker_replicas
    if FLAGS.mode == 'async' and FLAGS.job in ('controller', 'trainer', 'ps'):
      FLAGS.worker_job = '/job:trainer'
      FLAGS.worker_replicas = len(cluster_spec_dict['trainer'])
      FLAGS.ps_job = '/job:ps'
      FLAGS.ps_replicas = len(cluster_spec_dict['ps'])

  def MaybeConfigCloudTpu(self):
    """If given `FLAGS.tpu`, update flags for running on a Cloud TPU."""
    if not FLAGS.tpu:
      return

    if not FLAGS.job:
      FLAGS.job = 'trainer_client'

    if FLAGS.job not in ('trainer_client', 'executor_tpu'):
      raise ValueError('Only trainer_client and executor_tpu jobs are '
                       'supported on TPU.')

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu,
        project=FLAGS.gcp_project,
        zone=FLAGS.tpu_zone,
        job_name=FLAGS.job)
    cluster_spec_dict = cluster_resolver.cluster_spec().as_dict()

    FLAGS.mode = 'sync'
    FLAGS.tf_master = cluster_resolver.master()

    FLAGS.worker_job = '/job:{}'.format(FLAGS.job)
    FLAGS.worker_replicas = 1
    FLAGS.worker_num_tpu_hosts = len(cluster_spec_dict[FLAGS.job])
    FLAGS.worker_tpus = (
        cluster_resolver.num_accelerators()['TPU'] * FLAGS.worker_num_tpu_hosts)
    FLAGS.ps_job = FLAGS.worker_job
    if FLAGS.job == 'trainer_client':
      FLAGS.ps_replicas = FLAGS.worker_replicas

    FLAGS.cluster_spec = ('@'.join('{}={}'.format(job, ','.join(hosts))
                                   for job, hosts in cluster_spec_dict.items()))

    FLAGS.xla_device = 'tpu'
    FLAGS.enable_asserts = False
    FLAGS.checkpoint_in_trainer_tpu = True

  def MaybeInferTPUClusterParams(self):
    """With flags enabled, update cluster params based on probing result.

    If FLAGS.infer_tpu_cluster is enabled, sets TPU cluster params
    based on query results of TPUClusterResolver.
    """
    if not FLAGS.infer_tpu_cluster:
      return

    if not FLAGS.job:
      FLAGS.job = 'trainer_client'

    if FLAGS.job not in ('trainer_client', 'executor_tpu'):
      raise ValueError('Only trainer_client and executor_tpu jobs are '
                       'supported on TPU.')

    FLAGS.mode = 'sync'
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tf_master)
    FLAGS.tf_master = cluster_resolver.master()
    # TPUClusterResolver does not populate cluster_spec for Borg clusters.
    # Use get_tpu_system_metadata instead.
    metadata = cluster_resolver.get_tpu_system_metadata()

    FLAGS.worker_num_tpu_hosts = metadata.num_hosts
    FLAGS.worker_tpus = (metadata.num_hosts * metadata.num_of_cores_per_host)
    # All tpu hosts should be in the same replica.
    FLAGS.worker_replicas = 1
    FLAGS.ps_job = FLAGS.worker_job
    FLAGS.ps_replicas = metadata.num_hosts
    if FLAGS.job == 'trainer_client':
      FLAGS.ps_replicas = FLAGS.worker_replicas

    FLAGS.xla_device = 'tpu'
    FLAGS.enable_asserts = False
    FLAGS.checkpoint_in_trainer_tpu = True

  def UpdateClusterParamsFromFlags(self, cluster, job_name):
    """Update `cluster` with a training cluster configuration from flags."""
    cluster.mode = FLAGS.mode
    cluster.job = job_name
    cluster.task = FLAGS.task
    cluster.do_eval = job_name in ['evaler', 'decoder']
    cluster.logdir = FLAGS.logdir

    cluster.controller.name = FLAGS.controller_job
    cluster.controller.gpus_per_replica = FLAGS.controller_gpus

    cluster.worker.name = FLAGS.worker_job
    cluster.worker.replicas = FLAGS.worker_replicas
    cluster.worker.gpus_per_replica = FLAGS.worker_gpus
    cluster.worker.tpus_per_replica = FLAGS.worker_tpus
    cluster.worker.num_tpu_hosts = FLAGS.worker_num_tpu_hosts
    cluster.worker.devices_per_split = FLAGS.worker_split_size
    if FLAGS.additional_worker_jobs:
      for additional_job in FLAGS.additional_worker_jobs:
        cluster.worker.additional_worker_names.append(additional_job)

    if FLAGS.tpu:
      job_name = cluster.worker.name.replace('/job:', '', 1)
      worker_hosts = _GetClusterSpecDict()[job_name]
      if FLAGS.additional_worker_jobs:
        for additional_job in cluster.worker.additional_worker_names:
          additional_job_name = additional_job.replace('/job:', '', 1)
          worker_hosts.extend(_GetClusterSpecDict()[additional_job_name])
      cluster.worker.targets = ','.join(
          'grpc://{}'.format(host) for host in worker_hosts)

    cluster.ps.name = FLAGS.ps_job
    cluster.ps.replicas = FLAGS.ps_replicas
    cluster.ps.gpus_per_replica = FLAGS.ps_gpus

    cluster.input.name = FLAGS.input_job
    cluster.input.replicas = FLAGS.input_replicas
    cluster.input.targets = FLAGS.input_targets

    if py_utils.IsEagerMode():
      cluster.evaler.name = '/job:localhost'
      cluster.decoder.name = '/job:localhost'
    else:
      cluster.evaler.name = FLAGS.evaler_job
      cluster.decoder.name = FLAGS.decoder_job

    cluster.evaler.replicas = FLAGS.evaler_replicas
    cluster.evaler.gpus_per_replica = FLAGS.evaler_gpus
    cluster.decoder.replicas = FLAGS.decoder_replicas
    cluster.decoder.gpus_per_replica = FLAGS.decoder_gpus

    cluster.tf_data_service_address = FLAGS.tf_data_service_address

    cluster.add_summary = FLAGS.add_summary
    cluster.reporting_job = FLAGS.vizier_reporting_job

  def _CreateRunner(self, job, model_task_name, logdir, tf_master, trial):
    """Create a runner."""
    evaler_job_name_prefix = 'evaler_'
    decoder_job_name_prefix = 'decoder_'

    tf.logging.info('Job %s start', job)
    common_args = (model_task_name, logdir, tf_master, trial)
    if job == 'controller':
      cfg = self.GetParamsForDataset('controller', 'Train')
      cfg.cluster.xla_device = 'cpu'
      return self.Controller(cfg, *common_args)
    elif job == 'trainer':
      cfg = self.GetParamsForDataset('trainer', 'Train')
      return self.Trainer(cfg, *common_args)
    elif job == 'trainer_client':
      cfg = self.GetParamsForDataset('trainer_client', 'Train')
      if py_utils.use_tpu():
        cfg.cluster.xla_device = 'tpu'
        return self.TrainerTpu(cfg, *common_args)
      else:
        return self.Trainer(cfg, *common_args)
    elif job == 'train_summaries':
      cfg = self.GetParamsForDataset('train_summaries', 'Train')
      return self.TrainSummaries(cfg, *common_args)
    elif job.startswith(evaler_job_name_prefix):
      dataset_name = job[len(evaler_job_name_prefix):]
      cfg = self.GetParamsForDataset('evaler', dataset_name)
      return self.Evaler(dataset_name.lower(), cfg, *common_args)
    elif job.startswith(decoder_job_name_prefix):
      dataset_name = job[len(decoder_job_name_prefix):]
      cfg = self.GetParamsForDataset('decoder', dataset_name)
      return self.Decoder(dataset_name.lower(), cfg, *common_args)
    elif job in ('ps', 'worker', 'input'):
      self._tf_server.join()
    elif job == 'executor_tpu':
      ps_cfg_dict, train_cfg = self.GetExecutorParams()
      return self.ExecutorTpu(train_cfg, ps_cfg_dict, *common_args)
    else:
      raise ValueError('job %s is not supported' % job)

  def CreateRunners(self, jobs, logdir, trial=base_trial.NoOpTrial()):
    """Creates a list of runners based on `FLAGS.mode`.

    Args:
      jobs: a list of runner jobs.
      logdir: the directory used for logging, usually on CNS.
      trial: optional `Trial` object, used for reporting measures and early
        stopping.

    Returns:
      A list of `.BaseRunner`, one per job in `jobs`.
    """
    all_runners = []
    is_training = 'trainer' in jobs or 'trainer_client' in jobs
    for j in jobs:
      tf_master = FLAGS.tf_master
      # Ensure that decoder or evaler threads do not clobber variables being
      # updated by trainer by forcing them to use independent sessions.
      if is_training and (j.startswith('decoder') or j.startswith('evaler')):
        tf_master = ''

      runner = self._CreateRunner(j, FLAGS.model_task_name, logdir, tf_master,
                                  trial)
      all_runners.append(runner)
    return all_runners

  def StartRunners(self, all_runners):
    """Runs `all_runners` in parallel threads.

    Returns when all of them finish.

    Args:
      all_runners: a list of `.BaseRunner`.

    Returns:
      None.
    """
    tf.logging.info('Starting runners')
    if len(all_runners) == 1 and not all_runners[0].enqueue_ops:
      # If there is only one runner and it does not have an enqueue thread, just
      # run it directly here.
      all_runners[0].Start()
    else:
      threads = []
      for runner in all_runners:
        runner_class_name = str(runner)
        t = threading.Thread(target=runner.Start, name=runner_class_name)
        t.daemon = True
        t.start()
        threads.append(t)
        if runner.enqueue_ops:
          tf.logging.info('Total num runner.enqueue_ops: %d',
                          len(runner.enqueue_ops))
          for i, enqueue_op in enumerate(runner.enqueue_ops):

            def StartEnqueue(runner, op):
              tf.logging.info('Starting enqueue op %s', op.name)
              return lambda: runner.StartEnqueueOp(op)

            enqueue_name = '%s-enqueue-%d' % (runner_class_name, i)
            tq = threading.Thread(
                target=StartEnqueue(runner, enqueue_op), name=enqueue_name)
            tq.start()
            threads.append(tq)
      tf.logging.info('Waiting for runners to finish...')
      for t in threads:
        tf.logging.info('Waiting for thread to finish: %s' % t.name)
        while True:
          t.join(1)
          if not t.is_alive():
            break
    tf.logging.info('All runners done.')

  def RunTrial(self, job, logdir, trial):
    """A wrapper function for running a trial."""
    # Run each job in separate process/task
    # TODO(rpang): add support for running evaler_test and decoder.
    self.StartRunners(self.CreateRunners([job], logdir, trial))

  def MaybeConfigRunLocally(self):
    """Update flags if configured to run locally."""
    if not FLAGS.run_locally:
      # Do nothing
      return

    FLAGS.tf_master = tf.distribute.Server.create_local_server().target

    if not FLAGS.mode:
      FLAGS.mode = 'sync'

    if not FLAGS.job:
      if FLAGS.run_locally == 'tpu':
        FLAGS.job = 'trainer_client'
      elif FLAGS.mode == 'async':
        FLAGS.job = 'controller,trainer'
      else:
        FLAGS.job = 'controller,trainer_client'

    FLAGS.task = 0
    local_job = '/job:localhost'
    FLAGS.controller_job = local_job

    FLAGS.worker_job = local_job
    FLAGS.worker_replicas = 1
    if FLAGS.run_locally == 'gpu':
      if not FLAGS.worker_gpus:
        FLAGS.worker_gpus = 1
    else:
      FLAGS.worker_gpus = 0
    if FLAGS.run_locally == 'tpu':
      FLAGS.xla_device = 'tpu'
      FLAGS.enable_asserts = False
    else:
      FLAGS.worker_tpus = 0

    if not FLAGS.worker_split_size:
      FLAGS.worker_split_size = 1

    FLAGS.ps_job = local_job
    FLAGS.ps_replicas = 1
    FLAGS.ps_gpus = 0

    FLAGS.input_job = local_job
    FLAGS.input_replicas = 0

    FLAGS.evaler_job = local_job
    FLAGS.evaler_replicas = 1
    if FLAGS.run_locally == 'gpu':
      FLAGS.evaler_gpus = 1
    else:
      FLAGS.evaler_gpus = 0

    FLAGS.decoder_job = local_job
    FLAGS.decoder_replicas = 1
    if FLAGS.run_locally == 'gpu':
      FLAGS.decoder_gpus = 1
    else:
      FLAGS.decoder_gpus = 0

  def InspectParams(self):
    r"""Print out all the params.

    An example to run this mode:

    bazel-bin/lingvo/trainer --logtostderr \
      --model=image.mnist.LeNet5 --mode=inspect_params --logdir=/tmp/lenet5 \
      --run_locally=cpu
    """
    FLAGS.mode = 'sync'
    cls = self.model_registry.GetClass(self._model_name)
    tf.io.gfile.makedirs(FLAGS.logdir)
    for dataset in datasets.GetDatasets(cls):
      if FLAGS.inspect_params_dataset_name is not None:
        if dataset not in FLAGS.inspect_params_dataset_name:
          continue
      p = self.GetParamsForDataset('controller', dataset)
      outf = os.path.join(FLAGS.logdir, dataset.lower() + '-params.txt')
      tf.logging.info('Write all params for {} to {}'.format(dataset, outf))
      with tf.io.gfile.GFile(outf, 'w') as f:
        f.write(p.ToText())

  def InspectModel(self):
    """Prints out model analysis for the model."""
    FLAGS.mode = 'sync'
    p = self.GetParamsForDataset('controller', 'Train')
    c = cluster_factory.Cluster(p.cluster)
    model_part_regex = FLAGS.inspect_model_part_regex
    part_pattern = None
    if model_part_regex:
      part_pattern = {}
      for pat_str in model_part_regex:
        first_colon = pat_str.find(':')
        if first_colon < 0:
          msg = f'Cannot understand --inspect_model_part_regex={pat_str}.'
          raise ValueError(msg)
        name = pat_str[:first_colon]
        pattern = pat_str[first_colon + 1:]
        part_pattern[name] = pattern

    with tf.Graph().as_default(), c, tf.device(c.GetPlacer()):
      analysis, _ = summary_utils.ModelAnalysis(
          p.Instantiate(),
          topn=FLAGS.inspect_model_topn,
          part_pattern=part_pattern)
    print(analysis)

  def InspectDatasets(self):
    """Prints out datasets configured for the model."""
    cls = self.model_registry.GetClass(self._model_name)
    print(','.join([dataset.lower() for dataset in datasets.GetDatasets(cls)]))

  def InspectDecoder(self):
    """Prints out datasets configured for the decoder."""
    cls = self.model_registry.GetClass(self._model_name)
    params = cls()

    has_decoder = False
    if issubclass(cls, base_model_params.SingleTaskModelParams):
      has_decoder = params.Task(
      ).cls.CreateDecoderMetrics != base_model.BaseTask.CreateDecoderMetrics
    else:
      for _, task_param in params.Model().task_params.IterParams():
        has_decoder |= (
            task_param.cls.CreateDecoderMetrics !=
            base_model.BaseTask.CreateDecoderMetrics)
    if has_decoder:
      # We assume that the proper decoder is implemented.
      self.InspectDatasets()
    else:
      print('')

  def SetModelName(self, model_name):
    """Sets the model name."""
    self._model_name = model_name

  def WriteInferenceGraph(self, cfg=None, prune_graph=True):
    """Generates the inference graphs for a given model.

    Args:
      cfg: Full `~.hyperparams.Params` for the model class. If present, this cfg
        will be used instead of retrieving from model_registry.
      prune_graph: If true, prune the graph to just the parts we need.

    Returns:
      InferenceGraph proto for cpu.
    """
    inference_graph_dir = os.path.join(FLAGS.logdir, 'inference_graphs')
    tf.io.gfile.makedirs(inference_graph_dir)
    tf.logging.info('Writing inference graphs to dir: %s', inference_graph_dir)

    if not cfg:
      cfg = self.model_registry.GetParams(self._model_name,
                                          FLAGS.inference_dataset_name)

    task_names = [FLAGS.model_task_name]
    if (issubclass(cfg.cls, base_model.MultiTaskModel) and
        not FLAGS.model_task_name):
      task_names = base_model.MultiTaskModel.TaskNames(cfg)

    inference_graph_proto = None

    if FLAGS.inference_graph_filename:
      # Custom inference graph.
      for task_name in task_names:
        filename_prefix = FLAGS.inference_graph_filename
        if task_name:
          filename_prefix = '%s_inference' % task_name
        filename_prefix = os.path.join(inference_graph_dir, filename_prefix)

        device = ''
        var_options = None
        if FLAGS.inference_graph_device == 'tpu':
          device = 'tpu'
          var_options = 'ON_DEVICE'
        device_options = inference_graph_exporter.InferenceDeviceOptions(
            device=device,
            retain_device_placement=False,
            var_options=var_options,
            gen_init_op=FLAGS.inference_gen_tpu_init_op,
            dtype_override=None,
            fprop_dtype_override=None)
        inference_graph_proto = (
            self.inference_graph_exporter.InferenceGraphExporter.Export(
                model_cfg=cfg,
                model_task_name=task_name,
                device_options=device_options,
                export_path=filename_prefix + '.pbtxt',
                random_seed=FLAGS.inference_graph_random_seed,
                prune_graph=prune_graph))
    else:
      for task_name in task_names:
        filename_prefix = 'inference'
        if task_name:
          filename_prefix = '%s_inference' % task_name
        filename_prefix = os.path.join(inference_graph_dir, filename_prefix)

        # Standard inference graph.
        try:
          inference_graph_proto = (
              self.inference_graph_exporter.InferenceGraphExporter.Export(
                  model_cfg=cfg,
                  model_task_name=task_name,
                  export_path=filename_prefix + '.pbtxt',
                  random_seed=FLAGS.inference_graph_random_seed,
                  prune_graph=prune_graph))
        except NotImplementedError as e:
          tf.logging.error('Cannot write inference graph: %s', e)

        # TPU inference graph. Not all models support it so fail silently.
        try:
          device_options = self.inference_graph_exporter.InferenceDeviceOptions(
              device='tpu',
              retain_device_placement=False,
              var_options='ON_DEVICE',
              gen_init_op=FLAGS.inference_gen_tpu_init_op,
              dtype_override=None,
              fprop_dtype_override=None)
          self.inference_graph_exporter.InferenceGraphExporter.Export(
              model_cfg=cfg,
              model_task_name=task_name,
              device_options=device_options,
              export_path=filename_prefix + '_tpu.pbtxt',
              random_seed=FLAGS.inference_graph_random_seed,
              prune_graph=prune_graph)
        except Exception as e:  # pylint: disable=broad-except
          tf.logging.error('Error exporting TPU inference graph: %s' % e)

    if FLAGS.graph_def_filename and inference_graph_proto:
      for graph_def_filename in FLAGS.graph_def_filename:
        tf.logging.info('Writing graphdef: %s', graph_def_filename)
        dir_path = os.path.dirname(graph_def_filename)
        if (not tf.io.gfile.exists(dir_path) or
            not tf.io.gfile.isdir(dir_path)):
          tf.io.gfile.makedirs(dir_path)
        with tf.io.gfile.GFile(graph_def_filename, 'w') as f:
          f.write(text_format.MessageToString(inference_graph_proto.graph_def))

    return inference_graph_proto

  def RunEvalerOnce(self):
    """Run once evaler."""
    m = re.match(r'evaler_once_([^_@]+)@(\d+)', FLAGS.job)
    dataset_name, ckpt_id = m.group(1), int(m.group(2))
    cfg = self.GetParamsForDataset('evaler', dataset_name)
    evaler = self.Evaler(dataset_name.lower(), cfg, FLAGS.model_task_name,
                         FLAGS.logdir, FLAGS.tf_master)
    evaler.EvalCheckpoint(ckpt_id)

  def Start(self):
    """Start the process."""
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('tf_api_version: %s', tf.summarize_tf2_status())

    if FLAGS.mode == 'inspect_params':
      self.InspectParams()
      return

    if FLAGS.mode == 'inspect_model':
      self.InspectModel()
      return

    if FLAGS.mode == 'inspect_evaler':
      self.InspectDatasets()
      return

    if FLAGS.mode == 'inspect_decoder':
      self.InspectDecoder()
      return

    if FLAGS.mode == 'write_inference_graph':
      self.WriteInferenceGraph()
      return

    if FLAGS.mode == 'shell':
      runners.StartShell(locals())
      return

    assert FLAGS.mode in ['sync', 'async']

    self.MaybeConfigRunLocally()
    self.MaybeConfigRunDistributed()
    self.MaybeConfigCloudTpu()
    self.MaybeInferTPUClusterParams()
    self.MaybeLaunchTensorFlow()

    if FLAGS.job.startswith('evaler_once_'):
      # E.g., trainer --model=foo.bar.Model --logdir=...
      # --run_locally=cpu --mode=sync --job=evaler_once_test@65200
      self.RunEvalerOnce()
      return

    self.StartRunners(self.CreateRunners(FLAGS.job.split(','), FLAGS.logdir))


def main(unused_argv):
  RunnerManager(FLAGS.model).Start()


if __name__ == '__main__':
  tf.flags.mark_flag_as_required('model')
  FLAGS(sys.argv, known_only=True)
  if FLAGS.disable_tf2:
    tf.disable_v2_behavior()
  py_utils.SetEagerMode(FLAGS.use_eager)
  tf.config.run_functions_eagerly(FLAGS.run_functions_eagerly)
  if FLAGS.enable_tf_data_debug_mode:
    tf.data.experimental.enable_debug_mode()
  model_imports.ImportParams(FLAGS.model)
  FLAGS.unparse_flags()
  tf.app.run(main)
