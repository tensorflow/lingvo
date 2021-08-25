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
"""Helper classes for computing performance metrics."""

import collections

import lingvo.compat as tf
from lingvo.core import hyperparams
from lingvo.core import plot
from lingvo.core import py_utils
from lingvo.core import scorers
import numpy as np
try:
  import sklearn.metrics  # pylint: disable=g-import-not-at-top
  HAS_SKLEARN = True
except ImportError:
  HAS_SKLEARN = False
try:
  import scipy.stats  # pylint: disable=g-import-not-at-top
  HAS_SCIPY_STATS = True
except ImportError:
  HAS_SCIPY_STATS = False


def CreateScalarSummary(name, simple_value):
  return tf.Summary(
      value=[tf.Summary.Value(tag=name, simple_value=simple_value)])


class BaseMetric:
  """Base class for aggregating statistics to compute a performance metric."""

  def Update(self, *args, **kwargs):
    """Updates this metric (e.g. accumulates statistics) from the arguments."""
    pass

  @property
  def value(self):
    """Current value of this metric."""
    return None

  def Summary(self, name):
    """Converts the current state of this metric to a `tf.Summary`.

    Args:
      name: A string to use as the summary value tag.

    Returns:
      A `tf.Summary` proto.
    """
    return CreateScalarSummary(name, self.value)


class ConfigurableMetric(BaseMetric):
  """A Metric class with configurable params."""

  @classmethod
  def Params(cls):
    p = hyperparams.InstantiableParams(cls)
    return p

  def __init__(self, params):
    self.params = params


class AverageMetric(BaseMetric):
  """Class to compute a weighted (arithmetic) average value metric."""

  def __init__(self):
    self._total_value = 0.0
    self._total_weight = 0.0

  def Update(self, value, weight=1.0):
    if weight < 0.0:
      raise ValueError('weight must be non-negative.  Got: %f' % weight)
    self._total_value += value * weight
    self._total_weight += weight

  # We may want both a getter and a setter method for total_value and
  # total_weight, respectively.
  def GetTotalValue(self):
    return self._total_value

  def SetTotalValue(self, val):
    self._total_value = val

  total_value = property(GetTotalValue, SetTotalValue)

  def GetTotalWeight(self):
    return self._total_weight

  def SetTotalWeight(self, val):
    self._total_weight = val

  total_weight = property(GetTotalWeight, SetTotalWeight)

  @property
  def value(self):
    return (self._total_value /
            self._total_weight if self._total_weight > 0 else 0)


class F1Metric(BaseMetric):
  """Class to compute F1 metrics."""

  def __init__(self):
    self._true_pos = 0.0
    self._false_pos = 0.0
    self._false_neg = 0.0

  def UpdateTruePositive(self, count=1.0):
    self._true_pos += count

  def UpdateFalsePositive(self, count=1.0):
    self._false_pos += count

  def UpdateFalseNegative(self, count=1.0):
    self._false_neg += count

  @property
  def value(self):
    if (self._true_pos + self._false_pos) > 0:
      precision = self._true_pos / (self._true_pos + self._false_pos)
    else:
      precision = 0.0
    if (self._true_pos + self._false_neg) > 0:
      recall = self._true_pos / (self._true_pos + self._false_neg)
    else:
      recall = 0.0
    if (precision + recall) > 0:
      return 2.0 * precision * recall / (precision + recall)
    else:
      return 0.0


class MCCMetric(F1Metric):
  """Class to compute Matthews correlation coefficient metric."""

  def __init__(self):
    super().__init__()
    self._true_neg = 0.0

  def UpdateTrueNegative(self, count=1.0):
    self._true_neg += count

  @property
  def value(self):
    tp = self._true_pos
    tn = self._true_neg
    fp = self._false_pos
    fn = self._false_neg

    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator > 0.0:
      return (tp * tn - fp * fn) / denominator**0.5
    else:
      return 0.0


class CorpusBleuMetric(BaseMetric):
  """Metric class to compute the corpus-level BLEU score."""

  def __init__(self, **kwargs):
    self._scorer = scorers.BleuScorer(**kwargs)

  def Update(self, ref_str, hyp_str):
    self._scorer.AddSentence(ref_str, hyp_str)

  @property
  def unsegmenter(self):
    return self._scorer.unsegmenter

  @property
  def value(self):
    return self._scorer.ComputeOverallScore()


class TpuEvalMetrics:
  """Manages computation of metrics during TPU execution.

  TPU execution runs a training loop on device. To get eval metrics out of this,
  metric values and weights must be carried through the loop. This requires
  passing initial values to the loop setup, updated the values during the loop,
  and doing a final aggregation after the loop. This class wraps the metrics
  dictionary so that the needed ops can be built at the right time as the
  training loop is built.

  Note that because the model is not constructed until the loop body function is
  called, the initial values must be known statically. This is done currently by
  hardcoding a limit on the number of metrics and casting each metric and value
  count to float32, regardless of the number of actual metrics the model
  produced.

  Note that this implementation computes the metrics over all replicas, for the
  last step of the loop only (could be changed to average over all loop steps
  instead).
  """

  def __init__(self, max_metrics=256):
    self._metrics = None
    self._max_metrics = max_metrics

    # Loop-carried values alternate value and weight; all values are scalars.
    self._initial_values = (2 *
                            self._max_metrics) * [tf.constant(0, tf.float32)]

  def SetMetrics(self, metric_dict, step_args):
    """Sets the metrics to evaluate and the per-step output tensors.

    Args:
      metric_dict: dict of (name -> (tensor of values, tensor of weights))
      step_args: the tensors being passed to the training loop body. These share
        the same structure of alternating value and weight scalars as the
        initial values and the output of this function.

    Returns:
      The tensors to return from the training loop body.  For entries that are
      for metrics in self._metrics, returns the value computed within the loop
      (the step_args value passed in); for all others, the value will never be
      used at the end and so the step_args value is passed through (which has
      the effect of passing the initial values through every iteration of the
      loop).
    """
    num_metrics = len(metric_dict)
    assert num_metrics <= self._max_metrics, ('Increase max_metrics to >= %d' %
                                              num_metrics)
    self._metrics = metric_dict

    # self._metrics contains a map of (metric_value,
    # metric_weight). We convert it into [metric_value *
    # metric_weight, metric_weight] to make it easier to aggregate
    # metric values across steps and TPU replicas.
    ret = []
    for _, (value, weight) in sorted(self._metrics.items()):
      assert value.shape.is_fully_defined(), ('%s' % value)
      assert weight.shape.is_fully_defined(), ('%s' % weight)
      weight = tf.cast(weight, tf.float32)
      value = tf.cast(value, tf.float32) * weight
      ret += [value, weight]
    # Each metric has two tensors: value and weight.
    assert len(ret) == 2 * num_metrics
    ret += list(step_args)[len(ret):]
    return ret

  @property
  def initial_values(self):
    """Returns the initial loop values."""
    return self._initial_values

  @property
  def metrics(self):
    return self._metrics

  def _Zip(self, values):
    assert isinstance(values, list)
    return list(zip(values[::2], values[1::2]))

  def FinalizeMetrics(self, loop_result):
    """Compute final average of the metrics, given loop_result tensors.

    To be called outside the training loop body , but still in the scope of
    tpu.batch_parallel.

    Args:
      loop_result: Result of the training loop.

    Returns:
      The tensors of the final avg values and total weights.
    """
    # Each metric has two tensors in the loop carrying result.
    metrics = loop_result[:2 * len(self._metrics.items())]
    # Aggregate across tpu replicas.
    metrics = [tf.tpu.cross_replica_sum(x) for x in metrics]
    ret = []
    for (value, weight) in self._Zip(metrics):
      value, weight = py_utils.WeightedAvg(
          tf.math.divide_no_nan(value, weight), weight)
      ret += [value, weight]
    return ret

  def PackMetricsValues(self, values):
    """Packs numpy values into a dict of metrics."""
    for k, v in zip(sorted(self._metrics.keys()), self._Zip(values)):
      self._metrics[k] = v

  def ToAverageMetrics(self):
    """Wrap the final metric values into AverageMetric objects.

    Returns:
      A dict that maps metric names to AverageMetric objects.
    """
    ret = {}
    for name, (value, weight) in self._metrics.items():
      avg_metric = AverageMetric()
      avg_metric.total_weight = weight
      avg_metric.total_value = weight * value
      ret[name] = avg_metric
    return ret


class AUCMetric(BaseMetric):
  """Class to compute the AUC score for binary classification."""

  def __init__(self, mode='roc', samples=-1):
    """Constructor of the class.

    Args:
      mode: Possible values: 'roc' or 'pr'.
      samples: The number of sample points to compute the AUC. If -1, include
        all points seen thus far.

    Raises:
      ImportError: If user has not installed sklearn, raise an ImportError.
    """
    if not HAS_SKLEARN:
      raise ImportError('AUCMetric depends on sklearn.')
    self._mode = mode
    self._samples = samples
    self._label = []
    self._prob = []
    self._weight = []
    if self._mode == 'roc':
      self._curve_fn = sklearn.metrics.roc_curve
      self._score_fn = sklearn.metrics.roc_auc_score
      self._plot_labels = ['False Positive Rate', 'True Positive Rate']
    elif self._mode == 'pr':
      self._curve_fn = sklearn.metrics.precision_recall_curve
      self._score_fn = sklearn.metrics.average_precision_score
      self._plot_labels = ['Recall', 'Precision']
    else:
      raise ValueError('mode in AUCMetric must be one of "roc" or "pr".')

  def Update(self, label, prob, weight=None):
    """Updates the metrics.

    Args:
      label: An array to specify the groundtruth binary labels. Values must be
        either 0 or 1.
      prob: An array to specify the prediction probabilities. Values must be
        within [0, 1.0].
      weight: An array to specify the sample weight for the auc computation.
    """
    self._label += label
    self._prob += prob
    if weight:
      self._weight += weight
    else:
      self._weight += [1 for _ in range(len(label))]

    if self._samples > 0:
      self._label = self._label[-self._samples:]
      self._prob = self._prob[-self._samples:]
      self._weight = self._weight[-self._samples:]

  @property
  def value(self):
    try:
      auc_val = self._score_fn(
          self._label, self._prob, sample_weight=self._weight)
      # For precision/recall, _score_fn returns nan if only one type of label is
      # present rather than throwing a ValueError.
      if np.isnan(auc_val):
        return 0.0
      else:
        return auc_val
    except ValueError as exception:
      # In case self._label still has just 1 type of label, e.g. all(labels==0).
      if 'Only one class present in y_true.' in str(exception):
        return 0.0
      else:
        raise

  def Summary(self, name):

    def _Setter(fig, axes):
      # 20 ticks betweein 0 and 1.
      ticks = np.arange(0, 1.05, 0.05)
      axes.grid(b=True)
      axes.set_xlabel(self._plot_labels[0])
      axes.set_xticks(ticks)
      axes.set_ylabel(self._plot_labels[1])
      axes.set_yticks(ticks)
      fig.tight_layout()

    xs, ys, _ = self._curve_fn(
        self._label, self._prob, sample_weight=self._weight)
    if self._mode == 'pr':
      # Swap because sklearn returns <'precision', 'recall'>.
      xs, ys = ys, xs
    ret = plot.Curve(name=name, figsize=(12, 12), xs=xs, ys=ys, setter=_Setter)
    ret.value.add(tag=name, simple_value=self.value)
    return ret


class MultiClassAUCMetric(BaseMetric):
  """Class to compute mAP or mAUC for multiclass/multilabel classification.

  This metric is equivalent to mAP (mean average precision) or mAUC (multiclass
  AUC-ROC) for non-binary classification tasks. Another perspective of this
  measure defines it as the per-class average of the auc roc or auc pr curves.
  """

  def __init__(self, num_classes, mode='roc', samples=-1):
    """Construct a MultiClassAUCMetric instance.

    Args:
      num_classes: The number of classes.
      mode: Possible values: 'roc' or 'pr'.
      samples: The number of sample points to compute the AUC. If -1, include
        all points seen thus far.
    """
    self._num_classes = num_classes
    self._class_auc_metrics = []
    for _ in range(self._num_classes):
      self._class_auc_metrics.append(AUCMetric(mode, samples))

    # Track the classes that are present in the evaluation set.
    self._labels_seen_dict = {i: 0 for i in range(self._num_classes)}

  def Update(self, labels, probs, weights=None):
    """Updates the MultiClassAUCMetric.

    Args:
      labels: An list of arrays specifying the groundtruth binary labels for
        each class. Values must be either 0 or 1.
      probs: A list of arrays specifying teh prediction probabilities for each
        class. Values must be within [0, 1.0]
      weights: An array to specify the sample weight for each example in the auc
        computation.
    """
    for i, auc_metric in enumerate(self._class_auc_metrics):
      sliced_labels = labels[i]
      sliced_probs = probs[i]
      auc_metric.Update(sliced_labels, sliced_probs, weights)

      for label in sliced_labels:
        self._labels_seen_dict[i] = max(label, self._labels_seen_dict[i])

  @property
  def value(self):
    auc_values = []
    for index, auc_metric in enumerate(self._class_auc_metrics):
      if self._labels_seen_dict[index]:
        auc_values.append(auc_metric.value)
    num_classes_present_in_eval = sum(self._labels_seen_dict.values())
    return np.sum(auc_values) / num_classes_present_in_eval


class CorrelationMetric(BaseMetric):
  """Class to compute correlation."""

  def __init__(self, mode='pearson', samples=-1):
    """Constructor of the class.

    Args:
      mode: Possible values: 'pearson', 'spearman', 'kendalltau'.
      samples: The number of sample points to compute the correlation. If -1,
        include all points seen thus far.

    Raises:
      ImportError: If user has not installed scipy.stats, raise an ImportError.
    """
    if not HAS_SCIPY_STATS:
      raise ImportError('CorrelationMetric depends on scipy.stats.')

    assert mode in ['pearson', 'spearman', 'kendalltau']
    self._mode = mode
    self._samples = samples
    self._target = []
    self._pred = []

  def Update(self, target, pred):
    """Updates the metrics.

    Args:
      target: An array to specify the groundtruth float target.
      pred: An array to specify the prediction.
    """
    self._target += target
    self._pred += pred

    if self._samples > 0:
      self._target = self._target[-self._samples:]
      self._pred = self._pred[-self._samples:]

  @property
  def value(self):
    # only use the correlation, p-value is ignored.
    if self._mode == 'pearson':
      return scipy.stats.pearsonr(self._target, self._pred)[0]
    elif self._mode == 'spearman':
      return scipy.stats.spearmanr(self._target, self._pred)[0]
    else:
      return scipy.stats.kendalltau(self._target, self._pred)[0]


class AverageKeyedCorrelationMetric(BaseMetric):
  """Class to compute correlation per key, then report average across all keys."""

  def __init__(self, mode='pearson', bypass_nan=True):
    """Constructor of the class.

    Args:
      mode: Possible values: 'pearson', 'spearman', 'kendalltau'.
      bypass_nan: for keys that has only 1 example, the metric will be NaN,
        turn on this flag to skip those keys.
        Depending on the scipy library, it may raise ValueException instead
        of produce NaN. In this case, the Exception will be suppressed and the
        key is skipped during calculation.

    Raises:
      ImportError: If user has not installed scipy.stats, raise an ImportError.
    """
    if not HAS_SCIPY_STATS:
      raise ImportError('CorrelationMetric depends on scipy.stats.')

    assert mode in ['pearson', 'spearman', 'kendalltau']
    self._mode = mode
    self._bypass_nan = bypass_nan
    self._target = collections.defaultdict(list)
    self._pred = collections.defaultdict(list)

  def Update(self, key, target, pred):
    """Updates the metrics.

    Args:
      key: The key this (target, pred) pair belongs to.
      target: An array to specify the groundtruth float target.
      pred: An array to specify the prediction.
    """
    self._target[key] += target
    self._pred[key] += pred

  @property
  def value(self):
    # only use the correlation, p-value is ignored.
    if self._mode == 'pearson':
      corr_f = scipy.stats.pearsonr
    elif self._mode == 'spearman':
      corr_f = scipy.stats.spearmanr
    else:
      corr_f = scipy.stats.kendalltau

    results = []
    for k in self._target:
      target = self._target[k]
      pred = self._pred[k]
      try:
        raw_corr = corr_f(target, pred)[0]
        if not self._bypass_nan or not np.isnan(raw_corr):
          results.append(raw_corr)
      except ValueError:
        continue

    if not self._bypass_nan or results:
      return np.mean(results)
    else:
      return 0.0


class SamplingMetric(ConfigurableMetric):
  """Sampling metric base class.

  Subclasses must implement _CreateSummary(); sampling will be handled
  by this base class.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_samples', 8, 'The number of samples to store uniformly.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self._sampler = py_utils.UniformSampler(num_samples=p.num_samples)
    self._summary = None

  @property
  def samples(self):
    """Returns an iterable of sampled decoded outputs to compute Summaries."""
    return self._sampler.samples

  def Update(self, decoded_outputs):
    """Samples the input decoded_outputs NestedMap.

    Args:
      decoded_outputs: A `.NestedMap`.
    """
    self._sampler.Add(decoded_outputs)
    # Invalidate cache.
    self._summary = None

  def Summary(self, name):
    if self._summary is None:
      self._summary = self._CreateSummary(name)
      self._sampler = py_utils.UniformSampler(
          num_samples=self.params.num_samples)
    return self._summary

  def _CreateSummary(self, name):
    """Returns a tf.Summary for this metric."""
    raise NotImplementedError()
