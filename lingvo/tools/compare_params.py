# Lint as: python3
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
"""Library for comparing two models / hyperparams."""

from lingvo import compat as tf
from lingvo import model_registry
import six


def _hyperparams_text_to_dict(cfg_text):
  """Converts hyperparams config text to a dictionary of key-value pairs."""
  txt_list = six.ensure_str(cfg_text).split("\n")
  pair_list = []
  for v in txt_list:
    if not v:
      continue
    vals = v.split(" : ")
    if len(vals) != 2:
      raise ValueError(v)
    pair_list.append(vals)
  return dict(pair_list)


def hyperparams_text_diff(cfg1_text, cfg2_text):
  """Computes the differences between two hyperparams.Params texts.

  Args:
    cfg1_text: A hyperparams.Params().ToText() of the first model config.
    cfg2_text: A hyperparams.Params().ToText() of the second model config.

  Returns:
    A tuple of 3 elements:

    - cfg1_not_cfg2: A list of keys in cfg1 but not cfg2.
    - cfg2_not_cfg1: A list of keys in cfg2 but not cfg1.
    - cfg1_and_cfg2_diff: A dict of common keys whose config values differ: each
      value is a tuple of the config values from cfg1 and cfg2 respectively.
  """
  cfg1_dict = _hyperparams_text_to_dict(cfg1_text)
  cfg2_dict = _hyperparams_text_to_dict(cfg2_text)
  cfg1_keys = set(cfg1_dict.keys())
  cfg2_keys = set(cfg2_dict.keys())

  cfg1_not_cfg2 = sorted(list(cfg1_keys - cfg2_keys))
  cfg2_not_cfg1 = sorted(list(cfg2_keys - cfg1_keys))

  def get_class_name(v):
    try:
      idx = v.rindex("/")
      return v[idx + 1:]
    except ValueError:
      return v

  cfg1_and_cfg2_diff = {}
  for k_intersection in cfg1_keys & cfg2_keys:
    c1v = cfg1_dict[k_intersection]
    c2v = cfg2_dict[k_intersection]
    if k_intersection.endswith(".cls"):
      c1v = get_class_name(c1v)
      c2v = get_class_name(c2v)

    if c1v != c2v:
      cfg1_and_cfg2_diff[k_intersection] = (c1v, c2v)

  return cfg1_not_cfg2, cfg2_not_cfg1, cfg1_and_cfg2_diff


def print_hyperparams_text_diff(path1, path2, cfg1_not_cfg2, cfg2_not_cfg1,
                                cfg1_and_cfg2_diff):
  """Prints the differences of the output of hyperparams_text_diff.

  Args:
    path1: Name of registered model or path to model 1.
    path2: Name of registered model or path to model 2.
    cfg1_not_cfg2: A list of keys in cfg1 but not cfg2.
    cfg2_not_cfg1: A list of keys in cfg2 but not cfg1.
    cfg1_and_cfg2_diff: A dictionary of common keys whose config values differ;
      each value is a tuple of the config values from cfg1 and cfg2
      respectively.
  """
  if cfg1_not_cfg2:
    print("\n\nKeys in %s but not %s: \n%s\n\n" %
          (path1, path2, "\n".join(cfg1_not_cfg2)))
  if cfg2_not_cfg1:
    print("\n\nKeys in %s but not %s: \n%s\n\n" %
          (path2, path1, "\n".join(cfg2_not_cfg1)))

  if cfg1_and_cfg2_diff:
    print("\n\nKeys with differences and their values: \n\n")
    for k, v in sorted(cfg1_and_cfg2_diff.items()):
      v1, v2 = v
      print("%s: [%s] vs. [%s]" % (k, v1, v2))
    print("\n\n")


def get_model_params_as_text(model_path):
  try:
    cfg = model_registry.GetParams(model_path, "Train")
    return cfg.ToText()
  except LookupError:
    # Try reading as file.
    return tf.io.gfile.GFile(model_path).read()
