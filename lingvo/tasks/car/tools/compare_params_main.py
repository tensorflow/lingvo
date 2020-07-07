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
"""Tool for comparing two models / hyperparams."""

from absl import flags
from lingvo import compat as tf
from lingvo.tasks.car.params import kitti  # pylint:disable=unused-import
from lingvo.tools import compare_params

FLAGS = flags.FLAGS
flags.DEFINE_string("model1", None,
                    "Registered name or path to params.txt of model 1.")
flags.DEFINE_string("model2", None,
                    "Registered name or path to params.txt of model 2.")


def main(argv):
  if len(argv) > 1:
    raise tf.app.UsageError("Too many command-line arguments.")

  cfg1_text = compare_params.get_model_params_as_text(FLAGS.model1)
  cfg2_text = compare_params.get_model_params_as_text(FLAGS.model2)

  cfg1_not_cfg2, cfg2_not_cfg1, cfg1_and_cfg2_diff = (
      compare_params.hyperparams_text_diff(cfg1_text, cfg2_text))

  compare_params.print_hyperparams_text_diff(FLAGS.model1, FLAGS.model2,
                                             cfg1_not_cfg2, cfg2_not_cfg1,
                                             cfg1_and_cfg2_diff)
  return 0


if __name__ == "__main__":
  tf.app.run(main)
