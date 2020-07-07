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
"""Converts a keras dataset into a tf checkpoint.

E.g.

.. code-block:: bash

  $ bazel run lingvo/tools:keras2ckpt -- --dataset=mnist
"""

import os
import lingvo.compat as tf
from tensorflow.python.ops import io_ops

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("dataset", "", "The dataset name.")
tf.flags.DEFINE_string("out", "", "The output checkpoint path prefix.")


def main(argv):
  del argv  # Unused.

  dataset = getattr(tf.keras.datasets, FLAGS.dataset)
  (x_train, y_train), (x_test, y_test) = dataset.load_data()

  def wrap(val):
    dtype = tf.as_dtype(val.dtype)
    assert dtype != tf.string  # tf.string is not supported by py_func.
    return tf.py_func(lambda: val, [], dtype)

  out_prefix = FLAGS.out or os.path.join("/tmp", FLAGS.dataset, FLAGS.dataset)
  tf.logging.info("Save %s dataset to %s ckpt." %
                       (FLAGS.dataset, out_prefix))

  with tf.Session() as sess:
    sess.run(
        io_ops.save_v2(
            prefix=out_prefix,
            tensor_names=["x_train", "y_train", "x_test", "y_test"],
            shape_and_slices=[""] * 4,
            tensors=[wrap(x_train),
                     wrap(y_train),
                     wrap(x_test),
                     wrap(y_test)]))


if __name__ == "__main__":
  tf.app.run(main)
