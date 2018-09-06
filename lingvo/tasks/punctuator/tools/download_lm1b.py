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
"""Downloads and processes lm1b dataset (http://www.statmt.org/lm-benchmark)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import string
import tensorflow as tf

tf.flags.DEFINE_string("outdir", "/tmp/lm1b", "The output directory.")

FLAGS = tf.flags.FLAGS


def main(_):
  basename = "1-billion-word-language-modeling-benchmark-r13output"
  fname = basename + ".tar.gz"
  url = "http://www.statmt.org/lm-benchmark/" + fname
  sha256hash = "01ba60381110baf7f189dfd2b8374de371e8c9a340835793f190bdae9e90a34e"

  tf.keras.utils.get_file(
      fname, url, file_hash=sha256hash, cache_subdir=FLAGS.outdir, extract=True)

  with open(os.path.join(FLAGS.outdir, basename, "grapheme.txt"), "w") as f:
    next_id = 0
    symbols = itertools.chain(["<unk>", "<s>", "</s>", " "],
                              string.ascii_lowercase, string.ascii_uppercase,
                              string.punctuation, range(10))
    for symbol in symbols:
      f.write("%s\t%d\n" % (symbol, next_id))
      next_id += 1


if __name__ == "__main__":
  tf.app.run(main)
