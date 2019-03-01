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

import glob
import os
import tensorflow as tf

tf.flags.DEFINE_string("outdir", "/tmp/lm1b", "The output directory.")
tf.flags.DEFINE_integer(
    "count_cutoff", 3, "Ignore tokens that appear fewer than "
    "this amount of times when creating the vocab file.")

FLAGS = tf.flags.FLAGS


def main(_):
  basename = "1-billion-word-language-modeling-benchmark-r13output"
  fname = basename + ".tar.gz"
  url = "http://www.statmt.org/lm-benchmark/" + fname
  sha256hash = "01ba60381110baf7f189dfd2b8374de371e8c9a340835793f190bdae9e90a34e"

  tf.keras.utils.get_file(
      fname, url, file_hash=sha256hash, cache_subdir=FLAGS.outdir, extract=True)

  tf.logging.info("Generating vocab file. This may take a few minutes.")
  vocab = {}
  for fname in glob.glob(
      os.path.join(FLAGS.outdir, basename,
                   "training-monolingual.tokenized.shuffled", "news.en*")):
    with open(fname) as f:
      for line in f:
        for w in line.split():
          vocab[w] = vocab.get(w, 0) + 1

  with open(os.path.join(FLAGS.outdir, basename, "vocab.txt"), "w") as f:
    f.write("<epsilon>\t0\n<S>\t1\n</S>\t2\n<UNK>\t3\n")
    id = 4
    for k, v in sorted(vocab.items(), key=lambda kv: (-kv[1], kv[0])):
      if v < FLAGS.count_cutoff:
        break
      f.write("%s\t%d\n" % (k, id))
      id += 1


if __name__ == "__main__":
  tf.app.run(main)
