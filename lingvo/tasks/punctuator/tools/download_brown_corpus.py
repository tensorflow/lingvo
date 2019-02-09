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
"""Downloads and processes the Brown Corpus (http://www.nltk.org/nltk_data)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import string
from xml.etree import ElementTree
import tensorflow as tf

tf.flags.DEFINE_string("outdir", "/tmp/punctuator_data",
                       "The output directory.")

FLAGS = tf.flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  basename = "brown_tei"
  fname = basename + ".zip"
  url = ("https://raw.githubusercontent.com/nltk/nltk_data/" +
         "gh-pages/packages/corpora/" + fname)
  sha256 = "335bec1ea6362751d5d5c46970137ebb01c80bf7d7d75558787729d275e0a687"

  tf.keras.utils.get_file(
      fname, url, file_hash=sha256, cache_subdir=FLAGS.outdir, extract=True)

  tf.logging.info("\nDownload completed. Preprocessing...")

  with open(os.path.join(FLAGS.outdir, basename, "Corpus.xml"), "r") as xml:
    root = ElementTree.fromstring(xml.read().replace(
        'xmlns="http://www.tei-c.org/ns/1.0"', ""))
  sentences = []
  for sentence in root.findall("./TEI/text/body/p/s"):
    # Example input sentence:
    # <s n="1"><w type="AT">The</w> <w subtype="TL" type="NP">Fulton</w>
    # <w subtype="TL" type="NN">County</w> <w subtype="TL" type="JJ">Grand</w>
    # <w subtype="TL" type="NN">Jury</w> <w type="VBD">said</w>
    # <w type="NR">Friday</w> <w type="AT">an</w> <w type="NN">investigation</w>
    # <w type="IN">of</w> <w type="NPg">Atlanta's</w> <w type="JJ">recent</w>
    # <w type="NN">primary</w> <w type="NN">election</w>
    # <w type="VBD">produced</w> <c type="pct">``</c> <w type="AT">no</w>
    # <w type="NN">evidence</w> <c type="pct">''</c> <w type="CS">that</w>
    # <w type="DTI">any</w> <w type="NNS">irregularities</w>
    # <w type="VBD">took</w> <w type="NN">place</w> <c type="pct">.</c> </s>
    # Example output text:
    # The Fulton County Grand Jury said Friday an investigation of Atlanta's
    # recent primary election produced "no evidence" that any irregularities
    # took place.
    text = ""
    prepend_space = False
    for child in sentence:
      if child.tag == "w":
        if prepend_space:
          text += " "
        text += child.text
        prepend_space = True
      elif child.tag == "c":
        if child.text == "``":
          if prepend_space:
            text += " "
          text += '"'
          prepend_space = False
        elif child.text == "''":
          text += '"'
          prepend_space = True
        elif child.text == "'":
          if prepend_space:
            text += " '"
            prepend_space = False
          else:
            text += "'"
            prepend_space = True
        elif child.text == "(" or child.text == "[":
          if prepend_space:
            text += " "
          text += child.text
          prepend_space = False
        elif child.text == "-" or child.text == "--":
          if prepend_space:
            text += " "
          text += child.text
          prepend_space = True
        else:
          text += child.text
          prepend_space = True
    text = text.replace("!!", "!").replace("??", "?").replace("--", "-")
    text = text.replace("**", "*").replace(";;", ";").replace("::", ":")
    text = text.replace(",,", ",")

    # Filter out bad sentences.
    if not set(text) & set(string.ascii_letters):
      # No letters.
      continue
    if text.count('"') % 2 != 0:
      # Uneven number of quotes.
      continue
    if text.count("(") != text.count(")") or text.count("[") != text.count("]"):
      # Unbalanced parenthesis.
      continue
    if (text[0] == '"' and text[-1] == '"' or
        text[0] == "(" and text[-1] == ")" or
        text[0] == "[" and text[-1] == "]"):
      text = text[1:-1]
    if text[0] not in string.ascii_letters and text[0] not in string.digits:
      # Doesn't start with a letter or number.
      continue
    text = text[:1].upper() + text[1:]
    sentences.append(text)
  sentences = sorted(set(sentences))
  random.seed(1234)
  random.shuffle(sentences)

  with open(os.path.join(FLAGS.outdir, "train.txt"), "w") as f:
    for line in sentences[:int(len(sentences) * 0.95)]:
      f.write("%s\n" % line)

  with open(os.path.join(FLAGS.outdir, "test.txt"), "w") as f:
    for line in sentences[int(len(sentences) * 0.95):]:
      f.write("%s\n" % line)

  tf.logging.info("All done.")


if __name__ == "__main__":
  tf.app.run(main)
