#!/bin/bash
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
#
# Updates PUBLICATIONS.md based on publications.bib.
#
# To run, you must first `apt-get install bibtex2html`.

readonly OUTPUT_FILE=PUBLICATIONS.md

echo -e "# List of publications using Lingvo.\n" > ${OUTPUT_FILE}

for topic in \
    'Translation' \
    'Speech recognition' \
    'Language understanding' \
    'Speech synthesis' \
    'Speech-to-text translation';  do
  echo -e "\n\n## ${topic}"
  bib2bib -c "annote='${topic}'" publications.bib \
      | bibtex2html -s ieeetr -nodoc -nobibsource -nofooter -nf pdf "pdf"
done >> ${OUTPUT_FILE}
