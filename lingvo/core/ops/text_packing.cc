/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "lingvo/core/ops/text_packing.h"

#include <utility>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace lingvo {

TextPacking::TextPacking(int columns, int batch, std::vector<int> times,
                         int align, bool pack, int spread_first_n,
                         bool use_last_fit)
    : columns_(columns),
      batch_(batch),
      times_(std::move(times)),
      align_(align),
      pack_(pack),
      spread_first_n_(std::min(spread_first_n, batch)),
      use_last_fit_(use_last_fit),
      wpos_(batch, std::vector<int>(columns, 0)),
      seq_(batch, 0),
      last_fit_(0),
      counter_(0) {
  CHECK_EQ(columns_, times_.size()) << "The size of `times` must be `columns`";
}

bool TextPacking::Add(const std::vector<int>& lens, PackingIndex* p) {
  CHECK_EQ(columns_, lens.size());
  CHECK(p);
  // Start searching for batch position where 'lens' could fit.
  // If we find a fit, on next call we start searching from the same position.
  // Because if we always start from 0 this loop becomes O(N^2) in batch size.
  // TODO(krikun): add a benchmark for very large batch sizes
  if (!use_last_fit_) {
    last_fit_ = 0;
  }
  // b is the index of the row on which we see if the current sequence fits.
  int b = last_fit_;
  for (int i = 0; i < batch_; i++, b++) {
    if (counter_ < spread_first_n_) {
      b = counter_;
      last_fit_ = 0;
    }
    b %= batch_;
    bool fits = true;
    for (int c = 0; c < columns_; c++) {
      if (wpos_[b][c] + lens[c] > times_[c]) {
        fits = false;
        break;
      }
    }
    if (fits) {
      last_fit_ = b;
      p->batch = b;
      p->time.resize(columns_);
      for (int c = 0; c < columns_; c++) {
        p->time[c] = wpos_[b][c];
        wpos_[b][c] += lens[c];
        if (align_ > 1) {
          int r = wpos_[b][c] % align_;
          if (r) wpos_[b][c] += (align_ - r);
        }
        if (!pack_) {
          wpos_[b][c] = times_[c];
        }
      }
      seq_[b]++;
      p->seq = seq_[b];
      counter_++;
      return true;
    }
  }
  return false;
}

void TextPacking::Reset() {
  for (int b = 0; b < batch_; b++) {
    seq_[b] = 0;
    for (int c = 0; c < columns_; c++) {
      wpos_[b][c] = 0;
    }
  }
  last_fit_ = 0;
  counter_ = 0;
}

}  // namespace lingvo
}  // namespace tensorflow
