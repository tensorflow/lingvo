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

#ifndef THIRD_PARTY_PY_LINGVO_CORE_OPS_TEXT_PACKING_H_
#define THIRD_PARTY_PY_LINGVO_CORE_OPS_TEXT_PACKING_H_

#include <vector>

namespace tensorflow {
namespace lingvo {

// Text packing.
//
// It takes tuples of variable-length sequences and packs into batches of
// size [batch, time_i] for different time_i per each column.
//
// For example this picture describes packing five 2-tuples of sequences
// into 2-tuple of matrices (batch, time) with batch=2, time=10, and align=2:
//
//                column 0                 column1
// batch 0 [ a a a - b b b b b - ]  [ A A A A B B B - - -]
// batch 1 [ c c c c c - d - e - ]  [ C C - D - E - - - -]
//
// Each tuple is assigned PackedIndex as following:
//  (a a a, A A A A)    // p.batch=0 p.time=(0, 0) p.seq=1
//  (b b b b b, B B B)  // p.batch=0 p.time=(4, 4) p.seq=2
//  (c c c c, C C)      // p.batch=1 p.time=(0, 0) p.seq=1
//  (d, D)              // p.batch=1 p.time=(6, 4) p.seq=2
//  (e, E)              // p.batch=1 p.time=(8, 6) p.seq=3
class TextPacking {
 public:
  // columns: number of columns
  // batch: batch dimension
  // times: time dimension, one element per each column.
  // align: align sequence start position modulo n
  // pack: set to false to disable packing
  // spread_first_n: The first n added sequences will be assigned to the first n
  //     rows. Note that n is at most `batch`. If a value larger than `batch` is
  //     provided, `batch` will be used for n instead.
  // use_last_fit: Whether to enable an optimization where Add() starts its
  //     search from the row of the previous successful Add(). When disabled (by
  //     default), we always start from 0, resulting in O(N^2) in batch size.
  TextPacking(int columns, int batch, std::vector<int> times, int align,
              bool pack, int spread_first_n, bool use_last_fit = false);

  // Same as above, except that all columns share the same `time` and
  // `spread_first_n` is set to 0.
  TextPacking(int columns, int batch, int time, int align, bool pack)
      : TextPacking(columns, batch, std::vector<int>(columns, time), align,
                    pack, /*spread_first_n=*/0, /*use_last_fit=*/false) {}

  // Describes the location of a packed item in the batch.
  struct PackingIndex {
    // The (row) index of this item in the packed batch.
    int batch;

    // A vector of size `columns`, one per column, indicating the starting
    // positions of this item.
    std::vector<int> time;

    // Sequence index: a one-based index indicating how many packed items come
    // before this item on the same row.
    int seq;
  };

  // Adds a new item with sequence lengths 'lens' to the batch.
  // Returns false if the item does not fit.
  bool Add(const std::vector<int>& lens, PackingIndex* p);

  // Resets internal state, forgetting any previously packed items.
  void Reset();

  // Current writing position at batch index 'b' column 'c'.
  int wpos(int b, int c) const { return wpos_[b][c]; }

 private:
  const int columns_;
  const int batch_;
  const std::vector<int> times_;
  const int align_;
  const bool pack_;
  const int spread_first_n_;
  const bool use_last_fit_;

  // Current write position in each column in each row in batch.
  std::vector<std::vector<int>> wpos_;
  // Current sequence index in each row in batch.
  std::vector<int> seq_;

  // Row index of the last successfully packed segment.
  int last_fit_;

  // How many sequences have been added.
  int counter_;
};

}  // namespace lingvo
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_LINGVO_CORE_OPS_TEXT_PACKING_H_
