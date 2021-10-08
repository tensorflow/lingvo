/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "lingvo/core/ops/text_packing.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace lingvo {

namespace {

template <int N>
using DSize = Eigen::DSizes<Eigen::DenseIndex, N>;

struct PackSequencesOutputs {
  Tensor* src_segment_ids = nullptr;
  Tensor* src_segment_pos = nullptr;
  Tensor* src_indices_in_input = nullptr;
  Tensor* tgt_segment_ids = nullptr;
  Tensor* tgt_segment_pos = nullptr;
  Tensor* tgt_indices_in_input = nullptr;
};

// The record of packing one input sequence, i.e. a (src, tgt) pair.
struct PackRecord {
  int index_in_input;
  TextPacking::PackingIndex packing;
};
}  // namespace

// An op that outputs a packing pattern based on actual sequence lengths.
class PackSequencesOp : public OpKernel {
 public:
  explicit PackSequencesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    int64 seed;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("packed_batch_size", &packed_batch_size_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("packed_src_seq_len", &packed_src_seq_len_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("packed_tgt_seq_len", &packed_tgt_seq_len_));
    if (seed == 0) {
      // If seed is unspecified, use completely random seed.
      std::random_device device("/dev/urandom");
      seed = std::mt19937_64(device())();
    }
    rnd_.seed(seed);
  }

  ~PackSequencesOp() override {}

  void Compute(OpKernelContext* ctx) override {
    ValidateInputs(ctx);
    if (!ctx->status().ok()) {
      return;
    }

    const auto input_num = ctx->input(0).vec<int32>().size();
    std::vector<PackRecord> pack_records;
    pack_records.reserve(input_num);
    const int output_num = PackEntireInputs(ctx, &pack_records);

    bool dropping_inputs;
    absl::flat_hash_map<int, int> new_indices;
    PackSequencesOutputs outputs;
    if (packed_batch_size_ == 0) {
      AllocateOutputs(ctx, &outputs, output_num);
      dropping_inputs = false;
    } else {
      AllocateOutputs(ctx, &outputs, packed_batch_size_);
      dropping_inputs = DropPackedRows(ctx, output_num, &new_indices);
    }
    WriteOutputs(ctx, pack_records, dropping_inputs, &new_indices, &outputs);
  }

 private:
  // Validates the shapes and types of inputs.
  void ValidateInputs(OpKernelContext* ctx);

  // Allocates (and zero initializes) all outputs.
  void AllocateOutputs(OpKernelContext* ctx, PackSequencesOutputs* outputs,
                       const int32 packed_batch_size);

  // Pack entire inputs. Returns the number of rows needed to pack all of input
  // sequences. Also outputs the packing records.
  int PackEntireInputs(OpKernelContext* ctx,
                       std::vector<PackRecord>* pack_records);

  // Returns whether we need to drop packed rows, and if so, returns a mapping
  // from the original output row index to its new index after dropping.
  bool DropPackedRows(OpKernelContext* ctx, int num_rows,
                      absl::flat_hash_map<int, int>* new_indices);

  // Writes the packing results to the output tensors.
  void WriteOutputs(OpKernelContext* ctx,
                    const std::vector<PackRecord>& pack_records,
                    bool dropping_inputs,
                    const absl::flat_hash_map<int, int>* new_indices,
                    PackSequencesOutputs* outputs);

  int packed_batch_size_;
  int packed_src_seq_len_;
  int packed_tgt_seq_len_;
  mutable absl::Mutex mu_;
  // Used for randomizing the dropping of input rows when needed.
  std::mt19937 rnd_ ABSL_GUARDED_BY(mu_);
  int64 total_src_tokens_ ABSL_GUARDED_BY(mu_) = 0;
  int64 total_tgt_tokens_ ABSL_GUARDED_BY(mu_) = 0;
  int64 total_examples_ ABSL_GUARDED_BY(mu_) = 0;
  int64 total_packed_records_ ABSL_GUARDED_BY(mu_) = 0;
  int64 total_dropped_packed_records_ ABSL_GUARDED_BY(mu_) = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(PackSequencesOp);
};

void PackSequencesOp::ValidateInputs(OpKernelContext* ctx) {
  const Tensor& src_actual_seq_len = ctx->input(0);
  OP_REQUIRES(ctx,
              TensorShapeUtils::IsVector(src_actual_seq_len.shape()) &&
                  (src_actual_seq_len.dtype() == DataType::DT_INT32),
              errors::InvalidArgument(
                  "src_actual_seq_len must be a vector of int32, got: ",
                  src_actual_seq_len.DebugString()));
  const Tensor& tgt_actual_seq_len = ctx->input(1);
  OP_REQUIRES(ctx,
              TensorShapeUtils::IsVector(tgt_actual_seq_len.shape()) &&
                  (tgt_actual_seq_len.dtype() == DataType::DT_INT32),
              errors::InvalidArgument(
                  "tgt_actual_seq_len must be a vector of int32, got: ",
                  tgt_actual_seq_len.DebugString()));
  OP_REQUIRES(ctx, src_actual_seq_len.shape() == tgt_actual_seq_len.shape(),
              errors::InvalidArgument(
                  "src_actual_seq_len must be the same shape as "
                  "tgt_actual_seq_len, got: src shape ",
                  src_actual_seq_len.shape().DebugString(), " vs. tgt shape ",
                  tgt_actual_seq_len.shape().DebugString()));
}

void PackSequencesOp::AllocateOutputs(OpKernelContext* ctx,
                                      PackSequencesOutputs* outputs,
                                      const int32 packed_batch_size) {
  TensorShape packed_src_shape({packed_batch_size, packed_src_seq_len_});
  TensorShape packed_tgt_shape({packed_batch_size, packed_tgt_seq_len_});

  int output_id = 0;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(output_id++, packed_src_shape,
                                           &outputs->src_segment_ids));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(output_id++, packed_src_shape,
                                           &outputs->src_segment_pos));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(output_id++, packed_src_shape,
                                           &outputs->src_indices_in_input));
  outputs->src_segment_ids->matrix<int32>().setZero();
  outputs->src_segment_pos->matrix<int32>().setZero();
  outputs->src_indices_in_input->matrix<int32>().setZero();

  OP_REQUIRES_OK(ctx, ctx->allocate_output(output_id++, packed_tgt_shape,
                                           &outputs->tgt_segment_ids));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(output_id++, packed_tgt_shape,
                                           &outputs->tgt_segment_pos));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(output_id++, packed_tgt_shape,
                                           &outputs->tgt_indices_in_input));
  outputs->tgt_segment_ids->matrix<int32>().setZero();
  outputs->tgt_segment_pos->matrix<int32>().setZero();
  outputs->tgt_indices_in_input->matrix<int32>().setZero();
}

int PackSequencesOp::PackEntireInputs(OpKernelContext* ctx,
                                      std::vector<PackRecord>* pack_records) {
  const auto& src_actual_seq_len = ctx->input(0).vec<int32>();
  const auto& tgt_actual_seq_len = ctx->input(1).vec<int32>();
  const auto input_num = src_actual_seq_len.size();

  // We ask for a sufficiently large output batch size to pack all input
  // sequences in its entirety. We drop input sequences if needed afterward,
  // We also ensure that the first `packed_batch_size` rows are never empty
  // by packing into them first.
  TextPacking packing(/*columns=*/2, /*batch=*/input_num,
                      {packed_src_seq_len_, packed_tgt_seq_len_},
                      /*align=*/1,
                      /*pack=*/true, /*spread_first_n=*/packed_batch_size_);
  int max_output_batch_idx = 0;
  int64 total_src_seq_len = 0, total_tgt_seq_len = 0;
  for (int i = 0; i < input_num; ++i) {
    int src_seq_len = src_actual_seq_len(i);
    int tgt_seq_len = tgt_actual_seq_len(i);
    if (src_seq_len < 1 || tgt_seq_len < 1) {
      LOG_EVERY_N(WARNING, 500)
          << "Input sequence with lengths (src=" << src_seq_len
          << ", tgt=" << tgt_seq_len
          << ") dropped as only positive actual lengths are allowed.";
      continue;
    }
    PackRecord p;
    p.index_in_input = i;
    if (!packing.Add({src_seq_len, tgt_seq_len}, &p.packing)) {
      // This only happens when the actual sequence length of the input is
      // larger than the packed sequence length specified on the output.
      LOG_EVERY_N(WARNING, 500)
          << "Input sequence with lengths (src=" << src_seq_len
          << ", tgt=" << tgt_seq_len
          << ") dropped as its lengths exceed the output sequence length.";
      continue;
    }
    max_output_batch_idx = std::max(p.packing.batch, max_output_batch_idx);
    pack_records->push_back(p);
    total_src_seq_len += src_seq_len;
    total_tgt_seq_len += tgt_seq_len;
  }
  int num_packed_records = max_output_batch_idx + 1;
  {
    absl::MutexLock l(&mu_);
    total_src_tokens_ += total_src_seq_len;
    total_tgt_tokens_ += total_tgt_seq_len;
    total_examples_ += input_num;
    total_packed_records_ += num_packed_records;
    if (packed_batch_size_ > 0 && num_packed_records > packed_batch_size_) {
      total_dropped_packed_records_ += num_packed_records - packed_batch_size_;
    }
    LOG_EVERY_N_SEC(INFO, 60)
        << "Total packed " << total_examples_ << " examples into "
        << total_packed_records_ << " rows, dropped "
        << total_dropped_packed_records_
        << " rows. Average tokens per example: src="
        << 1. * total_src_tokens_ / total_examples_ << ", tgt="
        << 1. * total_tgt_tokens_ / total_examples_
        << ". Post-packing space utilization: src="
        << 1. * total_src_tokens_ / total_packed_records_ / packed_src_seq_len_
        << ", tgt="
        << 1. * total_tgt_tokens_ / total_packed_records_ / packed_tgt_seq_len_;
  }
  return num_packed_records;
}

bool PackSequencesOp::DropPackedRows(
    OpKernelContext* ctx, int num_rows,
    absl::flat_hash_map<int, int>* new_indices) {
  if (num_rows <= packed_batch_size_) {
    return false;
  }

  // Simple reservoir sampling to pick `packed_batch_size` items out of
  // `num_rows`:
  // https://en.wikipedia.org/wiki/Reservoir_sampling#Simple_algorithm
  std::vector<int> indices_kept(packed_batch_size_);
  for (int i = 1; i < packed_batch_size_; ++i) {
    indices_kept[i] = i;
  }
  for (int i = packed_batch_size_; i < num_rows; ++i) {
    std::uniform_int_distribution<> distribution(0, i);
    int j;  // Uniformly picked on [0, i].
    {
      absl::MutexLock l(&mu_);
      j = distribution(rnd_);
    }
    if (j < packed_batch_size_) {
      indices_kept[j] = i;
    }
  }
  for (int i = 0; i < packed_batch_size_; ++i) {
    new_indices->insert({indices_kept[i], i});
  }
  return true;
}

void PackSequencesOp::WriteOutputs(
    OpKernelContext* ctx, const std::vector<PackRecord>& pack_records,
    bool dropping_inputs, const absl::flat_hash_map<int, int>* new_indices,
    PackSequencesOutputs* outputs) {
  const auto& src_actual_seq_len = ctx->input(0).vec<int32>();
  const auto& tgt_actual_seq_len = ctx->input(1).vec<int32>();

  auto src_segment_ids = outputs->src_segment_ids->matrix<int32>();
  auto src_segment_pos = outputs->src_segment_pos->matrix<int32>();
  auto src_indices_in_input = outputs->src_indices_in_input->matrix<int32>();
  auto tgt_segment_ids = outputs->tgt_segment_ids->matrix<int32>();
  auto tgt_segment_pos = outputs->tgt_segment_pos->matrix<int32>();
  auto tgt_indices_in_input = outputs->tgt_indices_in_input->matrix<int32>();

  for (const auto& p : pack_records) {
    int output_idx = p.packing.batch;
    if (dropping_inputs) {
      auto new_idx_iter = new_indices->find(output_idx);
      if (new_idx_iter == new_indices->end()) {
        // This row is being dropped.
        continue;
      }
      output_idx = new_idx_iter->second;
    }
    const int src_seq_idx = p.packing.time[0];
    for (int i = 0; i < src_actual_seq_len(p.index_in_input); ++i) {
      src_segment_ids(output_idx, src_seq_idx + i) = p.packing.seq;
      src_segment_pos(output_idx, src_seq_idx + i) = i;
      src_indices_in_input(output_idx, src_seq_idx + i) = p.index_in_input;
    }
    const int tgt_seq_idx = p.packing.time[1];
    for (int i = 0; i < tgt_actual_seq_len(p.index_in_input); ++i) {
      tgt_segment_ids(output_idx, tgt_seq_idx + i) = p.packing.seq;
      tgt_segment_pos(output_idx, tgt_seq_idx + i) = i;
      tgt_indices_in_input(output_idx, tgt_seq_idx + i) = p.index_in_input;
    }
  }
}

REGISTER_KERNEL_BUILDER(Name("PackSequences").Device(DEVICE_CPU),
                        PackSequencesOp);

// Pack a sequence into a smaller batch given a max length constraint.
// The output batch size is dynamic. No examples are dropped.
class PackSingleSequenceOp : public OpKernel {
 public:
  explicit PackSingleSequenceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_packed_length", &max_packed_length_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("require_sequential_order",
                                     &require_sequential_order_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_lengths_t = ctx->input(0);
    const auto& input_lengths = input_lengths_t.vec<int32>();
    const int batch_size = input_lengths.dimension(0);

    for (int i = 0; i < batch_size; ++i) {
      OP_REQUIRES(
          ctx,
          tensorflow::FastBoundsCheck(input_lengths(i), max_packed_length_ + 1),
          errors::InvalidArgument(
              "Input length at index ", i, " is too long: ", input_lengths(i),
              " vs max_packed_length ", max_packed_length_));
    }

    // The index of the output to write to.
    std::vector<int32> output_index;
    int packed_batch_size = ComputeOutputIndex(input_lengths_t, &output_index);

    Tensor* segment_ids_t = nullptr;
    Tensor* indices_in_input_t = nullptr;

    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, {packed_batch_size, max_packed_length_},
                                  &segment_ids_t));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, {packed_batch_size, max_packed_length_},
                                  &indices_in_input_t));

    auto segment_ids = segment_ids_t->matrix<int32>();
    segment_ids.setZero();
    auto indices_in_input = indices_in_input_t->matrix<int32>();
    indices_in_input.setConstant(-1);

    // Current lengths for each packed sequence.
    std::vector<int32> current_lengths(packed_batch_size, 0);
    // Current segment id for each packed sequence.
    std::vector<int32> current_segment_id(packed_batch_size, 1);

    for (int i = 0; i < batch_size; ++i) {
      int idx = output_index[i];
      segment_ids
          .slice(DSize<2>{idx, current_lengths[idx]},
                 DSize<2>{1, input_lengths(i)})
          .setConstant(current_segment_id[idx]);
      indices_in_input
          .slice(DSize<2>{idx, current_lengths[idx]},
                 DSize<2>{1, input_lengths(i)})
          .setConstant(i);
      current_lengths[idx] += input_lengths(i);
      current_segment_id[idx]++;
    }
  }

 private:
  int max_packed_length_;
  bool require_sequential_order_;

  // Populates output_index with the index that each input example should be
  // placed into, and returns the total number of bins used.
  int ComputeOutputIndex(const Tensor& input_lengths_t,
                         std::vector<int32>* output_index);

  TF_DISALLOW_COPY_AND_ASSIGN(PackSingleSequenceOp);
};

int PackSingleSequenceOp::ComputeOutputIndex(const Tensor& input_lengths_t,
                                             std::vector<int32>* output_index) {
  const auto& input_lengths = input_lengths_t.vec<int32>();
  const int batch_size = input_lengths.dimension(0);
  output_index->resize(batch_size);

  // Current lengths for each packed sequence.
  std::vector<int32> current_lengths;

  if (require_sequential_order_) {
    // Append each input in sequence to the last element.
    for (int i = 0; i < batch_size; ++i) {
      if (current_lengths.empty() ||
          current_lengths.back() + input_lengths(i) > max_packed_length_) {
        // Start a new bin.
        current_lengths.push_back(0);
      }
      // Add this input to the last bin.
      output_index->at(i) = current_lengths.size() - 1;
      current_lengths.back() += input_lengths(i);
    }
  } else {
    std::vector<int> idx(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      idx[i] = i;
    }
    // Sort by lengths descending.
    std::sort(idx.begin(), idx.end(), [&](int i, int j) {
      return std::make_pair(input_lengths(i), -i) >
             std::make_pair(input_lengths(j), -j);
    });
    // Best Fit Decreasing as it is easiest to implement in nlogn.
    // Maintain a Binary Search Tree of (remaining_space, bin_id).
    std::multiset<std::pair<int32, int32>> lookup;
    for (int i = 0; i < batch_size; ++i) {
      // First element with remaining_space >= input_length.
      auto it = lookup.lower_bound(std::make_pair(input_lengths(idx[i]), -1));
      if (it == lookup.end()) {
        // Start a new bin.
        current_lengths.push_back(0);
        it = lookup.insert(
            std::make_pair(max_packed_length_, current_lengths.size() - 1));
      }

      // Add to the bin.
      int bin_id = it->second;
      output_index->at(idx[i]) = bin_id;

      lookup.erase(it);
      current_lengths[bin_id] += input_lengths(idx[i]);
      lookup.insert(
          std::make_pair(max_packed_length_ - current_lengths[bin_id], bin_id));
    }
  }
  return current_lengths.size();
}

REGISTER_KERNEL_BUILDER(Name("PackSingleSequence").Device(DEVICE_CPU),
                        PackSingleSequenceOp);

// An op that applies a packing pattern on an input data.
template <typename T>
class ApplyPackingOp : public OpKernel {
 public:
  explicit ApplyPackingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  ~ApplyPackingOp() override {}

  void Compute(OpKernelContext* ctx) override {
    if (!ctx->status().ok()) {
      return;
    }
    const Tensor& input = ctx->input(0);
    if (TensorShapeUtils::IsMatrixOrHigher(input.shape())) {
      // Input is a matrix. We apply the packing pattern by returning a
      // denser, shorter matrix where the elements are re-arranged.
      Tensor* output = nullptr;
      // Allocates output and initializes with padding.
      const Tensor& segment_ids = ctx->input(2);
      auto output_dim_sizes = input.shape().dim_sizes();
      output_dim_sizes[0] = segment_ids.shape().dim_size(0);
      output_dim_sizes[1] = segment_ids.shape().dim_size(1);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0, TensorShape(output_dim_sizes), &output));
      const T padding = ctx->input(1).scalar<T>()();
      output->flat<T>().setConstant(padding);
      ApplyMatrix(ctx, output);
      return;
    }
    // Input is a vector. We apply the packing patterning by returning a
    // shorter vector where the elements are summed.
    Tensor* output = nullptr;
    TensorShape output_shape({ctx->input(3).shape().dim_size(0)});
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    ApplyVector(ctx, output);
  }

 private:
  void ApplyMatrix(OpKernelContext* ctx, Tensor* output) {
    if (ctx->input(0).NumElements() == 0) return;
    const auto& input = ctx->input(0).flat_outer_dims<T, 3>();
    const auto input_rows = input.dimension(0);
    const auto input_columns = input.dimension(1);
    const auto input_dim = input.dimension(2);
    const auto& segment_ids = ctx->input(2).matrix<int32>();
    const auto& indices_in_input = ctx->input(3).matrix<int32>();
    auto output_3d = output->flat_outer_dims<T, 3>();
    const int64 rows = output->dim_size(0);
    const int64 columns = output->dim_size(1);
    // The CPU cost per each row is linear in the number of elements in that
    // row
    // (`columns`). We need to read segment_ids to discern each segment, and
    // copy from input to output, plus a few extra cycles (e.g. checking
    // bounds). The constant is guesstimated to be 4.
    const int64 cost_per_unit = columns << 2;
    ctx->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        rows, cost_per_unit, [&](int64 begin, int64 end) {
          for (int i = begin; i < end; ++i) {
            for (int j = 0; j < columns; ++j) {
              if (segment_ids(i, j) <= 0) {
                // We do not skip the rest of the row because we cannot assume
                // 0 segment ids only occur at end of row. For example, it
                // might occur due to alignment requirements during packing.
                continue;
              }
              const int start = j;
              for (; j + 1 < columns &&
                     segment_ids(i, j) == segment_ids(i, j + 1);
                   ++j) {
              }
              // [start, j] have the same, positive segment_ids.
              const int actual_seq_len = j - start + 1;
              // At output position (i, start), we need to copy over
              // `actual_seq_len` T values from the input.
              const int index_in_input = indices_in_input(i, start);
              OP_REQUIRES(
                  ctx,
                  tensorflow::FastBoundsCheck(index_in_input, input_rows) &&
                      tensorflow::FastBoundsCheck(actual_seq_len,
                                                  input_columns + 1),
                  errors::InvalidArgument(
                      "out of bound found packing at (", i, ", ", start,
                      ") for input index ", index_in_input, " with length ",
                      actual_seq_len, " where input shape is ",
                      ctx->input(0).shape().DebugString()));
              output_3d.slice(DSize<3>{i, start, 0},
                              DSize<3>{1, actual_seq_len, input_dim}) =
                  input.slice(DSize<3>{index_in_input, 0, 0},
                              DSize<3>{1, actual_seq_len, input_dim});
            }
          }
        });
  }

  void ApplyVector(OpKernelContext* ctx, Tensor* output) {
    const auto& input = ctx->input(0).vec<T>();
    const auto num_input_rows = ctx->input(0).dim_size(0);
    const auto& segment_ids = ctx->input(2).matrix<int32>();
    const auto& indices_in_input = ctx->input(3).matrix<int32>();
    auto output_vec = output->vec<T>();
    for (int i = 0; i < output->dim_size(0); ++i) {
      // input_rows condenses row i of indices_in_input, e.g. from
      // [0, 0, 0, 3, 3, 4, 4, 4, 4, 0, 0] to [0, 3, 4].
      std::vector<int64> input_rows;
      for (int j = 0; j < ctx->input(3).dim_size(1); ++j) {
        auto row = indices_in_input(i, j);
        if (segment_ids(i, j) &&
            (input_rows.empty() || input_rows.back() != row)) {
          OP_REQUIRES(ctx, tensorflow::FastBoundsCheck(row, num_input_rows),
                      errors::InvalidArgument(
                          "out of bound found packing at (", i, ", ", j,
                          ") for input index ", row, " where input shape is ",
                          ctx->input(0).shape().DebugString()));
          input_rows.push_back(row);
        }
      }
      std::vector<T> elements;
      elements.reserve(input_rows.size());
      for (auto row : input_rows) {
        elements.push_back(input(row));
      }
      // Output on row i is the reduce_sum of the elements on that row.
      output_vec(i) =
          std::accumulate(elements.begin(), elements.end(), static_cast<T>(0));
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ApplyPackingOp);
};

template <>
class ApplyPackingOp<::tensorflow::tstring> : public OpKernel {
 public:
  explicit ApplyPackingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  ~ApplyPackingOp() override {}

  void Compute(OpKernelContext* ctx) override {
    ValidateInputs(ctx);
    if (!ctx->status().ok()) {
      return;
    }
    Tensor* output = nullptr;
    TensorShape output_shape({ctx->input(3).shape().dim_size(0)});
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    Apply(ctx, output);
  }

 private:
  // Validates the shapes and types of inputs.
  void ValidateInputs(OpKernelContext* ctx) {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(input.shape()),
        errors::InvalidArgument("input must be a vector, got input shape: ",
                                input.shape().DebugString()));

    const Tensor& padding = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(padding.shape()),
        errors::InvalidArgument("padding must be a scalar, got padding shape: ",
                                padding.shape().DebugString()));

    const Tensor& segment_ids = ctx->input(2);
    const Tensor& indices_in_input = ctx->input(3);
    OP_REQUIRES(
        ctx,
        segment_ids.shape() == indices_in_input.shape() &&
            TensorShapeUtils::IsMatrix(segment_ids.shape()),
        errors::InvalidArgument("segment_ids and indices_in_input must be "
                                "matrices of the same shape, got: ",
                                segment_ids.shape().DebugString(), " vs. ",
                                indices_in_input.shape().DebugString()));
  }

  void Apply(OpKernelContext* ctx, Tensor* output) {
    const auto& input = ctx->input(0).vec<::tensorflow::tstring>();
    const auto num_input_rows = ctx->input(0).dim_size(0);
    const auto& segment_ids = ctx->input(2).matrix<int32>();
    const auto& indices_in_input = ctx->input(3).matrix<int32>();
    auto output_vec = output->vec<::tensorflow::tstring>();
    const auto& sep = ctx->input(1).scalar<::tensorflow::tstring>()();
    for (int i = 0; i < output->dim_size(0); ++i) {
      // input_rows condenses row i of indices_in_input, e.g. from
      // [0, 0, 0, 3, 3, 4, 4, 4, 4, 0, 0] to [0, 3, 4].
      std::vector<int64> input_rows;
      for (int j = 0; j < ctx->input(3).dim_size(1); ++j) {
        auto row = indices_in_input(i, j);
        if (segment_ids(i, j) &&
            (input_rows.empty() || input_rows.back() != row)) {
          OP_REQUIRES(ctx, tensorflow::FastBoundsCheck(row, num_input_rows),
                      errors::InvalidArgument(
                          "out of bound found packing at (", i, ", ", j,
                          ") for input index ", row, " where input shape is ",
                          ctx->input(0).shape().DebugString()));
          input_rows.push_back(row);
        }
      }
      std::vector<absl::string_view> strs;
      strs.reserve(input_rows.size());
      for (auto row : input_rows) {
        strs.push_back(input(row));
      }
      // Output on row i is the joined input strings for that row.
      output_vec(i) = absl::StrJoin(strs, sep);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ApplyPackingOp);
};

#define REGISTER(TYPE)                                                   \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("ApplyPacking").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ApplyPackingOp<TYPE>);

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
TF_CALL_bfloat16(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_uint32(REGISTER);
TF_CALL_uint64(REGISTER);
TF_CALL_bool(REGISTER);
TF_CALL_string(REGISTER);

#undef REGISTER

}  // namespace lingvo
}  // namespace tensorflow
