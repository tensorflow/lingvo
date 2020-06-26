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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lingvo {
namespace {

const unsigned int kOpRandomSeed = 16;

int FindBucket(const float val, const float min_val,
               const float interval_size) {
  return static_cast<int>(std::floor((val - min_val) / interval_size));
}

class PointToGridOp : public OpKernel {
 public:
  explicit PointToGridOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("num_points_per_cell", &num_points_per_cell_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("x_intervals", &x_intervals_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("y_intervals", &y_intervals_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("z_intervals", &z_intervals_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("x_range", &x_range_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("y_range", &y_range_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("z_range", &z_range_));

    OP_REQUIRES(ctx, x_intervals_ > 0 && y_intervals_ > 0 && z_intervals_ > 0,
                errors::InvalidArgument("intervals must be positive."));
    OP_REQUIRES(
        ctx,
        x_range_.size() == 2 && y_range_.size() == 2 && z_range_.size() == 2,
        errors::InvalidArgument("intervals must be tuple or list of two."));
    OP_REQUIRES(
        ctx,
        x_range_[0] < x_range_[1] && y_range_[0] < y_range_[1] &&
            z_range_[0] < z_range_[1],
        errors::InvalidArgument(
            "intervals must have lower bounds smaller than upper bounds."));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_points = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(input_points.shape()),
                errors::InvalidArgument("points must be a matrix, but get ",
                                        input_points.shape().DebugString()));
    OP_REQUIRES(ctx, input_points.dim_size(1) >= 3,
                errors::InvalidArgument(
                    "points must have at least 3 on 2nd dimension."));

    const float xmin_ = x_range_[0];
    const float xmax_ = x_range_[1];
    const float ymin_ = y_range_[0];
    const float ymax_ = y_range_[1];
    const float zmin_ = z_range_[0];
    const float zmax_ = z_range_[1];
    const float x_interval_size = (xmax_ - xmin_) / x_intervals_;
    const float y_interval_size = (ymax_ - ymin_) / y_intervals_;
    const float z_interval_size = (zmax_ - zmin_) / z_intervals_;

    // Create a bucket structure where the first dimension is the bucket id and
    // the second vector contains the indices of the points that fall into the
    // bucket.
    //
    // We linearize the indexing so that each bucket coordinate maps to a single
    // offset.
    std::vector<std::vector<int>> buckets_vec;

    // The number of buckets is the product of all the intervals.
    buckets_vec.resize(x_intervals_ * y_intervals_ * z_intervals_);

    const auto& t_input_points = input_points.matrix<float>();
    const int n_input_points = input_points.dim_size(0);
    const int n_input_features = input_points.dim_size(1);

    std::vector<int> visit_order(n_input_points);
    std::iota(visit_order.begin(), visit_order.end(), 0);
    std::shuffle(visit_order.begin(), visit_order.end(),
                 std::default_random_engine(kOpRandomSeed));

    for (const int i : visit_order) {
      int bucket_x = FindBucket(t_input_points(i, 0), xmin_, x_interval_size);
      int bucket_y = FindBucket(t_input_points(i, 1), ymin_, y_interval_size);
      int bucket_z = FindBucket(t_input_points(i, 2), zmin_, z_interval_size);
      if (bucket_x >= 0 && bucket_x < x_intervals_ && bucket_y >= 0 &&
          bucket_y < y_intervals_ && bucket_z >= 0 && bucket_z < z_intervals_) {
        // Compute the linearized bucket offset.
        auto bucket_id = BucketId(bucket_x, bucket_y, bucket_z);
        buckets_vec[bucket_id].push_back(i);
      }
    }

    Tensor* output_points = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(
                       "output_points",
                       TensorShape({x_intervals_, y_intervals_, z_intervals_,
                                    num_points_per_cell_, n_input_features}),
                       &output_points));
    Tensor* grid_centers = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
                 "grid_centers",
                 TensorShape({x_intervals_, y_intervals_, z_intervals_, 3}),
                 &grid_centers));
    Tensor* num_points = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(
                       "num_points",
                       TensorShape({x_intervals_, y_intervals_, z_intervals_}),
                       &num_points));

    auto t_output_points = output_points->tensor<float, 5>();
    auto t_grid_centers = grid_centers->tensor<float, 4>();
    auto t_num_points = num_points->tensor<int32, 3>();

    // Padded points will be set to 0. Users can compute the mean by taking
    // the sum and dividing by effective number of points.
    t_output_points.setConstant(0);

    for (int bx = 0; bx < x_intervals_; ++bx) {
      for (int by = 0; by < y_intervals_; ++by) {
        for (int bz = 0; bz < z_intervals_; ++bz) {
          const float cell_x_center = xmin_ + (bx + 0.5) * x_interval_size;
          const float cell_y_center = ymin_ + (by + 0.5) * y_interval_size;
          const float cell_z_center = zmin_ + (bz + 0.5) * z_interval_size;
          t_grid_centers(bx, by, bz, 0) = cell_x_center;
          t_grid_centers(bx, by, bz, 1) = cell_y_center;
          t_grid_centers(bx, by, bz, 2) = cell_z_center;

          // Compute the bucket and the number of points that mapped to that
          // bucket.
          auto bucket_id = BucketId(bx, by, bz);
          const std::vector<int>& bucket = buckets_vec[bucket_id];
          const int effective_num =
              std::min(num_points_per_cell_, static_cast<int>(bucket.size()));
          t_num_points(bx, by, bz) = effective_num;

          // Add the points in bucket to the output.
          for (int i = 0; i < effective_num; ++i) {
            t_output_points.chip<0>(bx).chip<0>(by).chip<0>(bz).chip<0>(i) =
                t_input_points.chip<0>(bucket[i]);
          }
        }
      }
    }
  }

 private:
  // Linearizes the bucket index.
  int BucketId(int bucket_x, int bucket_y, int bucket_z) {
    return (bucket_z + (bucket_y * z_intervals_) +
            (bucket_x * y_intervals_ * z_intervals_));
  }

  int num_points_per_cell_;
  int x_intervals_;
  int y_intervals_;
  int z_intervals_;
  std::vector<float> x_range_;
  std::vector<float> y_range_;
  std::vector<float> z_range_;
};

REGISTER_KERNEL_BUILDER(Name("PointToGrid").Device(DEVICE_CPU), PointToGridOp);

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
