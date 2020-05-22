/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "lingvo/tasks/car/ops/ps_utils.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace lingvo {
namespace car {

namespace {

int BucketId(int bucket_x, int bucket_y, int bucket_z, int y_intervals,
             int z_intervals) {
  return (bucket_z + (bucket_y * z_intervals) +
          (bucket_x * y_intervals * z_intervals));
}

int FindBucket(const float val, const float min_val,
               const float interval_size) {
  return static_cast<int>(std::floor((val - min_val) / interval_size));
}

// Given a sequence of identifiers (Add), returns k elements uniformly sampled
// from the sequence.
class UniformSampler {
 public:
  explicit UniformSampler(int k, uint64 seed) : k_(k), rnd_(seed) {
    CHECK_GT(k, 0);
    ids_.reserve(k_);
  }

  void Add(int id, float unused_score) {
    ++num_;
    if (ids_.size() < k_) {
      ids_.push_back(Item{id});
    } else {
      const int64 rnd = rnd_() % num_;
      // Replace w/ prob k_ / num_ following the reservoir sampling algorithm R.
      if (rnd < k_) {
        ids_[rnd].id = id;
      }
    }
  }

  struct Item {
    int id;
  };

  const std::vector<Item>& Get() const { return ids_; }

  void Reset() {
    num_ = 0;
    ids_.clear();
  }

 private:
  const int32 k_;
  std::mt19937 rnd_;
  int32 num_ = 0;
  std::vector<Item> ids_;
};

// Given a sequence of identifiers with scores (Add), returns k elements with
// the smallest scores.
class TopKSampler {
 public:
  explicit TopKSampler(int k, uint64 unused_seed) : k_(k) { CHECK_GT(k, 0); }

  void Add(int id, float score) {
    if (selected_ && score > items_[k_ - 1].score) {
      return;
    }
    items_.push_back(Item{id, score});
    if (items_.size() > 2 * k_) {
      Shrink();
    }
  }

  struct Item {
    int id;
    float score;
  };

  const std::vector<Item>& Get() {
    if (items_.size() > k_) {
      Shrink();
    }
    return items_;
  }

  void Reset() {
    items_.clear();
    selected_ = false;
  }

 private:
  const int k_ = -1;
  bool selected_ = false;
  std::vector<Item> items_;
  struct Cmp {
    bool operator()(const Item& a, const Item& b) const {
      return a.score < b.score;
    }
  };
  Cmp cmp_;

  void Shrink() {
    std::nth_element(items_.begin(), items_.begin() + k_ - 1, items_.end(),
                     cmp_);
    items_.resize(k_);
    selected_ = true;
  }
};

// Uniform center selection among n points (w/o replacement).
class UniformSelector {
 public:
  UniformSelector(const std::vector<bool>& candidates, uint64 seed) {
    for (int i = 0; i < candidates.size(); ++i) {
      if (candidates[i]) {
        ids_.push_back(i);
      }
    }
    std::shuffle(ids_.begin(), ids_.end(), std::mt19937(seed));
  }

  int Get() {
    if (ids_.empty()) {
      return -1;
    }
    int id = ids_.back();
    ids_.resize(ids_.size() - 1);
    return id;
  }

  void Update(int j, float score) {}  // No op.

 private:
  std::vector<int> ids_;
};

// Select the point farthest from selected centers so far.
class FarthestSelector {
 public:
  FarthestSelector(std::vector<bool> candidates, uint64 seed)
      : num_(candidates.size()),
        candidates_(std::move(candidates)),
        min_dist_sq_(DT_FLOAT, {num_}),
        min_dist_sq_t_(min_dist_sq_.vec<float>()),
        farthest_(DT_INT32, {}),
        farthest_t_(farthest_.scalar<int>()) {
    CHECK_LT(kLargeDistA, kLargeDistB);
    // Initialize min_dist_sq_ so that the 1st selected point is random.
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(kLargeDistA, kLargeDistB);
    for (int i = 0; i < num_; ++i) {
      if (candidates_[i]) {
        ++num_valid_;
        min_dist_sq_t_(i) = dis(gen);
      } else {
        min_dist_sq_t_(i) = -1.0;
      }
    }
  }

  int Get() {
    if (num_sampled_ >= num_valid_) {
      return -1;
    }
    farthest_t_ = min_dist_sq_t_.argmax().template cast<int>().eval();
    ++num_sampled_;
    return farthest_t_();
  }

  // Updates j-th distance squared to all chosen centers.
  void Update(int j, float dist_sq) {
    DCHECK_LT(dist_sq, kLargeDistA);
    if (candidates_[j] && (dist_sq < min_dist_sq_t_(j))) {
      min_dist_sq_t_(j) = dist_sq;
    }
  }

 private:
  const float kLargeDistA = 1e30;
  const float kLargeDistB = kLargeDistA * 1.1;

  const int num_;                 // # of points.
  int num_valid_ = 0;             // # of points candidates_[i] is True.
  int num_sampled_ = 0;           // # of Get() has been called.
  std::vector<bool> candidates_;  // i-th can be selected iff candidates_[i].
  Tensor min_dist_sq_;
  decltype(min_dist_sq_.vec<float>()) min_dist_sq_t_;
  Tensor farthest_;
  decltype(farthest_.scalar<int>()) farthest_t_;
};

class RNG {
 public:
  RNG() : rng_(std::random_device("/dev/urandom")()) {}

  uint64 Get() {
    absl::MutexLock l(&mu_);
    return rng_();
  }

 private:
  absl::Mutex mu_;
  std::mt19937_64 rng_;
};

}  // namespace

uint64 PSUtils::Seed() const {
  if (opts_.random_seed != -1) {
    return opts_.random_seed;
  }
  static RNG* rng = new RNG();
  return rng->Get();
}

template <typename T>
T Square(T x) {
  return x * x;
}

template <typename Selector, typename Sampler>
PSUtils::Result PSUtils::DoSampling(const Tensor& points,
                                    const Tensor& points_padding,
                                    const int32 num_seeded_points) const {
  // Points must be of rank 3, and padding must be a matrix.
  DCHECK_EQ(points.dims(), 3);
  DCHECK_EQ(points_padding.dims(), 2);
  // 3D points.
  DCHECK_EQ(points.dim_size(2), 3);
  DCHECK_EQ(points.dim_size(0), points_padding.dim_size(0));
  DCHECK_EQ(points.dim_size(1), points_padding.dim_size(1));

  auto points_t = points.tensor<float, 3>();
  auto points_padding_t = points_padding.matrix<float>();
  const int64 batch_size = points.dim_size(0);
  const int64 num_points = points.dim_size(1);

  Result result;
  result.center = Tensor(DT_INT32, {batch_size, opts_.num_centers});
  result.center_padding = Tensor(DT_FLOAT, {batch_size, opts_.num_centers});
  result.indices =
      Tensor(DT_INT32, {batch_size, opts_.num_centers, opts_.num_neighbors});
  result.indices_padding =
      Tensor(DT_FLOAT, {batch_size, opts_.num_centers, opts_.num_neighbors});

  auto center_t = result.center.matrix<int32>();
  center_t.setConstant(0);
  auto center_padding_t = result.center_padding.matrix<float>();
  auto indices_t = result.indices.tensor<int32, 3>();
  indices_t.setConstant(0);
  auto padding_t = result.indices_padding.tensor<float, 3>();
  padding_t.setConstant(1.0);

  // Max distance squared as the threshold.
  const float threshold = Square(opts_.max_dist);

  // The idea behind the hash lookup is to only do neighbor / distance checks
  // for plausibly close neighbors, rather than looking at all points for each
  // center.  We do this by gridifying the points, and for each center only
  // looking at points in nearby grid cells.  This a cheap version of a more
  // sophisticated algorithm like using a KDTree or RangeTree.
  const bool use_hash_lookup =
      (opts_.neighbor_search_algorithm == PSUtils::Options::N_HASH);

  for (int cur_batch = 0; cur_batch < batch_size; ++cur_batch) {
    std::vector<bool> candidates(num_points);

    float xmin = std::numeric_limits<float>::max();
    float ymin = std::numeric_limits<float>::max();
    float zmin = std::numeric_limits<float>::max();
    float xmax = std::numeric_limits<float>::lowest();
    float ymax = std::numeric_limits<float>::lowest();
    float zmax = std::numeric_limits<float>::lowest();

    for (int i = 0; i < num_points; ++i) {
      // The first num_seeded_points are not candidates of the selector, because
      // they are always selected.
      candidates[i] =
          (i >= num_seeded_points && points_padding_t(cur_batch, i) == 0.0) &&
          (opts_.center_z_min <= points_t(cur_batch, i, 2)) &&
          (points_t(cur_batch, i, 2) <= opts_.center_z_max);

      // Find min / max points for computing grid buckets.
      if (use_hash_lookup && points_padding_t(cur_batch, i) == 0.0) {
        xmin = std::min(points_t(cur_batch, i, 0), xmin);
        xmax = std::max(points_t(cur_batch, i, 0), xmax);
        ymin = std::min(points_t(cur_batch, i, 1), ymin);
        ymax = std::max(points_t(cur_batch, i, 1), ymax);
        zmin = std::min(points_t(cur_batch, i, 2), zmin);
        zmax = std::max(points_t(cur_batch, i, 2), zmax);
      }
    }

    // Stores a mapping of bucket_id -> list of point indices.  The buckets are
    // the voxelized breakdown of the 3D space and points fall into these
    // voxels.  The length of the cube is the max_distance.
    std::vector<std::vector<int>> buckets_vec;
    std::vector<std::vector<float>> buckets_values;

    int x_intervals = 0;
    int y_intervals = 0;
    int z_intervals = 0;

    if (use_hash_lookup) {
      // Adjust boundaries to avoid edge conditions.  We use max_dist as a
      // conservative estimate.
      xmin -= opts_.max_dist;
      ymin -= opts_.max_dist;
      zmin -= opts_.max_dist;
      xmax += opts_.max_dist;
      ymax += opts_.max_dist;
      zmax += opts_.max_dist;

      x_intervals = std::ceil((xmax - xmin) / opts_.max_dist);
      y_intervals = std::ceil((ymax - ymin) / opts_.max_dist);
      z_intervals = std::ceil((zmax - zmin) / opts_.max_dist);

      // The number of buckets is the product of all the intervals.
      buckets_vec.resize(x_intervals * y_intervals * z_intervals);
      for (int i = 0; i < num_points; ++i) {
        // Compute which bucket each valid point falls into.
        //
        // A valid is a non-padded, non-seeded point.
        if (points_padding_t(cur_batch, i) == 0.0 && i >= num_seeded_points) {
          int bucket_x =
              FindBucket(points_t(cur_batch, i, 0), xmin, opts_.max_dist);
          int bucket_y =
              FindBucket(points_t(cur_batch, i, 1), ymin, opts_.max_dist);
          int bucket_z =
              FindBucket(points_t(cur_batch, i, 2), zmin, opts_.max_dist);
          if (bucket_x >= 0 && bucket_x < x_intervals && bucket_y >= 0 &&
              bucket_y < y_intervals && bucket_z >= 0 &&
              bucket_z < z_intervals) {
            // Compute the linearized bucket offset.
            auto bucket_id = BucketId(bucket_x, bucket_y, bucket_z, y_intervals,
                                      z_intervals);
            buckets_vec[bucket_id].push_back(i);
          }
        }
      }
    }

    Selector selector(candidates, Seed());
    Sampler sampler(opts_.num_neighbors, Seed());

    for (int i = 0; i < opts_.num_centers; ++i) {
      // Pick a point as i-th center.
      int k;
      if (i < num_seeded_points) {
        k = i;
      } else {
        // Pick a point as i-th center.
        k = selector.Get();
      }

      if (k < 0) {
        center_padding_t(cur_batch, i) = 1.0;
        continue;
      }
      center_padding_t(cur_batch, i) = 0.0;
      center_t(cur_batch, i) = k;

      // Goes through all *non-seeded* points. If j-th point is within a radius
      // of center, adds it to the sampler.
      sampler.Reset();

      std::vector<int> neighbor_idx;
      if (use_hash_lookup) {
        // For each center, compute the bucket it is in.
        int bucket_x =
            FindBucket(points_t(cur_batch, k, 0), xmin, opts_.max_dist);
        int bucket_y =
            FindBucket(points_t(cur_batch, k, 1), ymin, opts_.max_dist);
        int bucket_z =
            FindBucket(points_t(cur_batch, k, 2), zmin, opts_.max_dist);

        // Iterate over 3x3x3 buckets centered at [bucket_x, bucket_y, bucket_z]
        //
        // Extract the neighborhood indices from there.
        for (int bx = bucket_x - 1; bx <= bucket_x + 1; ++bx) {
          if (bx < 0 || bx >= x_intervals) continue;
          for (int by = bucket_y - 1; by <= bucket_y + 1; ++by) {
            if (by < 0 || by >= y_intervals) continue;
            for (int bz = bucket_z - 1; bz <= bucket_z + 1; ++bz) {
              if (bz < 0 || bz >= z_intervals) continue;
              auto bucket_id = BucketId(bx, by, bz, y_intervals, z_intervals);
              auto bucket_indices = buckets_vec[bucket_id];
              neighbor_idx.insert(neighbor_idx.end(), bucket_indices.begin(),
                                  bucket_indices.end());
            }
          }
        }

      } else {
        neighbor_idx.reserve(num_points);
        for (int j = num_seeded_points; j < num_points; ++j) {
          if (points_padding_t(cur_batch, j) == 0.0) {
            neighbor_idx.push_back(j);
          }
        }
      }

      // Iterate over all neighbor indices.
      for (int j : neighbor_idx) {
        auto ss_xy =
            Square(points_t(cur_batch, k, 0) - points_t(cur_batch, j, 0)) +
            Square(points_t(cur_batch, k, 1) - points_t(cur_batch, j, 1));
        auto z = points_t(cur_batch, j, 2);
        auto ss_xyz = ss_xy + Square(points_t(cur_batch, k, 2) - z);
        if (ss_xyz <= threshold) {
          sampler.Add(j, ss_xyz);
        }
        selector.Update(j, ss_xy);
      }
      auto ids = sampler.Get();
      CHECK_LE(0, ids.size());
      CHECK_LE(ids.size(), opts_.num_neighbors);
      for (int j = 0; j < ids.size(); ++j) {
        indices_t(cur_batch, i, j) = ids[j].id;
        padding_t(cur_batch, i, j) = 0.0f;
      }
    }
  }

  return result;
}

string PSUtils::Options::DebugString() const {
  // clang-format off
  return strings::Printf(
      "cmethod/#centers/zmin/zmax/nmethod/#neighbors/maxdist/seed "
      "%s/%d/%.3f/%.3f/%s/%d/%.3f/%d",
      cmethod == C_UNIFORM ? "uniform" : "farthest",
      num_centers,
      center_z_min,
      center_z_max,
      nmethod == N_UNIFORM ? "uniform" : "closest", num_neighbors,
      max_dist,
      random_seed);
  // clang-format on
}

PSUtils::Result PSUtils::Sample(const Tensor& points,
                                const Tensor& points_padding,
                                const int32 num_seeded_points) const {
  if (opts_.cmethod == Options::C_UNIFORM &&
      opts_.nmethod == Options::N_UNIFORM) {
    CHECK_EQ(num_seeded_points, 0)
        << "Seeding only supported for Farthest Point Sampling";
    return DoSampling<UniformSelector, UniformSampler>(points, points_padding,
                                                       num_seeded_points);
  }
  if (opts_.cmethod == Options::C_UNIFORM &&
      opts_.nmethod == Options::N_CLOSEST) {
    CHECK_EQ(num_seeded_points, 0)
        << "Seeding only supported for Farthest Point Sampling";
    return DoSampling<UniformSelector, TopKSampler>(points, points_padding,
                                                    num_seeded_points);
  }
  if (opts_.cmethod == Options::C_FARTHEST &&
      opts_.nmethod == Options::N_UNIFORM) {
    return DoSampling<FarthestSelector, UniformSampler>(points, points_padding,
                                                        num_seeded_points);
  }
  CHECK_EQ(opts_.cmethod, Options::C_FARTHEST);
  CHECK_EQ(opts_.nmethod, Options::N_CLOSEST);
  return DoSampling<FarthestSelector, TopKSampler>(points, points_padding,
                                                   num_seeded_points);
}

}  // namespace car
}  // namespace lingvo
}  // namespace tensorflow
