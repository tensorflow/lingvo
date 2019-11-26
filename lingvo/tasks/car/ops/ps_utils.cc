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

#include "lingvo/core/ops/mutex.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace lingvo {
namespace car {

namespace {

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
    MutexLock l(&mu_);
    return rng_();
  }

 private:
  Mutex mu_;
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
                                    const Tensor& points_padding) const {
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

  for (int cur_batch = 0; cur_batch < batch_size; ++cur_batch) {
    std::vector<bool> candidates(num_points);
    for (int i = 0; i < num_points; ++i) {
      candidates[i] = (points_padding_t(cur_batch, i) == 0.0) &&
                      (opts_.center_z_min <= points_t(cur_batch, i, 2)) &&
                      (points_t(cur_batch, i, 2) <= opts_.center_z_max);
    }

    Selector selector(candidates, Seed());
    Sampler sampler(opts_.num_neighbors, Seed());

    for (int i = 0; i < opts_.num_centers; ++i) {
      // Pick a point as i-th center.
      auto k = selector.Get();
      if (k < 0) {
        center_padding_t(cur_batch, i) = 1.0;
        continue;
      }
      center_padding_t(cur_batch, i) = 0.0;
      center_t(cur_batch, i) = k;

      // Goes through all points. If j-th point is within a radius of center,
      // adds it to the sampler.
      sampler.Reset();
      for (int j = 0; j < num_points; ++j) {
        if (points_padding_t(cur_batch, j) == 0.0) {
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
                                const Tensor& points_padding) const {
  if (opts_.cmethod == Options::C_UNIFORM &&
      opts_.nmethod == Options::N_UNIFORM) {
    return DoSampling<UniformSelector, UniformSampler>(points, points_padding);
  }
  if (opts_.cmethod == Options::C_UNIFORM &&
      opts_.nmethod == Options::N_CLOSEST) {
    return DoSampling<UniformSelector, TopKSampler>(points, points_padding);
  }
  if (opts_.cmethod == Options::C_FARTHEST &&
      opts_.nmethod == Options::N_UNIFORM) {
    return DoSampling<FarthestSelector, UniformSampler>(points, points_padding);
  }
  CHECK_EQ(opts_.cmethod, Options::C_FARTHEST);
  CHECK_EQ(opts_.nmethod, Options::N_CLOSEST);
  return DoSampling<FarthestSelector, TopKSampler>(points, points_padding);
}

}  // namespace car
}  // namespace lingvo
}  // namespace tensorflow
