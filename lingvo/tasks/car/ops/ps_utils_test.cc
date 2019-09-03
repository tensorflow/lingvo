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
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace lingvo {
namespace car {

struct Point {
  float x;
  float y;
};

// Generates n centers (i-th center is on (i, i)).
// Generate m points near each center.
Tensor GeneratePoints(int n, int m) {
  std::vector<Point> points;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      float v = i + j / 1000.0;
      points.push_back(Point{v, v});
    }
  }
  std::mt19937 rng(39183);
  std::shuffle(points.begin(), points.end(), rng);
  Tensor ret(DT_FLOAT, {n * m, 3});
  auto points_t = ret.matrix<float>();
  for (int i = 0; i < n * m; ++i) {
    points_t(i, 0) = points[i].x;
    points_t(i, 1) = points[i].y;
    points_t(i, 2) = 0;
  }
  return ret;
}

void Log(const Tensor& points, const PSUtils::Result& result) {
  const int n = result.center.dim_size(0);
  CHECK_EQ(result.indices.dim_size(0), n);
  const int m = result.indices.dim_size(1);
  CHECK_EQ(result.padding.dim_size(0), n);
  CHECK_EQ(result.padding.dim_size(1), m);
  auto points_t = points.matrix<float>();
  auto center_t = result.center.vec<int32>();
  auto indices_t = result.indices.matrix<int32>();
  auto padding_t = result.padding.matrix<float>();
  for (int i = 0; i < n; ++i) {
    fprintf(stdout, "(%6.3f %6.3f) : ", points_t(center_t(i), 0),
            points_t(center_t(i), 1));
    for (int j = 0; j < m; ++j) {
      fprintf(stdout, "(%6.3f %6.3f)/%1.0f", points_t(indices_t(i, j), 0),
              points_t(indices_t(i, j), 1), padding_t(i, j));
    }
    fprintf(stdout, "\n");
  }
}

std::vector<int> GetCenters(const Tensor& points,
                            const PSUtils::Result& result) {
  const int n = result.center.dim_size(0);
  CHECK_EQ(result.indices.dim_size(0), n);
  const int m = result.indices.dim_size(1);
  CHECK_EQ(result.padding.dim_size(0), n);
  CHECK_EQ(result.padding.dim_size(1), m);
  auto points_t = points.matrix<float>();
  auto center_t = result.center.vec<int32>();
  auto indices_t = result.indices.matrix<int32>();
  std::vector<int> centers;
  for (int i = 0; i < n; ++i) {
    const int center = static_cast<int>(points_t(center_t(i), 0));
    for (int j = 0; j < m; ++j) {
      CHECK_EQ(points_t(indices_t(i, j), 0), points_t(indices_t(i, j), 1));
      CHECK_EQ(center, static_cast<int>(points_t(indices_t(i, j), 0)));
    }
    centers.push_back(center);
  }
  return centers;
}

#if defined(PLATFORM_GOOGLE)
TEST(PSUtilsTest, Uniform_Uniform) {
  PSUtils::Options opts;
  opts.cmethod = PSUtils::Options::C_UNIFORM;
  opts.nmethod = PSUtils::Options::N_UNIFORM;
  opts.num_centers = 8;
  opts.num_neighbors = 16;
  opts.max_dist = 1.0;
  opts.random_seed = 12345;
  PSUtils fu(opts);
  auto points = GeneratePoints(8, 100);
  auto ret = fu.Sample(points);
  Log(points, ret);
  // Generated points will correspond to clusters like:
  // (0, 0), (0, 0.001), (0, 0.002), ...
  // (1, 1), (1, 1.001), (1, 1.002), ...
  // ...
  // (7, 7), (7, 7.001), (7, 7.002), ...
  //
  // GetCenters returns the first coordinate of each point, and hence, we
  // expect them to repeat.
  // Some clusters are sampled more than once.
  EXPECT_EQ(GetCenters(points, ret),
            std::vector<int>({1, 0, 2, 4, 0, 3, 1, 3}));
}

TEST(PSUtilsTest, Uniform_Closest) {
  PSUtils::Options opts;
  opts.cmethod = PSUtils::Options::C_UNIFORM;
  opts.nmethod = PSUtils::Options::N_CLOSEST;
  opts.num_centers = 8;
  opts.num_neighbors = 16;
  opts.max_dist = 1.0;
  opts.random_seed = 12345;
  PSUtils fu(opts);
  auto points = GeneratePoints(8, 100);
  auto ret = fu.Sample(points);
  Log(points, ret);
  // Some clusters are sampled more than once.
  EXPECT_EQ(GetCenters(points, ret),
            std::vector<int>({1, 0, 2, 4, 0, 3, 1, 3}));
}

TEST(PSUtilsTest, Farthest_Uniform) {
  PSUtils::Options opts;
  opts.cmethod = PSUtils::Options::C_FARTHEST;
  opts.nmethod = PSUtils::Options::N_UNIFORM;
  opts.num_centers = 8;
  opts.num_neighbors = 16;
  opts.max_dist = 1.0;
  opts.random_seed = 12345;
  PSUtils fu(opts);
  auto points = GeneratePoints(8, 100);
  auto ret = fu.Sample(points);
  Log(points, ret);
  // Generated points will correspond to clusters like:
  // (0, 0), (0, 0.001), (0, 0.002), ...
  // (1, 1), (1, 1.001), (1, 1.002), ...
  // ...
  // (7, 7), (7, 7.001), (7, 7.002), ...
  //
  // GetCenters returns the first coordinate of each point. With farthest point
  // sampling, we expect the samples to have all different first coordinates.
  // All 8 clusters are covered.
  EXPECT_EQ(GetCenters(points, ret),
            std::vector<int>({3, 7, 0, 5, 1, 6, 4, 2}));
}

TEST(PSUtilsTest, Farthest_Closest) {
  PSUtils::Options opts;
  opts.cmethod = PSUtils::Options::C_FARTHEST;
  opts.nmethod = PSUtils::Options::N_CLOSEST;
  opts.num_centers = 8;
  opts.num_neighbors = 16;
  opts.max_dist = 10.0;
  opts.random_seed = 12345;
  PSUtils fu(opts);
  auto points = GeneratePoints(8, 100);
  auto ret = fu.Sample(points);
  Log(points, ret);
  // All 8 clusters are covered.
  EXPECT_EQ(GetCenters(points, ret),
            std::vector<int>({3, 7, 0, 5, 1, 6, 4, 2}));
}

void BM_Farthest(int iters, int num_centers, int num_neighbors) {
  testing::StopTiming();
  testing::SetLabel(strings::Printf("#Centers=%4d #Neighbors=%4d",
                                            num_centers, num_neighbors));
  PSUtils::Options opts;
  opts.cmethod = PSUtils::Options::C_FARTHEST;
  opts.nmethod = PSUtils::Options::N_UNIFORM;
  opts.num_centers = num_centers;
  opts.num_neighbors = num_neighbors;
  opts.max_dist = 1.0;
  opts.random_seed = -1;
  PSUtils fu(opts);
  Tensor points = GeneratePoints(1000, 100);
  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    auto ret = fu.Sample(points);
  }
}

BENCHMARK(BM_Farthest)->RangePair(1, 1024, 1, 1024);
#endif

}  // namespace car
}  // namespace lingvo
}  // namespace tensorflow
