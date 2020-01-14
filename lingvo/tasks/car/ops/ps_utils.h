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
#ifndef THIRD_PARTY_PY_LINGVO_TASKS_CAR_OPS_PS_UTILS_H_
#define THIRD_PARTY_PY_LINGVO_TASKS_CAR_OPS_PS_UTILS_H_

#include <limits>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace lingvo {
namespace car {

// Point cloud sampling utilities.
class PSUtils {
 public:
  struct Options {
    // Center selection method.
    enum CenterSelectionMethod { C_UNIFORM, C_FARTHEST };
    CenterSelectionMethod cmethod = C_FARTHEST;

    // The number of centers to sample.
    int num_centers = 128;

    // Points with z outside of the following range are not considered for
    // center selection.
    float center_z_min = std::numeric_limits<float>::lowest();
    float center_z_max = std::numeric_limits<float>::max();

    // Center selection method.
    enum NeighborSelectionMethod { N_UNIFORM, N_CLOSEST };
    NeighborSelectionMethod nmethod = N_UNIFORM;

    // For each center, sample this many points within the neighorhood.
    int num_neighbors = 1024;

    // Points with L2 distances in 3D from a center larger than this threshold
    // is not considered to be in the neighborhood.
    float max_dist = std::numeric_limits<float>::max();

    // The random seed.
    int random_seed = -1;

    enum NeighborSearchAlgorithm { N_AUTO, N_HASH };
    NeighborSearchAlgorithm neighbor_search_algorithm = N_AUTO;

    string DebugString() const;
  };

  explicit PSUtils(const Options& opts) : opts_(opts) {}

  // Samples centers within 'points' (3D) and creates neighborhood for each
  // center.
  //
  // If num_seeded_points is > 0, then the first num_seeded_points are used as
  // seeds for the center selection component of sampling, but are ignored for
  // neighborhood selection.
  struct Result {
    // [num_centers]. Indices of the center points.
    Tensor center;

    // [num_centers]. 0/1 paddings indicating that 0 means i-th center is a real
    // sampled center point, while 1 means otherwise.
    Tensor center_padding;

    // [num_centers, num_neighbors]. Indices of neighbors.
    Tensor indices;

    // indices[i, j] is a real point if and only if indices_padding[i, j] is
    // 0. Otherwise, indices_padding[i, j] is 1.0.
    Tensor indices_padding;
  };
  Result Sample(const Tensor& points, const Tensor& points_padding,
                const int32 num_seeded_points) const;

 private:
  const Options opts_;

  uint64 Seed() const;

  template <typename Selector, typename Sampler>
  Result DoSampling(const Tensor& points, const Tensor& points_padding,
                    const int32 num_seeded_points) const;
};

}  // namespace car
}  // namespace lingvo
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_LINGVO_TASKS_CAR_OPS_PS_UTILS_H_
