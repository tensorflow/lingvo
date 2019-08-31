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

#include "lingvo/tasks/car/ops/image_metrics.h"

#include <algorithm>
#include <cmath>

namespace tensorflow {
namespace lingvo {
namespace image {

float Box2D::Length(const Box2D::Interval& a) {
  return std::max(0.f, a.max - a.min);
}

float Box2D::Intersection(const Box2D::Interval& a, const Box2D::Interval& b) {
  Interval c;
  c.min = std::max(a.min, b.min);
  c.max = std::min(a.max, b.max);
  return Length(c);
}

float Box2D::Area() const { return Length(x) * Length(y); }

float Box2D::Intersection(const Box2D& other) const {
  return Intersection(x, other.x) * Intersection(y, other.y);
}

float Box2D::Union(const Box2D& other) const {
  return Area() + other.Area() - Intersection(other);
}

float Box2D::IoU(const Box2D& other) const {
  const float total = Union(other);
  if (total > 0) {
    return Intersection(other) / total;
  } else {
    return 0.0;
  }
}

float Box2D::Overlap(const Box2D& other) const {
  const float intersection = Intersection(other);
  return intersection > 0 ? intersection / Area() : 0.0;
}

namespace KITTI {
bool IsBetterMatch(const MatchResult& match1, const MatchResult& match2,
                   MatchingCriterion criterion) {
  // Always true if match2 is "unmatched".
  if (match2.matched_idx == kUnMatched) {
    return true;
  }
  if (criterion == kBestScore) {
    return match1.detection_score > match2.detection_score;
  } else if (criterion == kBestIoU) {
    if (match1.pd_ignore_type == kDontIgnore) {
      if (match2.pd_ignore_type != kDontIgnore ||
          match1.matched_overlap > match2.matched_overlap) {
        return true;
      }
    }
  }
  return false;
}

float ComputePrecision(
    const std::unordered_map<int64_t, std::vector<MatchResult>>& pd_assignments,
    const float score_threshold) {
  int total = 0;
  int tp = 0;
  int self_ignore = 0, match_ignore = 0, unmatched = 0;
  for (const auto& per_image_assignment : pd_assignments) {
    for (const auto& match : per_image_assignment.second) {
      if (match.detection_score >= score_threshold &&
          match.gt_ignore_type == kDontIgnore &&
          match.pd_ignore_type == kDontIgnore) {
        total++;
        if (match.matched_idx != kUnMatched) {
          tp++;
        }
      }
      // Only for debugging.
      if (match.pd_ignore_type != kDontIgnore) {
        self_ignore++;
      }
      if (match.gt_ignore_type != kDontIgnore) {
        match_ignore++;
      }
      if (match.matched_idx == kUnMatched) {
        unmatched++;
      }
    }
  }

  if (total == 0) {
    return 1;
  } else {
    return static_cast<float>(tp) / total;
  }
}

std::vector<float> FindThresholds(
    const std::unordered_map<int64_t, std::vector<MatchResult>>& gt_assignment,
    const int num_recall_points) {
  std::vector<float> thresholds;
  std::vector<float> all_matched_scores;
  int total_gt = 0;
  for (const auto& per_img_assignmeht : gt_assignment) {
    for (const auto& match : per_img_assignmeht.second) {
      if (match.gt_ignore_type == kDontIgnore) {
        if (match.matched_idx != kUnMatched &&
            match.pd_ignore_type == kDontIgnore) {
          all_matched_scores.push_back(match.detection_score);
        }
        total_gt++;
      }
    }
  }
  if (total_gt == 0) {
    // There's no groundtruth to evaluate. We return an empty vector.
    return thresholds;
  }
  std::sort(all_matched_scores.begin(), all_matched_scores.end(),
            std::greater<float>());
  for (int i = 1; i <= all_matched_scores.size(); ++i) {
    const float left_recall = static_cast<float>(i) / total_gt;
    const float right_recall = static_cast<float>(i + 1) / total_gt;
    const float target_recall =
        thresholds.size() / static_cast<float>(num_recall_points - 1);
    if (right_recall - target_recall >= target_recall - left_recall ||
        i == all_matched_scores.size()) {
      thresholds.push_back(all_matched_scores[i - 1]);
    }
  }
  return thresholds;
}

}  // namespace KITTI

}  // namespace image
}  // namespace lingvo
}  // namespace tensorflow
