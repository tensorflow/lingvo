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

#ifndef THIRD_PARTY_PY_LINGVO_TASKS_CAR_OPS_IMAGE_METRICS_H_
#define THIRD_PARTY_PY_LINGVO_TASKS_CAR_OPS_IMAGE_METRICS_H_

#include <list>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace lingvo {
namespace image {

struct Box2D {
  struct Interval {
    float min = 0;
    float max = 0;
  };
  Interval x;
  Interval y;
  static float Length(const Interval& a);
  static float Intersection(const Interval& a, const Interval& b);
  float Area() const;
  float Intersection(const Box2D& other) const;
  float Union(const Box2D& other) const;
  // Intersection of this box and the given box normalized over the union of
  // this box and the given box.
  float IoU(const Box2D& other) const;
  // Intersection of this box and the given box normalized over the area of
  // this box.
  float Overlap(const Box2D& other) const;
};

// If the value is:
//   - kDontIgnore: The object is included in this evaluation.
//   - kIgnoreOneMatch: the first matched prediction bbox will be ignored. This
//      is useful when this groundtruth object is not intended to be evaluated.
//   - kIgnoreAllMatches: all matched prediction bbox will be ignored. Typically
//      it is used to mark an area that has not been labeled.
enum IgnoreType {
  kDontIgnore = 0,
  kIgnoreOneMatch = 1,
  kIgnoreAllMatches = 2,
};

template <class BoxType>
struct Detection {
 public:
  bool difficult = false;
  int64_t imgid = 0;
  float score = 0;
  BoxType box;
  IgnoreType ignore = IgnoreType::kDontIgnore;
};

// Precision and recall.
struct PR {
  float p = 0;
  float r = 0;
  PR(const float p_, const float r_) : p(p_), r(r_) {}
};

template <class BoxType>
class AveragePrecision {
 public:
  // iou_threshold: A predicted box matches a ground truth box if and only if
  //   IoU between these two are larger than this iou_threshold. Default: 0.5.
  // num_recall_points: AP is computed as the average of maximum precision at (1
  //   + num_recall_points) recall levels. E.g., if num_recall_points is 10,
  //   recall levels are 0., 0.1, 0.2, ..., 0.9, 1.0. Default: 100
  // normalized_coordinate: If true, bounding box's coordinates are within [0,
  //  1), and the interval is left-close and right-open. Hence, the length of an
  //  interval is simply interval.max - interval.min. Otherwise, coordinates are
  //  in the image pixel space and the interval is close on both ends. I.e., the
  //  length of an interval is interval.max - interval.min + 1.
  //  NOTE: voc_eval.py used by Fast-RCNN sets normalized_coordinate=false.
  struct Options {
    float iou_threshold = 0.5;
    int num_recall_points = 100;
  };
  AveragePrecision() : AveragePrecision(Options()) {}
  explicit AveragePrecision(const Options& opts) : opts_(opts) {}

  // Given a sequence of precision-recall points ordered by the recall in
  // non-increasing order, returns the average of maximum precisions at
  // different recall values (0.0, 0.1, 0.2, ..., 0.9, 1.0).
  // The p-r pairs at these fixed recall points will be written to pr_out, if
  // it is not null_ptr.
  float FromPRCurve(const std::vector<PR>& pr,
                    std::vector<PR>* pr_out = nullptr);

  // An axis aligned bounding box for an image with id 'imageid'.  Score
  // indicates its confidence.
  //
  // 'difficult' is a special bit specific to Pascal VOC dataset and tasks using
  // the data. If 'difficult' is true, by convention, the box is often ignored
  // during the AP calculation. I.e., if a predicted box matches a 'difficult'
  // ground box, this predicted box is ignored as if the model does not make
  // such a prediction.

  // Given the set of ground truth boxes and a set of predicted boxes, returns
  // the average of the maximum precisions at different recall values.
  float FromBoxes(const std::vector<Detection<BoxType>>& groundtruth,
                  const std::vector<Detection<BoxType>>& prediction,
                  std::vector<PR>* pr_out = nullptr);

  // The KITTI variant of AP calculation.
  // Comparing with VOC/COCO AP, there are 3 main differences.
  //    - KITTI AP has two matching passes. The 1st pass matches every gt bbox
  //    with the pd bbox of highest score while having an IoU larger than the
  //    threshold. The 2nd pass matches every gt with the pd bbox of largest
  //    IoU. In VOC/COCO AP there's only one pass that does the same as KITTI's
  //    1st pass.
  //
  //    - The operating points on PR curve are decided by the result of the 1st
  //    pass whereas the actuall PR values are computed using 2nd pass.
  //
  //    - Special "ignore" bit can be set on groundtruth or prediction bboxes.
  //       - On groundtruth, "ignore" means the bbox is of a neighbor class, or
  //         of a "difficuty" level that is not intended to be evaluated.
  //       - On prediction, "ignore" means the bbox, when projected to the
  //         camera image space, has its height smaller than a threshold.
  float FromBoxesKITTI(const std::vector<Detection<BoxType>>& groundtruth,
                       const std::vector<Detection<BoxType>>& prediction,
                       std::vector<PR>* pr_out = nullptr,
                       std::vector<float>* is_hit = nullptr,
                       std::vector<float>* score = nullptr);

 private:
  Options opts_;
};

template <class BoxType>
float AveragePrecision<BoxType>::FromPRCurve(const std::vector<PR>& pr,
                                             std::vector<PR>* pr_out) {
  // Because pr[...] are ordered by recall, iterate backward to compute max
  // precision. p(r) = max_{r' >= r} p(r') for r in 0.0, 0.1, 0.2, ..., 0.9,
  // 1.0. Then, take the average of (num_recal_points) quantities.
  float p = 0;
  float sum = 0;
  int r_level = opts_.num_recall_points;
  for (int i = pr.size() - 1; i >= 0; --i) {
    const PR& item = pr[i];
    if (i > 0) {
      CHECK_GE(item.r, pr[i - 1].r);  // Ordered.
    }
    // Because r takes values opts_.num_recall_points, opts_.num_recall_points -
    // 1, ..., 0, the following condition is checking whether item.r crosses r /
    // opts_.num_recall_points. I.e., 1.0, 0.90, ..., 0.01, 0.0.  We don't use
    // float to represent r because 0.01 is not representable precisely.
    while (item.r * opts_.num_recall_points < r_level) {
      const float recall =
          static_cast<float>(r_level) / opts_.num_recall_points;
      CHECK_GE(r_level, 0);
      sum += p;
      r_level -= 1;
      if (pr_out != nullptr) {
        pr_out->emplace_back(p, recall);
      }
    }
    p = std::max(p, item.p);
  }
  for (; r_level >= 0; --r_level) {
    const float recall = static_cast<float>(r_level) / opts_.num_recall_points;
    sum += p;
    if (pr_out != nullptr) {
      pr_out->emplace_back(p, recall);
    }
  }
  return sum / (1 + opts_.num_recall_points);
}

template <class BoxType>
float AveragePrecision<BoxType>::FromBoxes(
    const std::vector<Detection<BoxType>>& groundtruth,
    const std::vector<Detection<BoxType>>& prediction,
    std::vector<PR>* pr_out) {
  // Index ground truth boxes based on imageid.
  std::unordered_map<int64_t, std::list<Detection<BoxType>>> gt;
  int num_gt = 0;
  for (auto& box : groundtruth) {
    gt[box.imgid].push_back(box);
    if (!box.difficult && box.ignore == kDontIgnore) {
      ++num_gt;
    }
  }

  if (num_gt == 0) {
    return NAN;
  }

  // Sort all predicted boxes by their scores in a non-ascending order.
  std::vector<Detection<BoxType>> pd = prediction;
  std::sort(pd.begin(), pd.end(),
            [](const Detection<BoxType>& a, const Detection<BoxType>& b) {
              return a.score > b.score;
            });

  // Computes p-r for every prediction.
  std::vector<PR> pr;
  int correct = 0;
  int num_pd = 0;
  for (int i = 0; i < pd.size(); ++i) {
    const Detection<BoxType>& b = pd[i];
    auto* g = &gt[b.imgid];
    auto best = g->end();
    float best_iou = -INFINITY;
    for (auto it = g->begin(); it != g->end(); ++it) {
      const auto iou = b.box.IoU(it->box);
      if (iou > best_iou) {
        best = it;
        best_iou = iou;
      }
    }
    if ((best != g->end()) && (best_iou >= opts_.iou_threshold)) {
      if (best->difficult) {
        continue;
      }
      switch (best->ignore) {
        case kDontIgnore: {
          ++correct;
          ++num_pd;
          g->erase(best);
          pr.push_back({static_cast<float>(correct) / num_pd,
                        static_cast<float>(correct) / num_gt});
          break;
        }
        case kIgnoreOneMatch: {
          g->erase(best);
          break;
        }
        case kIgnoreAllMatches: {
          break;
        }
      }
    } else {
      ++num_pd;
      pr.push_back({static_cast<float>(correct) / num_pd,
                    static_cast<float>(correct) / num_gt});
    }
  }
  return FromPRCurve(pr, pr_out);
}

namespace KITTI {

enum MatchingCriterion {
  // Match a groundtruth to the prediction of highest score.
  kBestScore = 0,
  // Match a groundtruth to the prediction of largest IoU.
  kBestIoU = 1,
};

const int kUnMatched = -1;
struct MatchResult {
  int matched_idx = kUnMatched;
  // When MatchingMode is kBestIoU and matched_idx is not kUnMatched,
  // this field will contain the IoU between the prediction and the groundtruth.
  float matched_overlap = 0;
  // When MatchingMode is kBestIoU and matched_idx is not kUnMatched,
  // this field will contain the confidence score of the prediction.
  float detection_score = 0;
  IgnoreType gt_ignore_type = IgnoreType::kDontIgnore;
  IgnoreType pd_ignore_type = IgnoreType::kDontIgnore;
};

// Whether match1 is a better match than match2.
bool IsBetterMatch(const MatchResult& match1, const MatchResult& match2,
                   MatchingCriterion criterion);

// Compute precision from detection assignments given a score threshold.
// Input arguments:
//    pd_assignments: the 2nd pass matching results on all predictions.
//    score_threshold: only the predictions with score above this threshold will
//      be evaluated.
float ComputePrecision(
    const std::unordered_map<int64_t, std::vector<MatchResult>>& pd_assignments,
    const float score_threshold);

// Match groundtruth with predictions from one scene.
template <class BoxType>
void MatchOneScene(const std::vector<Detection<BoxType>>& groundtruth,
                   const std::vector<Detection<BoxType>>& prediction,
                   const MatchingCriterion criterion, const float iou_threshold,
                   const float score_threshold,
                   std::vector<MatchResult>* gt_assignment,
                   std::vector<MatchResult>* pd_assignment) {
  gt_assignment->clear();
  pd_assignment->clear();
  gt_assignment->resize(groundtruth.size());
  pd_assignment->resize(prediction.size());
  for (int j_pd = 0; j_pd < prediction.size(); ++j_pd) {
    // Transfer over detection scores and ignore type from 'prediction'
    pd_assignment->at(j_pd).detection_score = prediction[j_pd].score;
    pd_assignment->at(j_pd).pd_ignore_type = prediction[j_pd].ignore;
  }
  for (int i_gt = 0; i_gt < groundtruth.size(); ++i_gt) {
    if (groundtruth[i_gt].ignore == IgnoreType::kIgnoreAllMatches) {
      continue;
    }
    MatchResult best_matched_pd;
    best_matched_pd.gt_ignore_type = groundtruth[i_gt].ignore;
    for (int j_pd = 0; j_pd < prediction.size(); ++j_pd) {
      // For this groundtruth, find the best unassigned prediction that matches.
      if (pd_assignment->at(j_pd).matched_idx != kUnMatched ||
          prediction[j_pd].score < score_threshold) {
        // Skip predictions that have already been matched, or predictions below
        // the score threshold.
        continue;
      }
      const float iou = groundtruth[i_gt].box.IoU(prediction[j_pd].box);
      MatchResult curr;
      curr.matched_idx = j_pd;
      curr.matched_overlap = iou;
      curr.detection_score = prediction[j_pd].score;
      curr.gt_ignore_type = groundtruth[i_gt].ignore;
      curr.pd_ignore_type = prediction[j_pd].ignore;
      if (iou > iou_threshold &&
          IsBetterMatch(curr, best_matched_pd, criterion)) {
        best_matched_pd = curr;
      }
    }
    MatchResult best_matched_gt = best_matched_pd;
    best_matched_gt.matched_idx = i_gt;

    gt_assignment->at(i_gt) = best_matched_pd;
    if (best_matched_pd.matched_idx != kUnMatched) {
      pd_assignment->at(best_matched_pd.matched_idx) = best_matched_gt;
    }
  }
}

// Input arguments:
//   groundtruth: image_id -> list of bboxes mapping.
//   prediction: image_id -> list of bboxes mapping. The key set must be the
//    same as the one in groundtruth.
//   criterion: should be kBestScore in 1st pass matching and kBestIoU in the
//     2nd pass.
//   iou_threshold: bboxes can be matched if ther IoU is greater than this.
//   score_thresold: predictions with score lower than this are ignored.
//   gt_assignments: the matching results for every groundtruth.
//   pd_assignments: the matching results for every prediction.
template <class BoxType>
void MatchAll(
    const std::unordered_map<int64_t, std::vector<Detection<BoxType>>>&
        groundtruth,
    const std::unordered_map<int64_t, std::vector<Detection<BoxType>>>&
        prediction,
    const MatchingCriterion criterion, const float iou_threshold,
    const float score_threshold,
    std::unordered_map<int64_t, std::vector<MatchResult>>* gt_assignments,
    std::unordered_map<int64_t, std::vector<MatchResult>>* pd_assignments) {
  CHECK_EQ(groundtruth.size(), prediction.size());
  std::vector<int64_t> all_img_keys;
  for (const auto& gt_iter : groundtruth) {
    const int64_t img_id = gt_iter.first;
    const auto pd_iter = prediction.find(img_id);
    CHECK(pd_iter != prediction.end())
        << "Groundtruth and prediction must have the same key set";
    const auto& gt = gt_iter.second;
    const auto& pd = pd_iter->second;
    MatchOneScene(gt, pd, criterion, iou_threshold, score_threshold,
                  &((*gt_assignments)[img_id]), &((*pd_assignments)[img_id]));
  }
}

std::vector<float> FindThresholds(
    const std::unordered_map<int64_t, std::vector<MatchResult>>& gt_assignment,
    const int num_recall_points);

}  // namespace KITTI

// TODO(shlens): Consider removing score as a returned value as this is in
// theory already available in prediction. Another option is to combine as a
// std::pair with is_hit.
template <class BoxType>
float AveragePrecision<BoxType>::FromBoxesKITTI(
    const std::vector<Detection<BoxType>>& groundtruth,
    const std::vector<Detection<BoxType>>& prediction, std::vector<PR>* pr_out,
    std::vector<float>* is_hit, std::vector<float>* score) {
  std::unordered_map<int64_t, std::vector<Detection<BoxType>>> gt_bins;
  std::unordered_map<int64_t, std::vector<Detection<BoxType>>> pd_bins;
  // Ensure that gt_bins and pd_bins have the same key sets.
  for (const auto& box : groundtruth) {
    gt_bins[box.imgid].push_back(box);
    if (pd_bins.find(box.imgid) == pd_bins.end()) {
      pd_bins[box.imgid] = std::vector<Detection<BoxType>>();
    }
  }
  std::vector<int64_t> index_to_image_id;
  index_to_image_id.reserve(prediction.size());
  std::vector<int64_t> index_to_prediction_index;
  index_to_prediction_index.reserve(prediction.size());
  for (int i = 0; i < prediction.size(); ++i) {
    auto& box = prediction[i];
    pd_bins[box.imgid].push_back(box);
    index_to_image_id.push_back(box.imgid);
    index_to_prediction_index.push_back(pd_bins[box.imgid].size() - 1);
    if (gt_bins.find(box.imgid) == gt_bins.end()) {
      gt_bins[box.imgid] = std::vector<Detection<BoxType>>();
    }
  }
  // Pass 1 matching: find overlapping detection of best score.
  std::unordered_map<int64_t, std::vector<KITTI::MatchResult>> gt_assignments;
  std::unordered_map<int64_t, std::vector<KITTI::MatchResult>> pd_assignments;
  KITTI::MatchAll(gt_bins, pd_bins, KITTI::kBestScore, opts_.iou_threshold, 0,
                  &gt_assignments, &pd_assignments);
  for (int i = 0; i < prediction.size(); ++i) {
    const int image_id = index_to_image_id[i];
    const int prediction_index = index_to_prediction_index[i];
    const float detection_score =
        pd_assignments[image_id][prediction_index].detection_score;
    float correct = 0.0;
    if (pd_assignments[image_id][prediction_index].matched_idx !=
        KITTI::kUnMatched) {
      correct = 1.0;
    }
    if (score != nullptr) {
      score->push_back(detection_score);
    }
    if (is_hit != nullptr) {
      is_hit->push_back(correct);
    }
  }

  const auto thresholds =
      KITTI::FindThresholds(gt_assignments, opts_.num_recall_points);
  pr_out->clear();
  float ap = 0;
  for (int i = 0; i < thresholds.size(); ++i) {
    // Pass 2 matching: find detection above the score threshold with largest
    // overlap.
    KITTI::MatchAll(gt_bins, pd_bins, KITTI::kBestIoU, opts_.iou_threshold,
                    thresholds[i], &gt_assignments, &pd_assignments);
    const float precision =
        KITTI::ComputePrecision(pd_assignments, thresholds[i]);
    const float recall = static_cast<float>(i) / (opts_.num_recall_points - 1);

    pr_out->push_back({precision, recall});
    ap += precision;
  }

  for (int i = thresholds.size() - 2; i >= 0; --i) {
    pr_out->at(i).p = std::max(pr_out->at(i + 1).p, pr_out->at(i).p);
  }
  return ap / opts_.num_recall_points;
}

}  // namespace image
}  // namespace lingvo
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_LINGVO_TASKS_CAR_OPS_IMAGE_METRICS_H_
