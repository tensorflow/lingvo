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

#include <queue>
#include <unordered_map>
#include <vector>

#include "lingvo/tasks/car/ops/box_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace lingvo {
namespace {

struct PairHashFn {
  size_t operator()(const std::pair<int32, int32>& pair) const {
    return std::hash<uint64>{}((static_cast<uint64>(pair.first) << 32) |
                               static_cast<uint64>(pair.second));
  }
};

class NonMaxSuppression3DOp : public OpKernel {
 public:
  explicit NonMaxSuppression3DOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("max_boxes_per_class", &max_boxes_per_class_));
  }

  void CheckShapes(OpKernelContext* ctx) {
    const Tensor& bboxes_3d = ctx->input(0);
    const Tensor& class_scores = ctx->input(1);
    const Tensor& nms_iou_threshold = ctx->input(2);
    const Tensor& score_threshold = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(bboxes_3d.shape()),
                errors::InvalidArgument("In[0] must be a matrix, but get ",
                                        bboxes_3d.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(class_scores.shape()),
                errors::InvalidArgument("In[1] must be a matrix, but get ",
                                        class_scores.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(nms_iou_threshold.shape()),
        errors::InvalidArgument("In[2] must be a vector, but get ",
                                nms_iou_threshold.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(score_threshold.shape()),
                errors::InvalidArgument("In[3] must be a vector, but get ",
                                        score_threshold.shape().DebugString()));
    OP_REQUIRES(ctx, bboxes_3d.dim_size(1) == 7,
                errors::InvalidArgument("bboxes must be of shape [-1, 7]. Is: ",
                                        bboxes_3d.shape().DebugString()));
    const int num_classes = class_scores.dim_size(1);
    OP_REQUIRES(ctx, nms_iou_threshold.dim_size(0) == num_classes,
                errors::InvalidArgument(
                    "nms_iou_threshold must be of shape [num_classes]. Is: ",
                    nms_iou_threshold.shape().DebugString()));
    OP_REQUIRES(ctx, score_threshold.dim_size(0) == num_classes,
                errors::InvalidArgument(
                    "score_threshold must be of shape [num_classes]. Is: ",
                    nms_iou_threshold.shape().DebugString()));
  }

  void Compute(OpKernelContext* ctx) override {
    CheckShapes(ctx);
    const Tensor& bboxes_3d = ctx->input(0);
    const Tensor& class_scores = ctx->input(1);
    const Tensor& nms_iou_threshold = ctx->input(2);
    const Tensor& score_threshold = ctx->input(3);

    const int num_bboxes = bboxes_3d.dim_size(0);
    const int num_classes = class_scores.dim_size(1);

    // Parse data into usable forms
    std::vector<box::Upright3DBox> boxes = box::ParseBoxesFromTensor(bboxes_3d);
    const auto& t_class_scores = class_scores.matrix<float>();
    const auto& t_nms_iou_threshold = nms_iou_threshold.vec<float>();
    const auto& t_score_threshold = score_threshold.vec<float>();

    struct Candidate {
      int box_idx;
      float score;
    };

    auto score_cmp = [](const Candidate& box1, const Candidate& box2) {
      return box1.score < box2.score;
    };

    // Allocate outputs
    Tensor* bbox_indices = nullptr;
    Tensor* bbox_scores = nullptr;
    Tensor* valid_mask = nullptr;

    auto output_shape = TensorShape({num_classes, max_boxes_per_class_});
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("bbox_indices", output_shape, &bbox_indices));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("bbox_scores", output_shape, &bbox_scores));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("valid_mask", output_shape, &valid_mask));
    auto t_bbox_indices = bbox_indices->matrix<int>();
    auto t_bbox_scores = bbox_scores->matrix<float>();
    auto t_valid_mask = valid_mask->matrix<float>();
    t_bbox_indices.setZero();
    t_bbox_scores.setZero();
    t_valid_mask.setZero();

    // Create cache for IoU calculations
    // Keys are pairs of box ids where the first id is always less than the
    // second id due to symmetry in the calculation.
    std::unordered_map<std::pair<int32, int32>, float, PairHashFn> iou_cache;
    auto get_iou = [&iou_cache, &boxes](int32 idx1, int32 idx2) {
      std::pair<int32, int32> key{std::min(idx1, idx2), std::max(idx1, idx2)};
      auto it = iou_cache.find(key);
      if (it != iou_cache.end()) {
        return it->second;
      }

      float iou = boxes[idx1].IoU(boxes[idx2]);
      iou_cache.emplace(key, iou);
      return iou;
    };

    for (int cls_idx = 0; cls_idx < num_classes; ++cls_idx) {
      // Use priority queue to sort candidates above the score threshold
      std::priority_queue<Candidate, std::deque<Candidate>, decltype(score_cmp)>
          candidate_priority_queue(score_cmp);
      for (int box_idx = 0; box_idx < num_bboxes; ++box_idx) {
        if (t_class_scores(box_idx, cls_idx) >= t_score_threshold(cls_idx)) {
          candidate_priority_queue.emplace(
              Candidate({box_idx, t_class_scores(box_idx, cls_idx)}));
        }
      }

      std::vector<Candidate> selected;
      Candidate next_candidate;
      while ((selected.size() < max_boxes_per_class_) &&
             (!candidate_priority_queue.empty())) {
        next_candidate = candidate_priority_queue.top();
        candidate_priority_queue.pop();

        // Idea taken from tensorflow/core/kernels/non_max_suppression_op.cc
        // Overlapping boxes are likely to have similar scores,
        // therefore we iterate through the previously selected boxes backwards
        // in order to see if `next_candidate` should be suppressed.
        bool should_select = true;
        for (int selected_idx = static_cast<int>(selected.size()) - 1;
             selected_idx >= 0; --selected_idx) {
          if (get_iou(next_candidate.box_idx, selected[selected_idx].box_idx) >
              t_nms_iou_threshold(cls_idx)) {
            should_select = false;
            break;
          }
        }

        if (should_select) {
          selected.push_back(next_candidate);
        }
      }

      // For each class, copy results into output tensors.
      // We can just use size since we protect against ever selecting more
      // than max_boxes_per_class_ per class.
      for (int insert_idx = 0; insert_idx < selected.size(); insert_idx++) {
        const auto& to_insert = selected[insert_idx];
        t_bbox_indices(cls_idx, insert_idx) = to_insert.box_idx;
        t_bbox_scores(cls_idx, insert_idx) = to_insert.score;
        t_valid_mask(cls_idx, insert_idx) = 1.0;
      }
    }
  }

 private:
  // These are attributes and logically const.
  int max_boxes_per_class_;
};

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppression3D").Device(DEVICE_CPU),
                        NonMaxSuppression3DOp);

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
