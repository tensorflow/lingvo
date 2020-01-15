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

#include <cstddef>

#include "lingvo/tasks/car/ops/box_util.h"
#include "lingvo/tasks/car/ops/image_metrics.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace lingvo {
namespace {

enum class APAlgorithm {
  kVOC = 0,
  kKITTI = 1,
};

class AP3DOp final : public OpKernel {
 public:
  explicit AP3DOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_recall_points", &num_recall_points_));
    string ap_algorithm_name;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("algorithm", &ap_algorithm_name));
    if (ap_algorithm_name == "KITTI") {
      ap_algorithm_ = APAlgorithm::kKITTI;
    } else if (ap_algorithm_name == "VOC") {
      ap_algorithm_ = APAlgorithm::kVOC;
    } else {
      OP_REQUIRES(
          ctx, false,
          errors::InvalidArgument("algorithm must be one of \"KITTI\", \"VOC\","
                                  "but got ",
                                  ap_algorithm_name));
    }

    OP_REQUIRES(
        ctx, num_recall_points_ > 0,
        errors::InvalidArgument("num_recall_points must be positive but get ",
                                num_recall_points_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* gt_bbox = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("groundtruth_bbox", &gt_bbox));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(gt_bbox->shape()),
                errors::InvalidArgument("bbox must be a matrix, but get ",
                                        gt_bbox->shape().DebugString()));
    OP_REQUIRES(ctx, gt_bbox->dim_size(1) == 7,
                errors::InvalidArgument("bbox must be [:, 7], but get ",
                                        gt_bbox->shape().DebugString()));
    const int n = gt_bbox->dim_size(0);

    const Tensor* gt_imageid = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("groundtruth_imageid", &gt_imageid));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(gt_imageid->shape()),
                errors::InvalidArgument("imageid must be a vector, but get ",
                                        gt_imageid->shape().DebugString()));
    OP_REQUIRES(ctx, gt_imageid->dim_size(0) == n,
                errors::InvalidArgument("imageid shape mismatch, get ",
                                        gt_imageid->shape().DebugString()));
    const Tensor* gt_ignore = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("groundtruth_ignore", &gt_ignore));
    OP_REQUIRES(ctx, gt_imageid->shape().IsSameSize(gt_ignore->shape()),
                errors::InvalidArgument("gt_ignore shape mismatch: ",
                                        gt_ignore->shape().DebugString()));

    std::vector<image::Detection<box::Upright3DBox>> groundtruth;
    std::vector<box::Upright3DBox> groundtruth_boxes =
        box::ParseBoxesFromTensor(*gt_bbox);
    groundtruth.reserve(n);
    // Ignore difficulty as it's currently used by VOC only.
    for (int i = 0; i < n; ++i) {
      image::Detection<box::Upright3DBox> g;
      g.difficult = false;
      g.imgid = gt_imageid->flat<int32>()(i);
      g.score = 1.0;
      g.box = groundtruth_boxes[i];
      g.ignore = static_cast<image::IgnoreType>(gt_ignore->flat<int32>()(i));
      groundtruth.push_back(g);
    }

    const Tensor* pd_bbox = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("prediction_bbox", &pd_bbox));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(pd_bbox->shape()),
                errors::InvalidArgument("bbox must be a matrix, but get ",
                                        pd_bbox->shape().DebugString()));
    OP_REQUIRES(ctx, pd_bbox->dim_size(1) == 7,
                errors::InvalidArgument("bbox must be [:, 7], but get ",
                                        pd_bbox->shape().DebugString()));
    const int m = pd_bbox->dim_size(0);

    const Tensor* pd_imageid = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("prediction_imageid", &pd_imageid));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(pd_imageid->shape()),
                errors::InvalidArgument("imageid must be a vector, but get ",
                                        pd_imageid->shape().DebugString()));
    OP_REQUIRES(ctx, pd_imageid->dim_size(0) == m,
                errors::InvalidArgument("imageid shape mismatch, get ",
                                        pd_imageid->shape().DebugString()));

    const Tensor* pd_score = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("prediction_score", &pd_score));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(pd_score->shape()),
                errors::InvalidArgument("score must be a vector, but get ",
                                        pd_score->shape().DebugString()));
    OP_REQUIRES(ctx, pd_score->dim_size(0) == m,
                errors::InvalidArgument("score shape mismatch, get ",
                                        pd_score->shape().DebugString()));

    const Tensor* pd_ignore = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("prediction_ignore", &pd_ignore));
    OP_REQUIRES(ctx, pd_imageid->shape().IsSameSize(pd_ignore->shape()),
                errors::InvalidArgument("pt_ignore shape mismatch: ",
                                        pd_ignore->shape().DebugString()));

    const Tensor* iou_threshold = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("iou_threshold", &iou_threshold));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(iou_threshold->shape()),
        errors::InvalidArgument("iou_threshold must be a scalar, but get ",
                                iou_threshold->shape().DebugString()));

    std::vector<image::Detection<box::Upright3DBox>> prediction;
    std::vector<box::Upright3DBox> prediction_boxes =
        box::ParseBoxesFromTensor(*pd_bbox);

    prediction.reserve(m);
    for (int i = 0; i < m; ++i) {
      image::Detection<box::Upright3DBox> p;
      p.difficult = false;
      p.imgid = pd_imageid->flat<int32>()(i);
      p.score = pd_score->flat<float>()(i);
      p.box = prediction_boxes[i];
      p.ignore = static_cast<image::IgnoreType>(pd_ignore->flat<int32>()(i));
      prediction.push_back(p);
    }

    // Prediction score and binary indicator whether the prediction is a
    // positive hit for each prediction for the specified IOU threshold.
    Tensor out_score_and_hit(DT_FLOAT, {m, 2});
    auto t_out_score_and_hit = out_score_and_hit.matrix<float>();
    std::vector<float> is_hit;
    is_hit.reserve(m);
    std::vector<float> score;
    score.reserve(m);

    Tensor out_ap(DT_FLOAT, {});
    Tensor out_pr(DT_FLOAT, {num_recall_points_, 2});
    auto t_out_pr = out_pr.matrix<float>();
    image::AveragePrecision<box::Upright3DBox>::Options opts;
    opts.iou_threshold = iou_threshold->scalar<float>()();
    opts.num_recall_points = num_recall_points_;
    std::vector<image::PR> pr;
    if (ap_algorithm_ == APAlgorithm::kKITTI) {
      out_ap.scalar<float>()() =
          image::AveragePrecision<box::Upright3DBox>(opts).FromBoxesKITTI(
              groundtruth, prediction, &pr, &is_hit, &score);
    } else {
      out_ap.scalar<float>()() =
          image::AveragePrecision<box::Upright3DBox>(opts).FromBoxes(
              groundtruth, prediction, &pr);
      // TODO(shlens): Potentially implement this statistic for the VOC analysis
      // as well. In the mean time, we just insert dummy values.
      for (int i = 0; i < prediction.size(); ++i) {
        score.push_back(-1.0);
        is_hit.push_back(-1.0);
      }
    }
    // Save out the detection score and the is_hit binary indicator.
    CHECK_EQ(prediction.size(), is_hit.size());
    for (int i = 0; i < prediction.size(); ++i) {
      t_out_score_and_hit(i, 0) = static_cast<float>(score[i]);
      t_out_score_and_hit(i, 1) = static_cast<float>(is_hit[i]);
    }

    if (pr.size() < num_recall_points_) {
      LOG(WARNING) << "PR array size smaller than expected, expect: "
                   << num_recall_points_ << ", got " << pr.size();
    }
    for (int i = 0; i < pr.size() && i < num_recall_points_; ++i) {
      t_out_pr(i, 0) = pr[i].p;
      t_out_pr(i, 1) = pr[i].r;
    }
    for (int i = pr.size(); i < num_recall_points_; ++i) {
      t_out_pr(i, 0) = 0;
      t_out_pr(i, 1) = i / static_cast<float>(num_recall_points_ - 1);
    }
    ctx->set_output(0, out_ap);
    ctx->set_output(1, out_pr);
    ctx->set_output(2, out_score_and_hit);
  }

 private:
  int num_recall_points_ = -1;
  APAlgorithm ap_algorithm_;
};

REGISTER_KERNEL_BUILDER(Name("AveragePrecision3D").Device(DEVICE_CPU), AP3DOp);

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
