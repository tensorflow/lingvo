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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace {

REGISTER_OP("PairwiseIou3D")
    .Input("boxes_a: float")
    .Input("boxes_b: float")
    .Output("iou: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(
          0, c->MakeShape({c->Dim(c->input(0), 0), c->Dim(c->input(1), 0)}));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Calculate pairwise IoUs between two set of 3D bboxes. Every bbox is represented
as [center_x, center_y, center_z, dim_x, dim_y, dim_z, heading].
boxes_a: A tensor of shape [num_boxes_a, 7]
boxes_b: A tensor of shape [num_boxes_b, 7]
)doc");

REGISTER_OP("PointToGrid")
    .Input("points: float")
    .Output("output_points: float")
    .Output("grid_centers: float")
    .Output("num_points: int32")
    .Attr("num_points_per_cell: int")
    .Attr("x_intervals: int")
    .Attr("y_intervals: int")
    .Attr("z_intervals: int")
    .Attr("x_range: list(float)")
    .Attr("y_range: list(float)")
    .Attr("z_range: list(float)")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto feature_dim = c->Dim(c->input(0), 1);
      int gx, gy, gz, n;
      TF_RETURN_IF_ERROR(c->GetAttr("x_intervals", &gx));
      TF_RETURN_IF_ERROR(c->GetAttr("y_intervals", &gy));
      TF_RETURN_IF_ERROR(c->GetAttr("z_intervals", &gz));
      TF_RETURN_IF_ERROR(c->GetAttr("num_points_per_cell", &n));
      c->set_output(0, c->MakeShape({gx, gy, gz, n, feature_dim}));
      c->set_output(1, c->MakeShape({gx, gy, gz, 3}));
      c->set_output(2, c->MakeShape({gx, gy, gz}));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Re-organize input points into equally spaced grids. Points in each grid cell are
shuffled. When not enough available points, the center of each cell with all 0
on feature dimensions will be used as padding.

The number of points in each grid cell. I.e.,
output_points[i, j, k, num_points[i, j, k]\:, \:] are padded points.

points: [n, d]. d >= 3 and the first 3 dimensions are treated as x,y,z.
output_points:
  [x_intervals, y_intervals, z_intervals, num_per_grid, d].
grid_centers:
  [x_intervals, y_intervals, z_intervals, 3]. Grid cell centers.
num_points:
  [x_intervals, y_intervals, z_intervals].
num_points_per_cell: int. Number of points to keep in each cell.
x_intervals: int. Number of cells along x-axis.
y_intervals: int. Number of cells along y-axis.
z_intervals: int. Number of cells along z-axis.
x_range: tuple of two scalars\: (xmin, xmax). Spatial span of the grid.
y_range: tuple of two scalars\: (ymin, ymax). Spatial span of the grid.
z_range: tuple of two scalars\: (zmin, zmax). Spatial span of the grid.
)doc");

REGISTER_OP("NonMaxSuppression3D")
    .Input("bboxes: float")
    .Input("scores: float")
    .Input("nms_iou_threshold: float")
    .Input("score_threshold: float")
    .Output("bbox_indices: int32")
    .Output("bbox_scores: float")
    .Output("valid_mask: float")
    .Attr("max_boxes_per_class: int")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle bboxes_shape;
      shape_inference::ShapeHandle scores_shape;
      shape_inference::ShapeHandle iou_threshold_shape;
      shape_inference::ShapeHandle score_threshold_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &bboxes_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &scores_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &iou_threshold_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &score_threshold_shape));

      int max_boxes_per_class;
      TF_RETURN_IF_ERROR(
          c->GetAttr("max_boxes_per_class", &max_boxes_per_class));
      auto num_classes = c->Dim(c->input(1), 1);

      shape_inference::ShapeHandle output_shape =
          c->MakeShape({num_classes, max_boxes_per_class});
      c->set_output(0, output_shape);
      c->set_output(1, output_shape);
      c->set_output(2, output_shape);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Greedily selects the top subset of 3D (7 DOF format) bounding boxes per class.

This implementation is rotation and class aware, and for each class takes the
best boxes that are above our score_threshold and also don't overlap more than
our nms_iou_threshold with any better scoring boxes.

bboxes: A tf.float32 Tensor of shape [num_bboxes, 7] where the box is of
  format [center_x, center_y, center_z, dim_x, dim_y, dim_z, heading].
scores: A tf.float32 Tensor of shape [num_bboxes, num_classes] with a score
  per box for each class.
nms_iou_threshold: A tf.float32 Tensor of shape [num_classes] specifying the
  max overlap between two boxes we allow before saying these boxes overlap,
  and suppressing one of them.
score_threshold: A tf.float32 Tensor of shape [num_classes] specifying the
  minimum class score (per class) a box can have before it is removed.
max_boxes_per_class: An integer specifying how many (at most) boxes to
  return for each class.
bbox_indices:
  [num_classes, max_boxes_per_class] with the indices of selected boxes
  for each class.
bbox_scores:
  [num_classes, max_boxes_per_class] with the score of selected boxes
  for each class.
valid_mask:
  [num_classes, max_boxes_per_class] with a 1 for a valid box and a 0
  for invalid boxes for each class.
)doc");

REGISTER_OP("AveragePrecision3D")
    .Input("iou_threshold: float")
    .Input("groundtruth_bbox: float")
    .Input("groundtruth_imageid: int32")
    .Input("groundtruth_ignore: int32")
    .Input("prediction_bbox: float")
    .Input("prediction_imageid: int32")
    .Input("prediction_ignore: int32")
    .Input("prediction_score: float")
    .Output("average_precision: float")
    .Output("precision_recall: float")
    .Output("score_and_hit: float")
    .Attr("num_recall_points: int >= 1 = 1")
    .Attr("algorithm: string = \"KITTI\"")
    .Doc(R"doc(
Computes average precision for 3D bounding boxes.

The output PR is sorted by recall in descending order. When there isn't enough
data for num_recall_points + 1 sample points, this tensor will be zero-padded.

iou_threshold: IoU threshold.
groundtruth_bbox: [N, 7]. N ground truth bounding boxes.
groundtruth_imageid: [N]. N image ids for ground truth bounding boxes.
groundtruth_ignore: [N]. Valid values are 0 - Don't ignore; 1 - Ignore the
  first match; 2 - Ignore all matches.
prediction_bbox: [M, 7]. M predicted bounding boxes.
prediction_imageid: [M]. M image ids for the predicted bounding boxes.
prediction_score: [M]. M scores for each predicted bounding box.
prediction_ignore: [N]. The ignore types for predictions. Currently only used by
  the KITTI AP. Valid values are 0 - Don't ignore; 1 - Ignore the first match.
average_precision:
  A scalar. The AP metric.
precision_recall:
  [num_recall_points, 2]. List of PR points.
score_and_hit:
  [M, 2] Prediction score and corresponding binary indication whether a
  prediction detected a ground truth item.
algorithm: string. One of ["KITTI", "VOC"]. See this paper "Supervised
  learning and evaluation of KITTI's cars detector with DPM", Section III.A for
  the differences between KITTI AP and VOC AP.
)doc");

// TODO(vrv): Convert some of the attrs to inputs.
REGISTER_OP("SamplePoints")
    .Input("points: float")
    .Input("points_padding: float")
    .Input("num_seeded_points: int32")
    .Output("center: int32")
    .Output("center_padding: float")
    .Output("indices: int32")
    .Output("indices_padding: float")
    .Attr("center_selector: string")
    .Attr("neighbor_sampler: string")
    .Attr("neighbor_algorithm: string = \"auto\"")
    .Attr("num_centers: int")
    .Attr("center_z_min: float")
    .Attr("center_z_max: float")
    .Attr("num_neighbors: int")
    .Attr("max_distance: float")
    .Attr("random_seed: int = -1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle points;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &points));
      shape_inference::DimensionHandle batch_size = c->Dim(points, 0);

      int num_centers, num_neighbors;
      TF_RETURN_IF_ERROR(c->GetAttr("num_centers", &num_centers));
      TF_RETURN_IF_ERROR(c->GetAttr("num_neighbors", &num_neighbors));
      c->set_output(0, c->MakeShape({batch_size, num_centers}));
      c->set_output(1, c->MakeShape({batch_size, num_centers}));
      c->set_output(2, c->MakeShape({batch_size, num_centers, num_neighbors}));
      c->set_output(3, c->MakeShape({batch_size, num_centers, num_neighbors}));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Sample points among 'points'.

points: [B, N, K]. B is the batch size; N is the number of points; K is the
  number of dimensions of each point.
points_padding: [B, N]. 0/1 padding of points. If points_padding[b, i] is 0.,
  points[b, i, :] are valid point coordinates. Otherwise, point[b, i, :] are
  all zeros.
num_seeded_points: If num_seeded_points > 0, then the first
  num_seeded_points in points are considered to be seeded in the FPS
  sampling. Note that we assume that these points are *not* padded, and do
  not check padding when seeding them.
center:
  [B, M]. the indices of selected centers.
center_padding:
  [B, M]. If center_padding[b, i] is 0., center[b, i],
  indices[b, i, :] and indices_padding[b, i, :] are valid sampled center.
  Otherwise, center_padding[b, i] is 1.0, and center[b, i] and indices[b, i, :]
  are all zeros while indices_padding[b, i, :] are all 1.0.
indices:
  [B, M, P]. the indices of selected points.
indices_padding:
  [B, M, P]. 0/1 padding of indices.
center_selector: Valid options - 'farthest', 'uniform'.
neighbor_sampler: Valid options - 'uniform', 'closest'.
neighbor_algorithm: Valid options - 'auto', 'hash'.
num_centers: The number of centers to sample for each batch example (M).
center_z_min: Points with z less than center_z_min are not considered for
 center selection.
center_z_max: Points with z greater than center_z_max are not considered
 for center selection.
num_neighbors: Sample these many points within the neighborhood (P).
max_distance: Points with L2 distances from a center larger than this
  threshold are not considered to be in the neighborhood.
random_seed: The random seed.
)doc");

}  // namespace
}  // namespace tensorflow
