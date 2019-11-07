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

#include "lingvo/tasks/car/ops/box_util.h"

#include <algorithm>
#include <cmath>

namespace tensorflow {
namespace lingvo {
namespace box {

const double kEPS = 1e-8;

// Min,max box dimensions (length, width, height). Boxes with dimensions that
// exceed these values will have box intersections of 0.
constexpr double kMinBoxDim = 1e-3;
constexpr double kMaxBoxDim = 1e6;

// A line with the representation a*x + b*y + c = 0.
struct Line {
  double a = 0;
  double b = 0;
  double c = 0;

  Line(const Vertex& v1, const Vertex& v2)
      : a(v2.y - v1.y), b(v1.x - v2.x), c(v2.x * v1.y - v2.y * v1.x) {}

  // Computes the line value for a vertex v as a * v.x + b * v.y + c
  double LineValue(const Vertex& v) const { return a * v.x + b * v.y + c; }

  // Computes the intersection point with the other line.
  Vertex IntersectionPoint(const Line& other) const {
    const double w = a * other.b - b * other.a;
    CHECK_GT(std::fabs(w), kEPS) << "No intersection between the two lines.";
    return Vertex((b * other.c - c * other.b) / w,
                  (c * other.a - a * other.c) / w);
  }
};

// Computes the coordinates of its four vertices given a 2D rotated box,
std::vector<Vertex> ComputeBoxVertices(const double cx, const double cy,
                                       const double w, const double h,
                                       const double heading) {
  const double dxcos = (w / 2.) * std::cos(heading);
  const double dxsin = (w / 2.) * std::sin(heading);
  const double dycos = (h / 2.) * std::cos(heading);
  const double dysin = (h / 2.) * std::sin(heading);
  return {Vertex(cx - dxcos + dysin, cy - dxsin - dycos),
          Vertex(cx + dxcos + dysin, cy + dxsin - dycos),
          Vertex(cx + dxcos - dysin, cy + dxsin + dycos),
          Vertex(cx - dxcos - dysin, cy - dxsin + dycos)};
}

// Computes the intersection points between two rotated boxes, by following:
//
// 1. Initiazlizes the current intersection points with the vertices of one box,
//    and the other box is taken as the cutting box;
//
// 2. For each cutting line in the cutting box (four cutting lines in total):
//      For each point in the current intersection points:
//        If the point is inside of the cutting line:
//          Adds it to the new intersection points;
//        if current point and its next point are in the opposite side of the
//          cutting line:
//          Computes the line of current points and its next point as tmp_line;
//          Computes the intersection point between the cutting line and
//            tmp_line;
//          Adds the intersection point to the new intersection points;
//      After checking each cutting line, sets current intersection points as
//      new intersection points;
//
// 3. Returns the final intersection points.
std::vector<Vertex> ComputeIntersectionPoints(
    const std::vector<Vertex>& rbox_1, const std::vector<Vertex>& rbox_2) {
  std::vector<Vertex> intersection = rbox_1;
  const int vertices_len = rbox_2.size();
  for (int i = 0; i < rbox_2.size(); ++i) {
    const int len = intersection.size();
    if (len <= 2) {
      break;
    }
    const Vertex& p = rbox_2[i];
    const Vertex& q = rbox_2[(i + 1) % vertices_len];
    Line cutting_line(p, q);
    // Computes line value.
    std::vector<double> line_values;
    line_values.reserve(len);
    for (int j = 0; j < len; ++j) {
      line_values.push_back(cutting_line.LineValue(intersection[j]));
    }
    // Updates current intersection points.
    std::vector<Vertex> new_intersection;
    for (int j = 0; j < len; ++j) {
      const double s_val = line_values[j];
      const Vertex& s = intersection[j];
      // Adds the current vertex.
      if (s_val <= 0 || std::fabs(s_val) <= kEPS) {
        new_intersection.push_back(s);
      }
      const double t_val = line_values[(j + 1) % len];
      // Skips the checking of intersection point if the next vertex is on the
      // line.
      if (std::fabs(t_val) <= kEPS) {
        continue;
      }
      // Adds the intersection point.
      if ((s_val > 0 && t_val < 0) || (s_val < 0 && t_val > 0)) {
        Line s_t_line(s, intersection[(j + 1) % len]);
        new_intersection.push_back(cutting_line.IntersectionPoint(s_t_line));
      }
    }
    intersection = new_intersection;
  }
  return intersection;
}

// Computes the area of a convex polygon,
double ComputePolygonArea(const std::vector<Vertex>& convex_polygon) {
  const int len = convex_polygon.size();
  if (len <= 2) {
    return 0;
  }
  double area = 0;
  for (int i = 0; i < len; ++i) {
    const Vertex& p = convex_polygon[i];
    const Vertex& q = convex_polygon[(i + 1) % len];
    area += p.x * q.y - p.y * q.x;
  }
  return std::fabs(0.5 * area);
}

RotatedBox2D::RotatedBox2D(const double cx, const double cy, const double w,
                           const double h, const double heading)
    : cx_(cx), cy_(cy), w_(w), h_(h), heading_(heading) {
  // Compute loose bounds on dimensions of box that doesn't require computing
  // full intersection.  We can do this by trying to compute the largest circle
  // swept by rotating the box around its center.  The radius of that circle
  // is the length of the ray from the center to the box corner.  The upper
  // bound for this value is the length of the longer dimension divided by two
  // and then multiplied by root(2) (worst-case being a square box); we choose
  // 1.5 as slightly higher than root(2), and then use these extrema to do
  // simple extrema box checks without having to compute the true cos/sin value.
  double max_dim = std::max(w_, h_) / 2. * 1.5;
  loose_min_x_ = cx_ - max_dim;
  loose_max_x_ = cx_ + max_dim;
  loose_min_y_ = cy_ - max_dim;
  loose_max_y_ = cy_ + max_dim;

  extreme_box_dim_ = (w_ <= kMinBoxDim || h_ <= kMinBoxDim);
  extreme_box_dim_ |= (w_ >= kMaxBoxDim || h_ >= kMaxBoxDim);
}

double RotatedBox2D::Area() const {
  if (area_ < 0) {
    const double area = ComputePolygonArea(box_vertices());
    area_ = std::fabs(area) <= kEPS ? 0 : area;
  }
  return area_;
}

const std::vector<Vertex>& RotatedBox2D::box_vertices() const {
  if (box_vertices_.empty()) {
    box_vertices_ = ComputeBoxVertices(cx_, cy_, w_, h_, heading_);
  }

  return box_vertices_;
}

bool RotatedBox2D::NonZeroAndValid() const { return !extreme_box_dim_; }

bool RotatedBox2D::MaybeIntersects(const RotatedBox2D& other) const {
  // If the box dimensions of either box are too small / large,
  // assume they are not well-formed boxes (otherwise we are
  // subject to issues due to catastrophic cancellation).
  if (extreme_box_dim_ || other.extreme_box_dim_) {
    return false;
  }

  // Check whether the loose extrema overlap -- if not, then there is
  // no chance that the two boxes overlap even when computing the true,
  // more expensive overlap.
  if ((loose_min_x_ > other.loose_max_x_) ||
      (loose_max_x_ < other.loose_min_x_) ||
      (loose_min_y_ > other.loose_max_y_) ||
      (loose_max_y_ < other.loose_min_y_)) {
    return false;
  }

  return true;
}

double RotatedBox2D::Intersection(const RotatedBox2D& other) const {
  // Do a fast intersection check - if the boxes are not near each other
  // then we can return early.  If they are close enough to maybe overlap,
  // we do the full check.
  if (!MaybeIntersects(other)) {
    return 0.0;
  }

  // Computes the intersection polygon.
  const std::vector<Vertex> intersection_polygon =
      ComputeIntersectionPoints(box_vertices(), other.box_vertices());
  // Computes the intersection area.
  const double intersection_area = ComputePolygonArea(intersection_polygon);

  return std::fabs(intersection_area) <= kEPS ? 0 : intersection_area;
}

double RotatedBox2D::IoU(const RotatedBox2D& other) const {
  // Computes the intersection area.
  const double intersection_area = Intersection(other);
  if (intersection_area == 0) {
    return 0;
  }
  // Computes the union area.
  const double union_area = Area() + other.Area() - intersection_area;
  if (std::fabs(union_area) <= kEPS) {
    return 0;
  }
  return intersection_area / union_area;
}

std::vector<Upright3DBox> ParseBoxesFromTensor(const Tensor& boxes_tensor) {
  int num_boxes = boxes_tensor.dim_size(0);

  const auto t_boxes_tensor = boxes_tensor.matrix<float>();

  std::vector<Upright3DBox> bboxes3d;
  bboxes3d.reserve(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    const double center_x = t_boxes_tensor(i, 0);
    const double center_y = t_boxes_tensor(i, 1);
    const double center_z = t_boxes_tensor(i, 2);
    const double dimension_x = t_boxes_tensor(i, 3);
    const double dimension_y = t_boxes_tensor(i, 4);
    const double dimension_z = t_boxes_tensor(i, 5);
    const double heading = t_boxes_tensor(i, 6);
    const double z_min = center_z - dimension_z / 2;
    const double z_max = center_z + dimension_z / 2;
    RotatedBox2D box2d(center_x, center_y, dimension_x, dimension_y, heading);
    if (dimension_x <= 0 || dimension_y <= 0) {
      bboxes3d.emplace_back(RotatedBox2D(), z_min, z_max);
    } else {
      bboxes3d.emplace_back(box2d, z_min, z_max);
    }
  }
  return bboxes3d;
}

bool Upright3DBox::NonZeroAndValid() const {
  // If min is larger than max, the upright box is invalid.
  //
  // If the min and max are equal, the height of the box is 0. and thus the box
  // is zero.
  if (z_min - z_max >= 0.) {
    return false;
  }

  return rbox.NonZeroAndValid();
}

double Upright3DBox::IoU(const Upright3DBox& other) const {
  // Check that both boxes are non-zero and valid.  Otherwise,
  // return 0.
  if (!NonZeroAndValid() || !other.NonZeroAndValid()) {
    return 0;
  }

  // Quickly check whether z's overlap; if they don't, we can return 0.
  const double z_inter =
      std::max(.0, std::min(z_max, other.z_max) - std::max(z_min, other.z_min));
  if (z_inter == 0) {
    return 0;
  }

  const double base_inter = rbox.Intersection(other.rbox);
  if (base_inter == 0) {
    return 0;
  }

  const double volume_1 = rbox.Area() * (z_max - z_min);
  const double volume_2 = other.rbox.Area() * (other.z_max - other.z_min);
  const double volume_inter = base_inter * z_inter;
  const double volume_union = volume_1 + volume_2 - volume_inter;
  return volume_inter > 0 ? volume_inter / volume_union : 0;
}

double Upright3DBox::Overlap(const Upright3DBox& other) const {
  // Check that both boxes are non-zero and valid.  Otherwise,
  // return 0.
  if (!NonZeroAndValid() || !other.NonZeroAndValid()) {
    return 0;
  }

  const double z_inter =
      std::max(.0, std::min(z_max, other.z_max) - std::max(z_min, other.z_min));
  if (z_inter == 0) {
    return 0;
  }

  const double base_inter = rbox.Intersection(other.rbox);
  if (base_inter == 0) {
    return 0;
  }

  const double volume_1 = rbox.Area() * (z_max - z_min);
  const double volume_inter = base_inter * z_inter;
  // Normalizes intersection of volume by the volume of this box.
  return volume_inter > 0 ? volume_inter / volume_1 : 0;
}

}  // namespace box
}  // namespace lingvo
}  // namespace tensorflow
