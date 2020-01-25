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

#ifndef THIRD_PARTY_PY_LINGVO_TASKS_CAR_OPS_BOX_UTIL_H_
#define THIRD_PARTY_PY_LINGVO_TASKS_CAR_OPS_BOX_UTIL_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace lingvo {
namespace box {

struct Vertex;

// A rotated 2D bounding box represented as (cx, cy, w, h, r). cx, cy are the
// box center coordinates; w, h are the box width and height; heading is the
// rotation angle in radian relative to the 'positive x' direction.
class RotatedBox2D {
 public:
  // Creates an empty rotated 2D box.
  RotatedBox2D() : RotatedBox2D(0, 0, 0, 0, 0) {}

  RotatedBox2D(const double cx, const double cy, const double w, const double h,
               const double heading);

  // Returns the area of the box.
  double Area() const;

  // Returns the intersection area between this box and the given box.
  double Intersection(const RotatedBox2D& other) const;

  // Returns the IoU between this box and the given box.
  double IoU(const RotatedBox2D& other) const;

  // Returns true if the box is valid (width and height are not extremely
  // large or small).
  bool NonZeroAndValid() const;

 private:
  // Computes / caches box_vertices_ calculation.
  const std::vector<Vertex>& box_vertices() const;

  // Returns true if this box and 'other' might intersect.
  //
  // If this returns false, the two boxes definitely do not intersect.  If this
  // returns true, it is still possible that the two boxes do not intersect, and
  // the more expensive intersection code will be called.
  bool MaybeIntersects(const RotatedBox2D& other) const;

  double cx_ = 0;
  double cy_ = 0;
  double w_ = 0;
  double h_ = 0;
  double heading_ = 0;

  // Loose boundaries for fast intersection test.
  double loose_min_x_ = -1;
  double loose_max_x_ = -1;
  double loose_min_y_ = -1;
  double loose_max_y_ = -1;

  // True if the dimensions of the box are very small or very large in any
  // dimension.
  bool extreme_box_dim_ = false;

  // The following fields are computed on demand.  They are logically
  // const.

  // Cached area.  Access via Area() public API.
  mutable double area_ = -1;

  // Stores the vertices of the box.  Access via box_vertices().
  mutable std::vector<Vertex> box_vertices_;
};

// A 3D box of 7-DOFs: only allows rotation around the z-axis.
struct Upright3DBox {
  RotatedBox2D rbox = RotatedBox2D();
  double z_min = 0;
  double z_max = 0;

  // Creates an empty rotated 3D box.
  Upright3DBox() = default;

  // Creates a 3D box from the raw input data with size 7. The data format is
  // (center_x, center_y, center_z, dimension_x, dimension_y, dimension_z,
  // heading)
  Upright3DBox(const std::vector<double>& raw)
      : rbox(raw[0], raw[1], raw[3], raw[4], raw[6]),
        z_min(raw[2] - raw[5] / 2.0),
        z_max(raw[2] + raw[5] / 2.0) {}

  Upright3DBox(const RotatedBox2D& rb, const double z_min, const double z_max)
      : rbox(rb), z_min(z_min), z_max(z_max) {}

  // Computes intersection over union (of the volume).
  double IoU(const Upright3DBox& other) const;

  // Computes overlap: intersection of this box and the given box normalized
  // over the volume of this box.
  double Overlap(const Upright3DBox& other) const;

  // Returns true if the box is valid (width and height are not extremely
  // large or small, and zmin < zmax).
  bool NonZeroAndValid() const;
};

// Converts a [N, 7] tensor to a vector of N Upright3DBox objects.
std::vector<Upright3DBox> ParseBoxesFromTensor(const Tensor& boxes_tensor);

// A vertex with (x, y) coordinate.
//
// This is an internal implementation detail of RotatedBox2D.
struct Vertex {
  double x = 0;
  double y = 0;

  // Creates an empty Vertex.
  Vertex() = default;

  Vertex(const double x, const double y) : x(x), y(y) {}
};

}  // namespace box
}  // namespace lingvo
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_LINGVO_TASKS_CAR_OPS_BOX_UTIL_H_
