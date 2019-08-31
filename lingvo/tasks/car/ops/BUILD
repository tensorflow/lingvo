package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//lingvo:lingvo.bzl", "py2and3_test")
load(
    "//lingvo:lingvo.bzl",
    "custom_kernel_library",
    "gen_op_cclib",
    "gen_op_pylib",
    "lingvo_cc_library",
)

lingvo_cc_library(
    name = "box_util",
    srcs = ["box_util.cc"],
    hdrs = ["box_util.h"],
)

lingvo_cc_library(
    name = "image_metrics",
    srcs = ["image_metrics.cc"],
    hdrs = ["image_metrics.h"],
)

# Op definitions to be used from python.
gen_op_cclib(
    name = "car_ops",
    srcs = ["car_ops.cc"],
    deps = [],
)

gen_op_pylib(
    name = "py_car_ops",
    srcs = ["__init__.py"],
    cc_lib_name = "car_ops",
    kernel_deps = [
        ":op_kernels",
    ],
    py_deps = [
        "//lingvo:compat",
    ],
)

py_library(
    name = "ops",
    srcs = ["__init__.py"],
    deps = [
        ":py_car_ops",  # buildcleaner: keep
        "//lingvo:compat",
    ],
)

# Op kernel deps.
cc_library(
    name = "op_kernels",
    deps = [
        ":average_precision_3d_op",
        ":nms_3d_op",
        ":pairwise_iou_op",
        ":point_grid_op",
    ],
)

custom_kernel_library(
    name = "pairwise_iou_op",
    srcs = ["pairwise_iou_op.cc"],
    op_def_lib = [":car_ops"],
    deps = [
        ":box_util",
    ],
)

custom_kernel_library(
    name = "point_grid_op",
    srcs = ["point_grid_op.cc"],
    op_def_lib = [":car_ops"],
)

custom_kernel_library(
    name = "average_precision_3d_op",
    srcs = ["average_precision_3d_op.cc"],
    op_def_lib = [":car_ops"],
    deps = [
        ":box_util",
        ":image_metrics",
    ],
)

py_test(
    name = "average_precision_3d_op_test",
    srcs = ["average_precision_3d_op_test.py"],
    deps = [
        ":ops",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "point_grid_op_test",
    srcs = ["point_grid_op_test.py"],
    deps = [
        ":ops",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

custom_kernel_library(
    name = "nms_3d_op",
    srcs = ["nms_3d_op.cc"],
    op_def_lib = [":car_ops"],
    deps = [
        ":box_util",
    ],
)

py_test(
    name = "nms_3d_op_test",
    srcs = ["nms_3d_op_test.py"],
    deps = [
        ":ops",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)
