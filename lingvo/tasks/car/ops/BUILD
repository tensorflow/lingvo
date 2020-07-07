load(
    "//lingvo:lingvo.bzl",
    "custom_kernel_library",
    "gen_op_cclib",
    "gen_op_pylib",
    "lingvo_cc_library",
    "lingvo_cc_test",
)

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

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
    srcs_version = "PY3",
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
        ":sampling_ops",
    ],
)

custom_kernel_library(
    name = "sampling_ops",
    srcs = ["sampling_ops.cc"],
    op_def_lib = [":car_ops"],
    deps = [
        ":ps_utils",
    ],
)

py_test(
    name = "sampling_ops_test",
    srcs = ["sampling_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

lingvo_cc_library(
    name = "ps_utils",
    srcs = ["ps_utils.cc"],
    hdrs = ["ps_utils.h"],
    deps = [
        # Implicit absl.synchronization dependency.
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
    python_version = "PY3",
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

lingvo_cc_test(
    name = "ps_utils_test",
    srcs = ["ps_utils_test.cc"],
    deps = [
        ":ps_utils",
    ],
)
