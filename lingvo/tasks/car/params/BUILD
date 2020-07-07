package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "kitti",
    srcs = ["kitti.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:datasource",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/car:input_preprocessors",
        "//lingvo/tasks/car:kitti_input_generator",
        "//lingvo/tasks/car:lr_util",
        "//lingvo/tasks/car:starnet",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "waymo",
    srcs = ["waymo.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:hyperparams",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/car:input_preprocessors",
        "//lingvo/tasks/car:lr_util",
        "//lingvo/tasks/car:starnet",
        "//lingvo/tasks/car/waymo:waymo_decoder",
        "//lingvo/tasks/car/waymo:waymo_metadata",
        "//lingvo/tasks/car/waymo:waymo_open_input_generator",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "params",
    srcs = ["params.py"],
    srcs_version = "PY3",
    tags = ["keep_dep"],
    deps = [
        ":kitti",
        ":waymo",
    ],
)
