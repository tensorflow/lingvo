package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "kitti",
    srcs = ["kitti.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/car:input_preprocessors",
        "//lingvo/tasks/car:kitti_input_generator",
        "//lingvo/tasks/car:lr_util",
        "//lingvo/tasks/car:starnet",
    ],
)

py_library(
    name = "params",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":kitti",
    ],
)
