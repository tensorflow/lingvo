package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "param_files",
    srcs = glob(
        include = [
            "*.py",
        ],
    ),
)

py_library(
    name = "params",
    srcs = [
        ":param_files",
    ],
    srcs_version = "PY3",
    tags = ["keep_dep"],
    deps = [
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/tasks/mt:base_config",
        "//lingvo/tasks/mt:input_generator",
        "//lingvo/tasks/mt:model",
    ],
)
