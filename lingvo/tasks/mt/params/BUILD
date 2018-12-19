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
    deps = [
        "//lingvo:model_registry",
        "//lingvo/core:attention",
        "//lingvo/core:base_model_params",
        "//lingvo/core:layers",
        "//lingvo/core:lr_schedule",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:rnn_layers",
        "//lingvo/tasks/mt:decoder",
        "//lingvo/tasks/mt:encoder",
        "//lingvo/tasks/mt:input_generator",
        "//lingvo/tasks/mt:model",
    ],
)
