package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "codelab",
    srcs = ["codelab.py"],
    deps = [
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/core:lr_schedule",
        "//lingvo/core:optimizer",
        "//lingvo/tasks/mt:model",
        "//lingvo/tasks/mt/params",
        "//lingvo/tasks/punctuator:input_generator",
    ],
)

py_library(
    name = "params",
    srcs = ["__init__.py"],
    deps = [
        ":codelab",
    ],
)
