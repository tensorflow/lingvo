package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "mnist",
    srcs = ["mnist.py"],
    deps = [
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/core:program_lib",
        "//lingvo/tasks/image:classifier",
        "//lingvo/tasks/image:input_generator",
    ],
)

py_library(
    name = "params",
    srcs = ["params.py"],
    tags = ["keep_dep"],
    deps = [
        ":mnist",
    ],
)
