# Placeholder: load py_library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "mnist",
    srcs = ["mnist.py"],
    strict_deps = False,
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
    strict_deps = False,
    tags = ["keep_dep"],
    deps = [
        ":mnist",
    ],
)
