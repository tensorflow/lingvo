load("//lingvo:lingvo.bzl", "py2and3_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "rnn_steps",
    srcs = ["rnn_steps.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:builder_layers",
        "//lingvo/core:py_utils",
        "//lingvo/core:rnn_cell",
    ],
)

py_test(
    name = "rnn_steps_test",
    size = "small",
    srcs = ["rnn_steps_test.py"],
    deps = [
        ":rnn_steps",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:recurrent",
        "//lingvo/core:step",
        "//lingvo/core:test_utils",
    ],
)
