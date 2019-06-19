# tools for ASR task, including WER computation and breakdown analytics

load("//lingvo:lingvo.bzl", "py2and3_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "simple_wer",
    srcs = ["simple_wer.py"],
    srcs_version = "PY2AND3",
    deps = [
        # Implicit six dependency.
    ],
)

py_test(
    name = "simple_wer_test",
    srcs = ["simple_wer_test.py"],
    deps = [
        ":simple_wer",
        "//lingvo/core:test_utils",
        # Implicit tensorflow dependency.
    ],
)
