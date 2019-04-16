# tools for ASR task, including WER computation and breakdown analytics

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "simple_wer",
    srcs = ["simple_wer.py"],
)

py_test(
    name = "simple_wer_test",
    srcs = ["simple_wer_test.py"],
    deps = [
        ":simple_wer",
        # Implicit tensorflow dependency.
        "//lingvo/core:test_utils",
    ],
)
