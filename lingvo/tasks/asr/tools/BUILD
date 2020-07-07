# tools for ASR task, including WER computation and breakdown analytics

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "simple_wer",
    srcs = ["simple_wer.py"],
    srcs_version = "PY3",
)

py_test(
    name = "simple_wer_test",
    srcs = ["simple_wer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":simple_wer",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "simple_wer_v2",
    srcs = ["simple_wer_v2.py"],
    srcs_version = "PY3",
)

py_test(
    name = "simple_wer_v2_test",
    srcs = ["simple_wer_v2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":simple_wer_v2",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)
