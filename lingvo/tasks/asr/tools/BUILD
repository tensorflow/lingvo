# tools for ASR task, including WER computation and breakdown analytics

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "simple_wer",
    srcs = ["simple_wer.py"],
)

py_test(
    name = "simple_wer_test",
    srcs = ["simple_wer_test.py"],
    deps = [
        ":simple_wer",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "simple_wer_v2",
    srcs = ["simple_wer_v2.py"],
)

py_test(
    name = "simple_wer_v2_test",
    srcs = ["simple_wer_v2_test.py"],
    deps = [
        ":custom_html_handlers",
        ":simple_wer_v2",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "custom_html_handlers",
    srcs = ["custom_html_handlers.py"],
    deps = [":simple_wer_v2"],
)
