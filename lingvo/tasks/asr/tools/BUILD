# tools for ASR task, including WER computation and breakdown analytics

# Placeholder: load py_library
# Placeholder: load py_test

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "simple_wer",
    srcs = ["simple_wer.py"],
    strict_deps = False,
)

py_test(
    name = "simple_wer_test",
    srcs = ["simple_wer_test.py"],
    strict_deps = False,
    deps = [
        ":simple_wer",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "simple_wer_v2",
    srcs = ["simple_wer_v2.py"],
    strict_deps = False,
)

py_test(
    name = "simple_wer_v2_test",
    srcs = ["simple_wer_v2_test.py"],
    strict_deps = False,
    deps = [
        ":custom_html_handlers",
        ":simple_wer_v2",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "custom_html_handlers",
    srcs = ["custom_html_handlers.py"],
    strict_deps = False,
    deps = [":simple_wer_v2"],
)
