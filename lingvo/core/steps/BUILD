package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "attention_steps",
    srcs = ["attention_steps.py"],
    srcs_version = "PY3",
    deps = [
        ":rnn_steps",
        "//lingvo:compat",
        "//lingvo/core:attention",
        "//lingvo/core:py_utils",
        "//lingvo/core:step",
    ],
)

py_test(
    name = "attention_steps_test",
    size = "small",
    srcs = ["attention_steps_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":attention_steps",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "embedding_steps",
    srcs = ["embedding_steps.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        "//lingvo/core:step",
    ],
)

py_test(
    name = "embedding_steps_test",
    size = "small",
    srcs = ["embedding_steps_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":embedding_steps",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "rnn_steps",
    srcs = ["rnn_steps.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:step",
    ],
)

py_test(
    name = "rnn_steps_test",
    size = "small",
    srcs = ["rnn_steps_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":rnn_steps",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:step",
        "//lingvo/core:test_utils",
    ],
)
