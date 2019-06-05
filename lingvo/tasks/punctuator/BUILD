load("//lingvo:lingvo.bzl", "py2and3_test")

# Punctuator tutorial task.
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    srcs_version = "PY2AND3",
    deps = [
        # Implicit tensorflow dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_layer",
        "//lingvo/core:generic_input",
        "//lingvo/core:py_utils",
        "//lingvo/core:tokenizers",
    ],
)

py2and3_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    data = [
        "//lingvo/tasks/lm/testdata:lm1b_100",
        "//lingvo/tasks/punctuator/params:brown_corpus_wpm_data",
    ],
    deps = [
        ":input_generator",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:py_utils",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY2AND3",
    deps = [
        # Implicit tensorflow dependency.
        "//lingvo/core:py_utils",
        "//lingvo/tasks/mt:model",
    ],
)

py2and3_test(
    name = "model_test",
    size = "large",
    srcs = ["model_test.py"],
    data = [
        "//lingvo/tasks/lm/testdata:lm1b_100",
        "//lingvo/tasks/punctuator/params:brown_corpus_wpm_data",
    ],
    shard_count = 4,
    deps = [
        ":input_generator",
        ":model",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)
