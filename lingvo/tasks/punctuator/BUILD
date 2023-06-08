# Punctuator tutorial task.
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:datasource",
        "//lingvo/core:py_utils",
        "//lingvo/core:tokenizers",
    ],
)

py_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    data = [
        "//lingvo/tasks/lm/testdata:lm1b_100",
        "//lingvo/tasks/punctuator/params:brown_corpus_wpm_data",
    ],
    shard_count = 3,
    deps = [
        ":input_generator",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/mt:model",
    ],
)

py_test(
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
        "//lingvo:compat",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)
