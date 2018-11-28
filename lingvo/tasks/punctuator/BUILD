# Punctuator tutorial task.
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//lingvo:lingvo.bzl",
    "lingvo_cuda_py_test",
)

py_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    deps = [
        # Implicit tensorflow dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
        "//lingvo/core:tokenizers",
        "//lingvo/core/ops:py_x_ops",
    ],
)

py_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    data = [
        "//lingvo/tasks/lm/testdata:lm1b_100",
        "//lingvo/tasks/punctuator/testdata:test_vocab",
    ],
    deps = [
        ":input_generator",
        # Implicit tensorflow dependency.
        "//lingvo/core:test_helper",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        # Implicit tensorflow dependency.
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/mt:model",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    data = [
        "//lingvo/tasks/lm/testdata:lm1b_100",
        "//lingvo/tasks/punctuator/testdata:test_vocab",
    ],
    shard_count = 4,
    deps = [
        ":input_generator",
        ":model",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:py_utils",
        "//lingvo/core:test_helper",
        "//lingvo/tasks/mt:decoder",
        "//lingvo/tasks/mt:encoder",
    ],
)
