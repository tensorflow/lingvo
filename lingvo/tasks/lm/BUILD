# Language modelling related tasks.
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
    ],
    deps = [
        ":input_generator",
        # Implicit tensorflow dependency.
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    deps = [
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:layers_with_attention",
        "//lingvo/core:layers_with_gpipe",
        "//lingvo/core:py_utils",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:rnn_layers",
    ],
)

lingvo_cuda_py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    shard_count = 10,
    tags = ["noasan"],
    deps = [
        ":layers",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":layers",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_layer",
        "//lingvo/core:base_model",
        "//lingvo/core:lr_schedule",
        "//lingvo/core:py_utils",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    data = [
        "//lingvo/tasks/lm/testdata:lm1b_100",
        "//lingvo/tasks/lm/testdata:small_word_vocab",
    ],
    deps = [
        ":input_generator",
        ":model",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
        "//lingvo/core:tokenizers",
    ],
)
