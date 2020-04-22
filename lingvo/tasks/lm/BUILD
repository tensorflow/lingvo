load(
    "//lingvo:lingvo.bzl",
    "lingvo_cuda_py_test",
)

# Language modelling related tasks.
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:generic_input",
        "//lingvo/core:py_utils",
        "//lingvo/core:tokenizers",
    ],
)

py_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    data = [
        "//lingvo/tasks/lm/testdata:lm1b_100",
    ],
    python_version = "PY3",
    deps = [
        ":input_generator",
        "//lingvo:compat",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:layers_with_attention",
        "//lingvo/core:layers_with_gpipe",
        "//lingvo/core:py_utils",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:rnn_layers",
        # Implicit six dependency.
    ],
)

lingvo_cuda_py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    python_version = "PY3",
    shard_count = 10,
    tags = ["noasan"],
    deps = [
        ":layers",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:base_model",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
        # Implicit six dependency.
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    data = [
        "//lingvo/tasks/lm/testdata:lm1b_100",
        "//lingvo/tasks/lm/testdata:small_word_vocab",
    ],
    python_version = "PY3",
    deps = [
        ":input_generator",
        ":model",
        "//lingvo:compat",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
        "//lingvo/core:tokenizers",
        # Implicit six dependency.
    ],
)
