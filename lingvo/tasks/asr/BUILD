load("//lingvo:lingvo.bzl", "py2and3_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "frontend",
    srcs = ["frontend.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
    ],
)

py_test(
    name = "frontend_test",
    srcs = ["frontend_test.py"],
    data = [
        "//lingvo/tools/testdata:audio_data",
    ],
    deps = [
        ":frontend",
        # Additional FFT kernels dependency.
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_layer",
        "//lingvo/core:generic_input",
        "//lingvo/core:py_utils",
        # Implicit six dependency.
    ],
)

py_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    deps = [
        ":input_generator",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "encoder",
    srcs = ["encoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:model_helper",
        "//lingvo/core:plot",
        "//lingvo/core:py_utils",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:rnn_layers",
        "//lingvo/core:spectrum_augmenter",
        "//lingvo/core:summary_utils",
        # Implicit six dependency.
    ],
)

py_test(
    name = "encoder_test",
    srcs = ["encoder_test.py"],
    deps = [
        ":encoder",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "fusion",
    srcs = ["fusion.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/lm:layers",
    ],
)

py_library(
    name = "decoder_utils",
    srcs = ["decoder_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:symbolic",
        # Implicit six dependency.
    ],
)

py_test(
    name = "decoder_utils_test",
    srcs = ["decoder_utils_test.py"],
    deps = [
        ":decoder",
        ":decoder_utils",
        "//lingvo:compat",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:symbolic",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "contextualizer_base",
    srcs = ["contextualizer_base.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo/core:base_layer",
    ],
)

py_library(
    name = "decoder",
    srcs = ["decoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":contextualizer_base",
        ":decoder_utils",
        ":fusion",
        "//lingvo:compat",
        "//lingvo/core:attention",
        "//lingvo/core:base_decoder",
        "//lingvo/core:base_layer",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:layers",
        "//lingvo/core:plot",
        "//lingvo/core:py_utils",
        "//lingvo/core:recurrent",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:summary_utils",
        # Implicit matplotlib dependency.
        # Implicit six dependency.
    ],
)

py_test(
    name = "decoder_test",
    size = "large",
    srcs = ["decoder_test.py"],
    shard_count = 10,
    deps = [
        ":decoder",
        # Implicit python proto dependency.
        "//lingvo:compat",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        "//lingvo/core:symbolic",
        "//lingvo/core:test_utils",
        "//lingvo/core/ops:hyps_py_pb2",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "model_test_input_generator",
    srcs = ["model_test_input_generator.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:py_utils",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":decoder",
        ":decoder_utils",
        ":encoder",
        ":frontend",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:base_model",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
        "//lingvo/tools:audio_lib",
        # Implicit six dependency.
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    data = ["//lingvo/tools/testdata:audio_data"],
    shard_count = 15,
    deps = [
        ":decoder",
        ":input_generator",
        ":model",
        ":model_test_input_generator",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
        "//lingvo/core:summary_utils",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

filegroup(
    name = "wpm_files",
    srcs = ["wpm_16k_librispeech.vocab"],
)
