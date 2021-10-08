load("//lingvo:lingvo.bzl", "pytype_strict_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "frontend",
    srcs = ["frontend.py"],
    srcs_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:generic_input",
        "//lingvo/core:py_utils",
    ],
)

py_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":input_generator",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "encoder",
    srcs = ["encoder.py"],
    srcs_version = "PY3",
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
    ],
)

py_test(
    name = "encoder_test",
    srcs = ["encoder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":encoder",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "eos_normalization",
    srcs = ["eos_normalization.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "eos_normalization_test",
    srcs = ["eos_normalization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eos_normalization",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "fusion",
    srcs = ["fusion.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:symbolic",
        "//lingvo/tasks/asr:levenshtein_distance",
        # Implicit six dependency.
    ],
)

py_test(
    name = "decoder_utils_test",
    srcs = ["decoder_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        "//lingvo/core:base_layer",
    ],
)

py_library(
    name = "decoder",
    srcs = ["decoder.py"],
    srcs_version = "PY3",
    deps = [
        ":contextualizer_base",
        ":decoder_utils",
        ":fusion",
        "//lingvo:compat",
        "//lingvo/core:attention",
        "//lingvo/core:base_decoder",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:layers",
        "//lingvo/core:plot",
        "//lingvo/core:py_utils",
        "//lingvo/core:recurrent",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:summary_utils",
        "//lingvo/core:symbolic",
        # Implicit matplotlib dependency.
    ],
)

py_test(
    name = "decoder_test",
    size = "large",
    srcs = ["decoder_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
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
    ],
)

py_library(
    name = "model_test_input_generator",
    srcs = ["model_test_input_generator.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:py_utils",
    ],
)

pytype_strict_library(
    name = "decoder_metrics",
    srcs = ["decoder_metrics.py"],
    srcs_version = "PY3",
    deps = [
        ":decoder_utils",
        ":eos_normalization",
        ":metrics_calculator",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
    deps = [
        ":decoder",
        ":decoder_metrics",
        ":encoder",
        ":frontend",
        "//lingvo:compat",
        "//lingvo/core:base_model",
        "//lingvo/core:program_lib",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
        "//lingvo/tools:audio_lib",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    data = ["//lingvo/tools/testdata:audio_data"],
    python_version = "PY3",
    shard_count = 15,
    srcs_version = "PY3",
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
    ],
)

filegroup(
    name = "wpm_files",
    srcs = ["wpm_16k_librispeech.vocab"],
)

pytype_strict_library(
    name = "levenshtein_distance",
    srcs = ["levenshtein_distance.py"],
    srcs_version = "PY3",
)

pytype_strict_library(
    name = "metrics_calculator",
    srcs = ["metrics_calculator.py"],
    srcs_version = "PY3",
    deps = [
        ":decoder_utils",
        "//lingvo:compat",
    ],
)
