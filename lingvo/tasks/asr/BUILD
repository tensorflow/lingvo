package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "frontend",
    srcs = ["frontend.py"],
    deps = [
        # Implicit tensorflow dependency.
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
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:py_utils",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    deps = [
        # Implicit six dependency.
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
    deps = [
        ":input_generator",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "encoder",
    srcs = ["encoder.py"],
    deps = [
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_encoder",
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:model_helper",
        "//lingvo/core:plot",
        "//lingvo/core:py_utils",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:summary_utils",
    ],
)

py_test(
    name = "encoder_test",
    srcs = ["encoder_test.py"],
    deps = [
        ":encoder",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "fusion",
    srcs = ["fusion.py"],
    deps = [
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/lm:layers",
    ],
)

py_library(
    name = "decoder_utils",
    srcs = ["decoder_utils.py"],
    deps = [
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:py_utils",
    ],
)

py_test(
    name = "decoder_utils_test",
    srcs = ["decoder_utils_test.py"],
    deps = [
        ":decoder_utils",
        # Implicit tensorflow dependency.
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "contextualizer_base",
    srcs = ["contextualizer_base.py"],
    deps = [
        "//lingvo/core:base_layer",
    ],
)

py_library(
    name = "decoder",
    srcs = ["decoder.py"],
    deps = [
        ":contextualizer_base",
        ":decoder_utils",
        ":fusion",
        # Implicit matplotlib dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
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
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:cluster_factory",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        "//lingvo/core/ops:hyps_py_pb2",
    ],
)

py_library(
    name = "model_test_input_generator",
    srcs = ["model_test_input_generator.py"],
    deps = [
        # Implicit tensorflow dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:py_utils",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":decoder",
        ":decoder_utils",
        ":encoder",
        ":frontend",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_layer",
        "//lingvo/core:base_model",
        "//lingvo/core:lr_schedule",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
        "//lingvo/tools:audio_lib",
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
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_layer",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:lr_schedule",
        "//lingvo/core:py_utils",
        "//lingvo/core:summary_utils",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

filegroup(
    name = "wpm_files",
    srcs = ["wpm_16k_librispeech.vocab"],
)
