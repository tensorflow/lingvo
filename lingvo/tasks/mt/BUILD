package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "layers",
    srcs = ["layers.py"],
    deps = [
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:layers_with_attention",
    ],
)

py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    shard_count = 2,
    deps = [":layers_test_lib"],
)

py_library(
    name = "layers_test_lib",
    testonly = True,
    srcs = ["layers_test.py"],
    deps = [
        ":layers",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:layers_with_attention",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "encoder",
    srcs = ["encoder.py"],
    deps = [
        ":layers",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_encoder",
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:model_helper",
        "//lingvo/core:py_utils",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:summary_utils",
    ],
)

py_test(
    name = "encoder_test",
    srcs = ["encoder_test.py"],
    shard_count = 12,
    deps = [
        ":encoder",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "decoder",
    srcs = ["decoder.py"],
    deps = [
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow python framework_for_generated_wrappers dependency.
        "//lingvo/core:attention",
        "//lingvo/core:base_decoder",
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:layers_with_attention",
        "//lingvo/core:model_helper",
        "//lingvo/core:plot",
        "//lingvo/core:py_utils",
        "//lingvo/core:quant_utils",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:rnn_layers",
        "//lingvo/core:summary_utils",
    ],
)

py_test(
    name = "decoder_test",
    srcs = ["decoder_test.py"],
    shard_count = 20,
    deps = [":decoder_test_lib"],
)

py_library(
    name = "decoder_test_lib",
    testonly = True,
    srcs = ["decoder_test.py"],
    deps = [
        ":decoder",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_layer",
        "//lingvo/core:input_generator_helper",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        "//lingvo/core/ops:hyps_py_pb2",
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
    data = [
        "//lingvo/tasks/mt/testdata:wmt14_ende_tfexample",
    ],
    deps = [
        ":input_generator",
        # Implicit numpy dependency.
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
    deps = [
        ":decoder",
        ":encoder",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_layer",
        "//lingvo/core:base_model",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
        "//lingvo/core:summary_utils",
    ],
)

py_test(
    name = "model_test",
    size = "large",
    srcs = ["model_test.py"],
    shard_count = 20,
    deps = [":model_test_lib"],
)

py_library(
    name = "model_test_lib",
    testonly = True,
    srcs = ["model_test.py"],
    data = [
        "//lingvo/tasks/mt/testdata:wmt14_ende_tfexample",
    ],
    deps = [
        ":decoder",
        ":encoder",
        ":input_generator",
        ":model",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_layer",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:lr_schedule",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

filegroup(
    name = "wpm_ende",
    srcs = glob(include = [
        "wpm-ende*.voc",
    ]),
)
