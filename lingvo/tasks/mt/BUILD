load(
    "//lingvo:lingvo.bzl",
    "lingvo_proto_cc",
    "lingvo_proto_py",
)

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:layers_with_attention",
    ],
)

py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [":layers_test_lib"],
)

py_library(
    name = "layers_test_lib",
    testonly = 1,
    srcs = ["layers_test.py"],
    srcs_version = "PY3",
    deps = [
        ":layers",
        "//lingvo:compat",
        "//lingvo/core:layers_with_attention",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "encoder",
    srcs = ["encoder.py"],
    srcs_version = "PY3",
    deps = [
        ":layers",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:batch_major_attention",
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
    python_version = "PY3",
    shard_count = 12,
    srcs_version = "PY3",
    deps = [
        ":encoder",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:self_attention_layer",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "decoder",
    srcs = ["decoder.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:attention",
        "//lingvo/core:base_decoder",
        "//lingvo/core:batch_major_attention",
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
    python_version = "PY3",
    shard_count = 20,
    srcs_version = "PY3",
    deps = [":decoder_test_lib"],
)

py_library(
    name = "decoder_test_lib",
    testonly = 1,
    srcs = ["decoder_test.py"],
    srcs_version = "PY3",
    deps = [
        ":decoder",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:input_generator_helper",
        "//lingvo/core:layers",
        "//lingvo/core:layers_with_attention",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        "//lingvo/core/ops:hyps_py_pb2",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "data_augmenter",
    srcs = ["data_augmenter.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
        "//lingvo/core/ops",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "data_augmenter_test",
    srcs = ["data_augmenter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data_augmenter",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    srcs_version = "PY3",
    deps = [
        ":text_input_py_pb2",
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:generic_input",
        "//lingvo/core:hyperparams",
        "//lingvo/core:py_utils",
        "//lingvo/core:summary_utils",
        "//lingvo/core:tokenizers",
        "//lingvo/core/ops",
        # Implicit tensorflow_probability dependency.
    ],
)

py_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    data = [
        ":wpm_ende",
        "//lingvo/tasks/mt/testdata:doublebatch_tfexample",
        "//lingvo/tasks/mt/testdata:input_test_data",
        "//lingvo/tasks/mt/testdata:mlperf_tfexample",
        "//lingvo/tasks/mt/testdata:wmt14_ende_tfexample",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":input_generator",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
        "//lingvo/core:tokenizers",
        # Implicit mock dependency.
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
    deps = [
        ":decoder",
        ":encoder",
        "//lingvo:compat",
        "//lingvo/core:base_model",
        "//lingvo/core:insertion",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
    ],
)

py_test(
    name = "model_test",
    size = "large",
    srcs = ["model_test.py"],
    python_version = "PY3",
    shard_count = 20,
    srcs_version = "PY3",
    deps = [":model_test_lib"],
)

py_library(
    name = "model_test_lib",
    testonly = 1,
    srcs = ["model_test.py"],
    data = [
        "//lingvo/tasks/mt/testdata:doublebatch_tfexample",
        "//lingvo/tasks/mt/testdata:wmt14_ende_tfexample",
    ],
    srcs_version = "PY3",
    deps = [
        ":decoder",
        ":encoder",
        ":input_generator",
        ":model",
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_layer",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

filegroup(
    name = "wpm_ende",
    srcs = glob(include = [
        "wpm-ende*.voc",
    ]),
)

py_library(
    name = "base_config",
    srcs = ["base_config.py"],
    srcs_version = "PY3",
    deps = [
        ":decoder",
        ":encoder",
        ":input_generator",
        "//lingvo/core:attention",
        "//lingvo/core:layers",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:rnn_layers",
        "//lingvo/core:schedule",
    ],
)

lingvo_proto_cc(
    name = "text_input_proto",
    src = "text_input.proto",
)

lingvo_proto_py(
    name = "text_input_py_pb2",
    src = "text_input.proto",
    deps = [":text_input_proto"],
)
