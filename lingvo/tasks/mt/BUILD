load("//lingvo:lingvo.bzl", "py2and3_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:layers_with_attention",
        # Implicit six dependency.
    ],
)

py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    shard_count = 2,
    deps = [
        ":layers_test_lib",
    ],
)

py_library(
    name = "layers_test_lib",
    testonly = 1,
    srcs = ["layers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        "//lingvo:compat",
        "//lingvo/core:layers_with_attention",
        "//lingvo/core:py_utils",
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
        ":layers",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:model_helper",
        "//lingvo/core:py_utils",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:summary_utils",
        # Implicit six dependency.
    ],
)

py_test(
    name = "encoder_test",
    srcs = ["encoder_test.py"],
    shard_count = 12,
    deps = [
        ":encoder",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "decoder",
    srcs = ["decoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:attention",
        "//lingvo/core:base_decoder",
        "//lingvo/core:base_layer",
        "//lingvo/core:insertion",
        "//lingvo/core:layers",
        "//lingvo/core:layers_with_attention",
        "//lingvo/core:model_helper",
        "//lingvo/core:plot",
        "//lingvo/core:py_utils",
        "//lingvo/core:quant_utils",
        "//lingvo/core:rnn_cell",
        "//lingvo/core:rnn_layers",
        "//lingvo/core:summary_utils",
        # Implicit six dependency.
        # Implicit tensorflow python framework_for_generated_wrappers dependency.
    ],
)

py_test(
    name = "decoder_test",
    srcs = ["decoder_test.py"],
    shard_count = 20,
    deps = [
        ":decoder_test_lib",
    ],
)

py_library(
    name = "decoder_test_lib",
    testonly = 1,
    srcs = ["decoder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":decoder",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:input_generator_helper",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        "//lingvo/core/ops:hyps_py_pb2",
        # Implicit numpy dependency.
        # Implicit six dependency.
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
        "//lingvo/core:tokenizers",
        # Implicit six dependency.
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
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_helper",
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
        ":decoder",
        ":encoder",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:base_model",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
        # Implicit six dependency.
    ],
)

py_test(
    name = "model_test",
    size = "large",
    srcs = ["model_test.py"],
    shard_count = 20,
    deps = [
        ":model_test_lib",
    ],
)

py_library(
    name = "model_test_lib",
    testonly = 1,
    srcs = ["model_test.py"],
    data = [
        "//lingvo/tasks/mt/testdata:wmt14_ende_tfexample",
    ],
    srcs_version = "PY2AND3",
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
        # Implicit six dependency.
    ],
)

filegroup(
    name = "wpm_ende",
    srcs = glob(include = [
        "wpm-ende*.voc",
    ]),
)
