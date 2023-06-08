load("//lingvo:lingvo.bzl", "pytype_strict_library")
load("//lingvo:lingvo.bzl", "py_strict_test")

# Multimodal representation learning.
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

pytype_strict_library(
    name = "common_schema",
    srcs = ["common_schema.py"],
    deps = ["//lingvo:compat"],
)

pytype_strict_library(
    name = "constants",
    srcs = [
        "constants.py",
    ],
    deps = [],
)

pytype_strict_library(
    name = "dataset_spec",
    srcs = [
        "dataset_spec.py",
    ],
    deps = [
        ":constants",
        ":labels",
        # Implicit absl.logging dependency.
        # Implicit attr dependency.
        "//lingvo:compat",
    ],
)

pytype_strict_library(
    name = "dual_encoder",
    srcs = ["dual_encoder.py"],
    deps = [
        ":labels",
        ":score_functions",
        ":tpu_utils",
        ":utils",
        # Implicit absl.logging dependency.
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:base_model",
        "//lingvo/core:hyperparams",
        "//lingvo/core:layers",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
    ],
)

pytype_strict_library(
    name = "labels",
    srcs = ["labels.py"],
    deps = [
        ":tpu_utils",
        ":utils",
        # Implicit attr dependency.
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "score_functions",
    srcs = ["score_functions.py"],
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
    ],
)

pytype_strict_library(
    name = "tf_hub_layers",
    srcs = ["tf_hub_layers.py"],
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        # Implicit tensorflow_hub dependency.
    ],
)

pytype_strict_library(
    name = "transformers",
    srcs = ["transformers.py"],
    deps = [
        ":utils",
        "//lingvo:compat",
        "//lingvo/core:builder_layers",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/mt:layers",
    ],
)

py_strict_test(
    name = "labels_test",
    srcs = ["labels_test.py"],
    deps = [
        ":labels",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_strict_test(
    name = "score_functions_test",
    srcs = ["score_functions_test.py"],
    deps = [
        ":score_functions",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)

pytype_strict_library(
    name = "tpu_utils",
    srcs = ["tpu_utils.py"],
    deps = [
        # Implicit absl.logging dependency.
        "//lingvo:compat",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:py_utils",
    ],
)

py_strict_test(
    name = "tf_hub_layers_test",
    srcs = ["tf_hub_layers_test.py"],
    deps = [
        ":tf_hub_layers",
        "//lingvo:compat",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:test_utils",
        # Implicit tensorflow_hub dependency.
    ],
)

py_strict_test(
    name = "tpu_utils_test",
    srcs = ["tpu_utils_test.py"],
    deps = [
        ":tpu_utils",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit mock dependency.
    ],
)

py_strict_test(
    name = "transformers_test",
    srcs = ["transformers_test.py"],
    deps = [
        ":transformers",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        # Implicit absl.logging dependency.
        "//lingvo:compat",
        "//lingvo/core:builder_layers",
        "//lingvo/core:py_utils",
    ],
)

py_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

pytype_strict_library(
    name = "image_preprocessor",
    srcs = ["image_preprocessor.py"],
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
    ],
)

pytype_strict_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    deps = [
        # Implicit absl.logging dependency.
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:py_utils",
    ],
)

py_strict_test(
    name = "dataset_spec_test",
    srcs = ["dataset_spec_test.py"],
    deps = [
        ":dataset_spec",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)

py_strict_test(
    name = "dual_encoder_test",
    srcs = ["dual_encoder_test.py"],
    deps = [
        ":dual_encoder",
        ":labels",
        "//lingvo:compat",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_strict_test(
    name = "image_preprocessor_test",
    srcs = ["image_preprocessor_test.py"],
    deps = [
        ":image_preprocessor",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:test_utils",
    ],
)

py_strict_test(
    name = "input_generator_test",
    size = "medium",
    srcs = ["input_generator_test.py"],
    deps = [
        ":input_generator",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:test_utils",
    ],
)
