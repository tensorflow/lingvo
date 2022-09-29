# Description:
#   Language modeling-specific libraries and model configurations

load("//lingvo:lingvo.bzl", "pytype_strict_library")
load("//lingvo:lingvo.bzl", "py_strict_test")
load("//lingvo/jax:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_strict_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    deps = [
        # Implicit absl.logging dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:layers",
        "//lingvo/core/ops",
        "//lingvo/jax:base_input",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "model_params",
    srcs = ["model_params.py"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:asserts",
        "//lingvo/jax:base_layer",
        "//lingvo/jax:base_model",
        "//lingvo/jax:base_model_params",
        "//lingvo/jax:base_task",
        "//lingvo/jax:optimizers",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:schedules",
        "//lingvo/jax/layers",
        # Implicit numpy dependency.
    ],
)

filegroup(
    name = "testdata",
    testonly = 1,
    srcs = glob(["testdata/*"]),
)

py_strict_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    data = [":testdata"],
    deps = [
        ":input_generator",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        "//lingvo/core:test_helper",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)
