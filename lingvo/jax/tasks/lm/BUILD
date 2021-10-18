# Description:
#   Language modeling-specific libraries and model model configurations

load("//lingvo:lingvo.bzl", "pytype_strict_library")
load("//lingvo:lingvo.bzl", "py_strict_test")
load("//lingvo/jax:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_strict_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.logging dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:layers",
        "//lingvo/core/ops",
        "//lingvo/jax:py_utils",
        # Implicit tensorflow dependency.
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
    data = [
        ":testdata",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":input_generator",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:test_helper",
        "//lingvo/jax:py_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)
