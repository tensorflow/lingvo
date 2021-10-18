# Description:
#   Language modeling model configurations.

load("//lingvo:lingvo.bzl", "pytype_library")
load("//lingvo/jax:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_library(
    name = "params",
    srcs = [
        "bert.py",
        "params.py",
    ],
    srcs_version = "PY3",
    tags = ["keep_dep"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:base_model_params",
        "//lingvo/jax:model",
        "//lingvo/jax:model_registry",
        "//lingvo/jax:optimizers",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:schedules",
        "//lingvo/jax/layers",
        "//lingvo/jax/tasks/lm:input_generator",
        # Implicit numpy dependency.
    ],
)
