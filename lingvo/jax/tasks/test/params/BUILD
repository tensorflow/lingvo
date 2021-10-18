# Description:
#   Test model configurations.

load("//lingvo/jax:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

py_library(
    name = "params",
    srcs = [
        "params.py",
        "synthetic.py",
    ],
    srcs_version = "PY3",
    tags = ["keep_dep"],
    deps = [
        "//lingvo/jax:base_model_params",
        "//lingvo/jax:model_registry",
        "//lingvo/jax/layers",
    ],
)
