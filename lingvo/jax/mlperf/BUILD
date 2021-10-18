# Description:
#   MLPerf-specific modules.

load("//lingvo:lingvo.bzl", "pytype_strict_binary", "pytype_strict_library")
load("//lingvo/jax:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_strict_binary(
    name = "main",
    srcs = ["main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":mllog",
        ":train",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit clu dependency.
        # Implicit jax dependency.
        "//lingvo/jax:eval",
        "//lingvo/jax:model_imports",  # build_cleaner: keep
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "mllog",
    srcs = ["mllog.py"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:model",
        "//lingvo/jax:py_utils",
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY3",
    deps = [
        ":mllog",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:checkpoints",
        "//lingvo/jax:model_imports",  # build_cleaner: keep
        "//lingvo/jax:model_utils",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:summary_utils",
        "//lingvo/jax:trainer_lib",
        # Implicit tensorflow dependency.
    ],
)
