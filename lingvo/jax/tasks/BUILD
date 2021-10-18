# Description:
#   Includes all open source model configurations.

load("//lingvo/jax:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

py_library(
    name = "all_params",
    srcs_version = "PY3",
    # LINT.IfChange(task_dirs)
    deps = [
        "//lingvo/jax/tasks/lm/params",
        "//lingvo/jax/tasks/test/params",
    ],
    # LINT.ThenChange(../model_imports.py:task_dirs)
)
