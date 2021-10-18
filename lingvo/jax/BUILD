# Description:
#   Base directory for the lingvo Jax library.

load("//lingvo:lingvo.bzl", "pytype_strict_binary", "pytype_strict_library")
load("//lingvo:lingvo.bzl", "py_strict_test")
load("//lingvo/jax:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_strict_binary(
    name = "main",
    srcs = ["main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eval",
        ":train",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit clu dependency.
        # Implicit jax dependency.
        "//lingvo/jax:model_imports",
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "base_layer",
    srcs = ["base_layer.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        ":pytypes",
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "base_model_params",
    srcs = ["base_model_params.py"],
    srcs_version = "PY3",
    deps = [":py_utils"],
)

pytype_strict_library(
    name = "checkpoints",
    srcs = ["checkpoints.py"],
    srcs_version = "PY3",
    deps = [
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit flax's core dependency.
        # Implicit flax's training dependency.
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "eval",
    srcs = ["eval.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":base_model_params",
        ":checkpoints",
        ":model_utils",
        ":py_utils",
        ":pytypes",
        ":summary_utils",
        ":train_states",
        ":trainer_lib",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//lingvo/jax:model_imports",
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "gshard_utils",
    srcs = ["gshard_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":pytypes",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "metric_utils",
    srcs = ["metric_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        ":pytypes",
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":metric_utils",
        ":optimizers",
        ":py_utils",
        ":pytypes",
        ":train_states",
        # Implicit jax dependency.
        "//lingvo/jax/layers",
        # Implicit optax dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "model_imports_no_params",
    srcs = ["model_imports.py"],
    srcs_version = "PY3",
    deps = ["//lingvo:model_imports_no_params"],
)

pytype_strict_library(
    name = "model_imports",
    srcs = ["model_imports.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:model_imports_no_params",
        "//lingvo/jax/tasks:all_params",
    ],
)

pytype_strict_library(
    name = "model_registry",
    srcs = ["model_registry.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model_params",
        # Implicit absl.logging dependency.
    ],
)

pytype_strict_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model_params",
        ":model",
        ":py_utils",
        ":pytypes",
        ":summary_utils",
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//lingvo/jax:model_imports",
        "//lingvo/jax:model_registry",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    srcs_version = "PY3",
    deps = [
        ":gshard_utils",
        ":py_utils",
        ":pytypes",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit optax dependency.
    ],
)

pytype_strict_library(
    name = "py_utils",
    srcs = ["py_utils.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//lingvo/core:cluster",
        "//lingvo/core:hyperparams",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "pytypes",
    srcs = ["pytypes.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "schedules",
    srcs = ["schedules.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        ":pytypes",
        # Implicit jax dependency.
        # Implicit optax dependency.
    ],
)

pytype_strict_library(
    name = "summary_utils",
    srcs = ["summary_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":pytypes",
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        # Implicit absl.flags dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":checkpoints",
        ":model_utils",
        ":py_utils",
        ":summary_utils",
        ":trainer_lib",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//lingvo/jax:model_imports",
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "train_states",
    srcs = ["train_states.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        # Implicit flax's core dependency.
        # Implicit optax dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "trainer_lib",
    srcs = ["trainer_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":model",
        ":py_utils",
        ":pytypes",
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "model_imports_test",
    srcs = ["model_imports_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_imports",
        # Implicit absl.testing.absltest dependency.
    ],
)

py_strict_test(
    name = "model_registry_test",
    srcs = ["model_registry_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_model_params",
        ":model_registry",
        # Implicit absl.testing.absltest dependency.
        "//lingvo/jax/layers",
        "//lingvo/jax/tasks/test/params",
    ],
)

py_strict_test(
    name = "py_utils_test",
    srcs = ["py_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "schedules_test",
    srcs = ["schedules_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":schedules",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
    ],
)
