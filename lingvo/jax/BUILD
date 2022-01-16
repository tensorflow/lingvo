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
    name = "asserts",
    srcs = ["asserts.py"],
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
    name = "base_input",
    srcs = ["base_input.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        ":pytypes",
        "//lingvo/core:datasource",
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "base_input_test",
    srcs = ["base_input_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_input",
        ":py_utils",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:generic_input",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "base_model_params",
    srcs = ["base_model_params.py"],
    srcs_version = "PY3",
    deps = [
        ":base_input",
        ":py_utils",
    ],
)

pytype_strict_library(
    name = "checkpoints",
    srcs = ["checkpoints.py"],
    srcs_version = "PY3",
    deps = [
        ":asserts",
        ":checkpoint_py_pb2",
        ":py_utils",
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit flax's core dependency.
        # Implicit flax's training dependency.
        # Implicit jax dependency.
        # Implicit jax/experimental/gda_serialization:google_spec dependency.
        # Implicit jax/experimental/gda_serialization:serialization dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "checkpoint_managers",
    srcs = ["checkpoint_managers.py"],
    deps = [
        ":checkpoint_py_pb2",
        ":py_utils",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "eval",
    srcs = ["eval.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":base_model_params",
        ":checkpoint_py_pb2",
        ":model_utils",
        ":py_utils",
        ":pytypes",
        ":summary_utils",
        ":train_states",
        ":trainer_lib",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit jax:mesh_utils dependency.
        "//lingvo/jax:checkpoints",
        "//lingvo/jax:io_utils",
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
    name = "io_utils",
    srcs = ["io_utils.py"],
    srcs_version = "PY3",
)

pytype_strict_library(
    name = "learners",
    srcs = ["learners.py"],
    deps = [
        ":asserts",
        ":base_layer",
        ":optimizer_prefix_vectorization",
        ":optimizers",
        ":py_utils",
        ":pytypes",
        # Implicit jax dependency.
        # Implicit optax dependency.
        # Implicit tensorflow dependency.
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
        ":base_input",
        ":base_layer",
        ":learners",
        ":metric_utils",
        ":optimizers",
        ":py_utils",
        ":pytypes",
        ":train_states",
        # Implicit jax dependency.
        "//lingvo/jax/layers",
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "model_test",
    srcs = ["model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":model",
        ":optimizers",
        ":py_utils",
        ":schedules",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
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
        ":base_input",
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
        ":asserts",
        ":base_layer",
        ":gshard_utils",
        ":py_utils",
        ":pytypes",
        # Implicit optax_shampoo dependency.
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit optax dependency.
    ],
)

pytype_strict_library(
    name = "optimizer_prefix_vectorization",
    srcs = ["optimizer_prefix_vectorization.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizers",
        ":py_utils",
        ":pytypes",
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
        "//third_party/py/flax",
        # Implicit jax dependency.
        "//lingvo/core:cluster",
        "//lingvo/core:hyperparams",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
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
        ":base_layer",
        ":py_utils",
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
        ":base_input",
        ":base_model_params",
        ":checkpoint_managers",
        ":checkpoint_py_pb2",
        ":model_utils",
        ":py_utils",
        ":summary_utils",
        ":trainer_lib",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit jax:mesh_utils dependency.
        "//lingvo/jax:checkpoints",
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
        ":summary_utils",
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "asserts_test",
    srcs = ["asserts_test.py"],
    deps = [
        ":asserts",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
    ],
)

py_strict_test(
    name = "checkpoint_managers_test",
    srcs = ["checkpoint_managers_test.py"],
    deps = [
        ":checkpoint_managers",
        ":checkpoint_py_pb2",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "learners_test",
    srcs = ["learners_test.py"],
    deps = [
        ":base_layer",
        ":learners",
        ":optimizer_prefix_vectorization",
        ":optimizers",
        ":py_utils",
        ":schedules",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
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

proto_library(
    name = "checkpoint_proto",
    srcs = ["checkpoint.proto"],
)

py_proto_library(
    name = "checkpoint_py_pb2",
    api_version = 2,
    deps = [":checkpoint_proto"],
)
