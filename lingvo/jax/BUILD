# Description:
#   Base directory for the lingvo Jax library.

load("//lingvo:lingvo.bzl", "pytype_strict_binary", "pytype_strict_library")
load("//lingvo:lingvo.bzl", "py_strict_test")
load("//lingvo:lingvo.bzl", "lingvo_proto_cc", "lingvo_proto_py")
load("//lingvo/jax:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_strict_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        ":eval",
        ":model_imports",  # buildcleaner: keep
        ":py_utils",
        ":train",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit clu dependency.
        # Implicit jax dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "asserts",
    srcs = ["asserts.py"],
)

pytype_strict_library(
    name = "base_input",
    srcs = ["base_input.py"],
    deps = [
        ":py_utils",
        ":pytypes",
        # Implicit absl.logging dependency.
        "//lingvo/core:cluster_factory",
        "//lingvo/core:datasource",
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "base_layer",
    srcs = ["base_layer.py"],
    deps = [
        ":py_utils",
        ":pytypes",
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit flax's core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "base_metrics",
    srcs = ["base_metrics.py"],
    deps = [
        ":py_utils",
        ":pytypes",
        ":summary_utils",
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit flax's core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "base_model_params",
    srcs = ["base_model_params.py"],
    deps = [
        ":base_input",
        ":py_utils",
    ],
)

pytype_strict_library(
    name = "base_task",
    srcs = ["base_task.py"],
    deps = [
        ":base_layer",
        ":base_model",
        ":learners",
        ":optimizers",
        ":py_utils",
        ":train_states",
        # Implicit jax dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "checkpoints",
    srcs = ["checkpoints.py"],
    deps = [
        ":asserts",
        ":checkpoint_py_pb2",
        ":py_utils",
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit flax's core dependency.
        # Implicit flax's training dependency.
        "//third_party/py/flax/training:checkpoints",
        # Implicit jax dependency.
        # Implicit jax/experimental/array_serialization:google_spec dependency.
        # Implicit jax/experimental/array_serialization:serialization dependency.
        # Implicit numpy dependency.
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
    deps = [
        ":base_input",
        ":base_layer",
        ":base_metrics",
        ":base_model_params",
        ":base_task",
        ":checkpoint_py_pb2",
        ":checkpoints",
        ":io_utils",
        ":model_imports",
        ":model_utils",
        ":py_utils",
        ":pytypes",
        ":summary_utils",
        ":train_states",
        ":trainer_lib",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit jax:mesh_utils dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "gshard_utils",
    srcs = ["gshard_utils.py"],
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
    deps = [
        ":py_utils",
        ":pytypes",
        # Implicit jax dependency.
    ],
)

pytype_strict_library(
    name = "base_model",
    srcs = ["base_model.py"],
    deps = [
        ":base_input",
        ":base_layer",
        ":metric_utils",
        ":py_utils",
        ":train_states",
        # Implicit jax dependency.
        "//lingvo/jax/layers",
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "base_model_test",
    srcs = ["base_model_test.py"],
    deps = [
        ":base_layer",
        ":base_model",
        ":py_utils",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "model_imports_no_params",
    srcs = ["model_imports.py"],
    deps = ["//lingvo:model_imports_no_params"],
)

pytype_strict_library(
    name = "model_imports",
    srcs = ["model_imports.py"],
    deps = [
        "//lingvo:model_imports_no_params",
        "//lingvo/jax/tasks:all_params",
    ],
)

pytype_strict_library(
    name = "model_registry",
    srcs = ["model_registry.py"],
    deps = [
        ":base_model_params",
        # Implicit absl.logging dependency.
    ],
)

pytype_strict_library(
    name = "models_test_helper",
    testonly = True,
    srcs = ["models_test_helper.py"],
    deps = [
        ":base_input",
        ":base_task",
        # Implicit absl.testing.absltest dependency.
    ],
)

pytype_strict_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    deps = [
        ":base_input",
        ":base_model",
        ":base_model_params",
        ":model_imports",
        ":model_registry",
        ":py_utils",
        ":pytypes",
        ":summary_utils",
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
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
    deps = [
        # Implicit absl.logging dependency.
        # Implicit flax dependency.
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
    deps = [
        ":py_utils",
        # Implicit jax dependency.
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "schedules",
    srcs = ["schedules.py"],
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
    deps = [
        ":base_layer",
        ":py_utils",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":base_input",
        ":base_model_params",
        ":checkpoint_managers",
        ":checkpoint_py_pb2",
        ":checkpoints",
        ":model_imports",
        ":model_utils",
        ":py_utils",
        ":summary_utils",
        ":trainer_lib",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit jax:mesh_utils dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "train_states",
    srcs = ["train_states.py"],
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
    deps = [
        ":base_layer",
        ":base_model",
        ":base_task",
        ":py_utils",
        ":pytypes",
        ":summary_utils",
        ":train_states",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
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
    name = "base_input_test",
    srcs = ["base_input_test.py"],
    deps = [
        ":base_input",
        ":py_utils",
        ":test_utils",
        # Implicit absl.flags dependency.
        # Implicit absl.testing.absltest dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:generic_input",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "base_metrics_test",
    srcs = ["base_metrics_test.py"],
    deps = [
        ":base_metrics",
        ":py_utils",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit tensorflow dependency.
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
        ":test_utils",
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
    deps = [
        ":model_imports",
        # Implicit absl.testing.absltest dependency.
    ],
)

py_strict_test(
    name = "model_registry_test",
    srcs = ["model_registry_test.py"],
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
    deps = [
        ":base_layer",
        ":py_utils",
        ":test_utils",
        ":train_states",
        # Implicit absl.testing.absltest dependency.
        # Implicit flax's core dependency.
        # Implicit jax dependency.
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "schedules_test",
    srcs = ["schedules_test.py"],
    deps = [
        ":schedules",
        ":test_utils",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
    ],
)

lingvo_proto_cc(
    name = "checkpoint_proto",
    src = "checkpoint.proto",
)

lingvo_proto_py(
    name = "checkpoint_py_pb2",
    src = "checkpoint.proto",
    deps = [":checkpoint_proto"],
)
