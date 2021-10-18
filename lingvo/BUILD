load(
    "//lingvo:lingvo.bzl",
    "lingvo_py_binary",
)
load("//lingvo:lingvo.bzl", "pytype_strict_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE"])

config_setting(
    name = "cuda",
    values = {"define": "using_cuda=true"},
)

py_library(
    name = "base_runner",
    srcs = ["base_runner.py"],
    srcs_version = "PY3",
    deps = [
        ":base_trial",
        ":compat",
        ":pdb_wrapper",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:early_stop",
        "//lingvo/core:py_utils",
    ],
)

py_library(
    name = "base_trial",
    srcs = ["base_trial.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo/core:hyperparams",
    ],
)

py_library(
    name = "compat",
    srcs = ["compat.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "compat_test",
    srcs = ["compat_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":compat",
    ],
)

py_library(
    name = "model_imports_no_params",
    srcs = ["model_imports.py"],
    srcs_version = "PY3",
)

# Depend on this for access to the model registry with params for all tasks as
# transitive deps.  Only py_binary should depend on this target.
py_library(
    name = "model_imports",
    srcs_version = "PY3",
    deps = [
        ":model_imports_no_params",
        "//lingvo/tasks:all_params",
    ],
)

py_test(
    name = "model_import_test",
    srcs = ["model_import_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":compat",
        ":model_imports_no_params",
    ],
)

py_library(
    name = "models_test_helper",
    testonly = 1,
    srcs = ["models_test_helper.py"],
    srcs_version = "PY3",
    deps = [
        ":compat",
        ":datasets_lib",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_model",
        "//lingvo/core:bn_layers",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:hyperparams",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":compat",
        ":model_imports",
        ":model_registry",
        ":model_registry_test_lib",
        ":models_test_helper",
        "//lingvo/core:base_model",
    ],
)

py_library(
    name = "model_registry",
    srcs = ["model_registry.py"],
    srcs_version = "PY3",
    deps = [
        ":compat",
        ":model_imports_no_params",
        "//lingvo/core:base_model_params",
        "//lingvo/core:program_lib",
    ],
)

py_test(
    name = "model_registry_test",
    srcs = ["model_registry_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":model_registry_test_lib"],
)

py_library(
    name = "model_registry_test_lib",
    testonly = 1,
    srcs = ["model_registry_test.py"],
    srcs_version = "PY3",
    deps = [
        ":compat",
        ":model_registry",
        # Implicit absl.testing.flagsaver dependency.
        # Implicit absl.testing.parameterized dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_model",
        "//lingvo/core:base_model_params",
        "//lingvo/core:program_lib",
        "//lingvo/core:test_utils",
    ],
)

pytype_strict_library(
    name = "datasets_lib",
    srcs = ["datasets.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.logging dependency.
    ],
)

py_test(
    name = "datasets_test",
    srcs = ["datasets_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":compat",
        ":datasets_lib",
        "//lingvo/core:base_model_params",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "trainer_impl",
    srcs = ["trainer_impl.py"],
    srcs_version = "PY3",
    deps = [
        ":compat",
        "//lingvo:base_runner",
        "//lingvo/core:base_model",
        "//lingvo/core:checkpointer_lib",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
        "//lingvo/core:summary_utils",
    ],
)

py_library(
    name = "trainer_lib",
    srcs = ["trainer.py"],
    srcs_version = "PY3",
    deps = [
        ":base_trial",
        ":compat",
        ":datasets_lib",
        ":executor_lib",
        ":model_imports_no_params",
        ":model_registry",
        ":pdb_wrapper",
        ":trainer_impl",
        ":trainer_utils",
        # Implicit network file system dependency.
        # Implicit python proto dependency.
        # Implicit IPython dependency.
        "//lingvo:base_runner",
        "//lingvo/core:base_model",
        "//lingvo/core:base_model_params",
        "//lingvo/core:checkpointer_lib",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:inference_graph_exporter",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
        "//lingvo/core:summary_utils",
        "//lingvo/core:tpu_embedding_layers",
        # Implicit numpy dependency.
        # Implicit tensorflow grpc dependency.
    ],
)

lingvo_py_binary(
    name = "trainer",
    srcs = [":trainer_lib"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_imports",
        ":trainer_lib",
    ],
)

py_test(
    name = "trainer_test",
    size = "large",
    timeout = "long",
    srcs = ["trainer_test.py"],
    python_version = "PY3",
    shard_count = 12,
    srcs_version = "PY3",
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "optonly",
    ],
    deps = [
        ":base_trial",
        ":compat",
        ":model_registry",
        ":trainer_lib",
        # Implicit absl.testing.flagsaver dependency.
        # Implicit absl.testing.parameterized dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_model",
        "//lingvo/core:base_model_params",
        "//lingvo/core:hyperparams",
        "//lingvo/core:inference_graph_py_pb2",
        "//lingvo/core:test_utils",
        "//lingvo/core:trainer_test_utils",
        "//lingvo/tasks/image:input_generator",
        "//lingvo/tasks/image/params:mnist",  # build_cleaner: keep
        "//lingvo/tasks/punctuator/params:codelab",  # build_cleaner: keep
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "trainer_test_lib",
    testonly = 1,
    srcs = ["trainer_test.py"],
    srcs_version = "PY3",
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "optonly",
    ],
    deps = [
        ":base_trial",
        ":compat",
        ":model_registry",
        ":trainer_lib",
        # Implicit absl.testing.flagsaver dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_model",
        "//lingvo/core:base_model_params",
        "//lingvo/core:hyperparams",
        "//lingvo/core:inference_graph_py_pb2",
        "//lingvo/core:test_utils",
        "//lingvo/core:trainer_test_utils",
        "//lingvo/tasks/image:input_generator",
        "//lingvo/tasks/image/params:mnist",  # build_cleaner: keep
        "//lingvo/tasks/punctuator/params:codelab",  # build_cleaner: keep
        # Implicit numpy dependency.
    ],
)

lingvo_py_binary(
    name = "ipython_kernel",
    srcs = ["ipython_kernel.py"],
    data = [
        "//lingvo/tasks/punctuator/tools:download_brown_corpus",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_imports",  # build_cleaner: keep
        ":trainer_lib",  # build_cleaner: keep
        # Implicit absl.app dependency.
        "//lingvo/core:predictor_lib",  # build_cleaner: keep
        "//lingvo/core/ops:hyps_py_pb2",  # build_cleaner: keep
    ],
)

genrule(
    name = "tf_dot_protos",
    srcs = [],
    outs = ["tf_protos.tar"],
    cmd =
        "$(location //lingvo/tools:" +
        "generate_tf_dot_protos) $(location " +
        "//lingvo/tools:generate_proto_def) $(@D)",
    tools = [
        "//lingvo/tools:generate_proto_def",
        "//lingvo/tools:generate_tf_dot_protos",
    ],
)

py_library(
    name = "executor_lib",
    srcs = ["executor.py"],
    srcs_version = "PY3",
    deps = [
        ":compat",
        ":pdb_wrapper",
        # Implicit network file system dependency.
        "//lingvo:base_runner",
        "//lingvo/core:base_model",
        "//lingvo/core:checkpointer_lib",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:ml_perf_log",
        "//lingvo/core:multitask_model",
        "//lingvo/core:py_utils",
        "//lingvo/core:task_scheduler",
        "//lingvo/core:tpu_embedding_layers",
        # Implicit numpy dependency.
        # Implicit tensorflow grpc dependency.
    ],
)

py_library(
    name = "trainer_utils",
    srcs = ["trainer_utils.py"],
    deps = [":compat"],
)

py_library(
    name = "pdb_wrapper",
    srcs = ["pdb_wrapper.py"],
    deps = [
        # Implicit pdb dependency.
        # Implicit absl.flags dependency.
    ],
)
