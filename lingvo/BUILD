package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

config_setting(
    name = "cuda",
    values = {"define": "using_cuda=true"},
)

config_setting(
    name = "py3_mode",
    values = {"force_python": "PY3"},
)

load(
    "//lingvo:lingvo.bzl",
    "lingvo_py_binary",
)

py_library(
    name = "base_runner",
    srcs = ["base_runner.py"],
    deps = [
        ":base_trial",
        # Implicit tensorflow dependency.
        # Implicit tensorflow py proto dependency.
        "//lingvo/core:cluster_factory",
        "//lingvo/core:early_stop",
        "//lingvo/core:py_utils",
    ],
)

py_library(
    name = "base_trial",
    srcs = ["base_trial.py"],
    deps = [
        "//lingvo/core:hyperparams",
    ],
)

py_library(
    name = "model_imports_no_params",
    srcs = ["model_imports.py"],
    deps = [
        ":model_registry",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

# Depend on this for access to the model registry with params for all tasks as
# transitive deps.  Only py_binary should depend on this target.
py_library(
    name = "model_imports",
    deps = [
        ":model_imports_no_params",
        "//lingvo/tasks:all_params",
    ],
)

py_test(
    name = "model_import_test",
    srcs = ["model_import_test.py"],
    deps = [
        ":model_imports_no_params",
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "models_test_helper",
    testonly = 1,
    srcs = ["models_test_helper.py"],
    deps = [
        # Implicit tensorflow dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_model",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    deps = [
        ":model_imports",
        ":model_registry",
        ":model_registry_test_lib",
        ":models_test_helper",
        # Implicit tensorflow dependency.
        "//lingvo/core:base_model_params",
    ],
)

py_library(
    name = "model_registry",
    srcs = ["model_registry.py"],
    deps = [
        # Implicit tensorflow dependency.
        "//lingvo/core:base_model_params",
    ],
)

py_test(
    name = "model_registry_test",
    srcs = ["model_registry_test.py"],
    deps = [":model_registry_test_lib"],
)

py_library(
    name = "model_registry_test_lib",
    testonly = True,
    srcs = ["model_registry_test.py"],
    deps = [
        ":model_registry",
        # Implicit tensorflow dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_model",
        "//lingvo/core:base_model_params",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "trainer_lib",
    srcs = ["trainer.py"],
    deps = [
        ":base_trial",
        ":model_imports_no_params",
        ":model_registry",
        # Implicit network file system dependency.
        "//lingvo:base_runner",
        # Implicit IPython dependency.
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow py proto dependency.
        # Implicit tensorflow grpc dependency.
        "//lingvo/core:base_model",
        "//lingvo/core:base_model_params",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:inference_graph_exporter",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
    ],
)

lingvo_py_binary(
    name = "trainer",
    srcs = [":trainer_lib"],
    deps = [
        ":model_imports",
        ":trainer_lib",
    ],
)

py_test(
    name = "trainer_test",
    size = "large",
    srcs = ["trainer_test.py"],
    shard_count = 5,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "optonly",
    ],
    deps = [":trainer_test_lib"],
)

py_library(
    name = "trainer_test_lib",
    testonly = True,
    srcs = ["trainer_test.py"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "optonly",
    ],
    deps = [
        ":base_trial",
        ":model_registry",
        ":trainer_lib",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_layer",
        "//lingvo/core:base_model",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        "//lingvo/core:trainer_test_utils",
        "//lingvo/tasks/image:input_generator",
        "//lingvo/tasks/image/params:mnist",
        "//lingvo/tasks/punctuator/params",
    ],
)

lingvo_py_binary(
    name = "ipython_kernel",
    srcs = ["ipython_kernel.py"],
    data = [
        "//lingvo/tasks/punctuator/tools:download_brown_corpus",
    ],
    deps = [
        ":model_imports",  # build_cleaner: keep
        ":trainer_lib",  # build_cleaner: keep
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
