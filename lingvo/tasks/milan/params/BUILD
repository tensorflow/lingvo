load("//lingvo:lingvo.bzl", "pytype_strict_library")
load("//lingvo:lingvo.bzl", "py_strict_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

pytype_strict_library(
    name = "params",
    srcs = [
        "cxc.py",
        "dual_encoder_recipe.py",
        "params.py",
    ],
    srcs_version = "PY3",
    tags = ["keep_dep"],
    deps = [
        ":generic_datasets",
        # Implicit attr dependency.
        "//lingvo:compat",
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/core:hyperparams",
        "//lingvo/core:layers",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
        "//lingvo/tasks/milan:common_schema",
        "//lingvo/tasks/milan:constants",
        "//lingvo/tasks/milan:dataset_spec",
        "//lingvo/tasks/milan:dual_encoder",
        "//lingvo/tasks/milan:image_preprocessor",
        "//lingvo/tasks/milan:input_generator",
        "//lingvo/tasks/milan:labels",
        "//lingvo/tasks/milan:tf_hub_layers",
        "//lingvo/tasks/milan:transformers",
        "//lingvo/tasks/milan:utils",
    ],
)

pytype_strict_library(
    name = "generic_datasets",
    srcs = [
        "generic_datasets.py",
    ],
    srcs_version = "PY3",
    deps = [
        # Implicit attr dependency.
        "//lingvo:compat",
        "//lingvo/core:hyperparams",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/milan:common_schema",
        "//lingvo/tasks/milan:dataset_spec",
        "//lingvo/tasks/milan:labels",
        "//lingvo/tasks/milan:utils",
    ],
)

py_strict_test(
    name = "generic_datasets_test",
    srcs = ["generic_datasets_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":generic_datasets",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)
