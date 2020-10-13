package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "param_files",
    srcs = glob(
        include = [
            "*.py",
        ],
    ),
)

py_library(
    name = "params",
    srcs = [
        ":param_files",
    ],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo:model_registry",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_model_params",
        "//lingvo/core:gshard_builder",
        "//lingvo/core:layers",
        "//lingvo/core:optimizer",
        "//lingvo/core:program_lib",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
        "//lingvo/core:tokenizers",
        "//lingvo/tasks/lm:input_generator",
        "//lingvo/tasks/lm:layers",
        "//lingvo/tasks/lm:model",
        # Implicit numpy dependency.
    ],
)
