# Params for ASR models.

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
    data = ["//lingvo/tasks/asr:wpm_files"],
    srcs_version = "PY3",
    tags = ["keep_dep"],
    deps = [
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/core:datasource",
        "//lingvo/core:program_lib",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
        "//lingvo/core:tokenizers",
        "//lingvo/tasks/asr:input_generator",
        "//lingvo/tasks/asr:model",
    ],
)
