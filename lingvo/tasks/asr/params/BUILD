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
    deps = [
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/core:lr_schedule",
        "//lingvo/core:py_utils",
        "//lingvo/core:tokenizers",
        "//lingvo/tasks/asr:input_generator",
        "//lingvo/tasks/asr:model",
    ],
)
