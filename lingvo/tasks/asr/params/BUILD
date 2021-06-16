# Params for ASR models.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "params",
    srcs = [
        "librispeech.py",
        "params.py",
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
