# Params for ASR models.

# Placeholder: load py_library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "params",
    srcs = [
        "librispeech.py",
        "params.py",
    ],
    data = ["//lingvo/tasks/asr:wpm_files"],
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
