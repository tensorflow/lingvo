# Placeholder: load py_library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "brown_corpus_wpm_data",
    srcs = glob(include = ["brown_corpus_wpm.*"]),
)

py_library(
    name = "codelab",
    srcs = ["codelab.py"],
    data = [":brown_corpus_wpm_data"],
    deps = [
        "//lingvo:compat",
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/tasks/mt:base_config",
        "//lingvo/tasks/mt/params",
        "//lingvo/tasks/punctuator:input_generator",
        "//lingvo/tasks/punctuator:model",
    ],
)

py_library(
    name = "params",
    srcs = ["params.py"],
    tags = ["keep_dep"],
    deps = [
        ":codelab",
    ],
)
