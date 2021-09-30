package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "params",
    srcs = [
        "one_billion_wds.py",
        "params.py",
        "synthetic_packed_input.py",
        "wiki_bert.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo:model_registry",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_model_params",
        "//lingvo/core:gshard_builder",
        "//lingvo/core:gshard_utils",
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
