package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "params",
    srcs = [
        "params.py",
        "wmt14_en_de.py",
        "wmtm16_en_de.py",
        "xendec/wmt14_en_de.py",
    ],
    tags = ["keep_dep"],
    deps = [
        "//lingvo:model_registry",
        "//lingvo/core:base_model_params",
        "//lingvo/tasks/mt:base_config",
        "//lingvo/tasks/mt:input_generator",
        "//lingvo/tasks/mt:model",
    ],
)
