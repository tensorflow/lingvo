package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    deps = [
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
        "//lingvo/core:tokenizers",
        "//lingvo/core/ops:py_x_ops",
    ],
)

py_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    deps = [
        ":input_generator",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:py_utils",
        "//lingvo/core:test_helper",
    ],
)
