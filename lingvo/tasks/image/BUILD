# Image related tasks.
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "classifier",
    srcs = ["classifier.py"],
    deps = [
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:base_layer",
        "//lingvo/core:base_model",
        "//lingvo/core:layers",
        "//lingvo/core:lr_schedule",
        "//lingvo/core:metrics",
        "//lingvo/core:plot",
        "//lingvo/core:py_utils",
    ],
)

py_test(
    name = "classifier_test",
    srcs = ["classifier_test.py"],
    deps = [
        ":classifier",
        ":input_generator",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//lingvo:model_registry",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        "//lingvo/tasks/image/params:mnist",
    ],
)

py_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    deps = [
        # Implicit tensorflow dependency.
        "//lingvo/core:base_input_generator",
        "//lingvo/core:py_utils",
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
        "//lingvo/core:test_utils",
    ],
)
