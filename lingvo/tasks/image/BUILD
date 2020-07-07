# Image related tasks.
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "classifier",
    srcs = ["classifier.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_model",
        "//lingvo/core:layers",
        "//lingvo/core:metrics",
        "//lingvo/core:plot",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "classifier_test",
    srcs = ["classifier_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":classifier",
        ":input_generator",
        "//lingvo:compat",
        "//lingvo:model_registry",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        "//lingvo/tasks/image/params:mnist",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
    ],
)

py_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":input_generator",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)
