# Placeholder: load py_library
# Placeholder: load py_test

# Image related tasks.
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "classifier",
    srcs = ["classifier.py"],
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
    deps = [
        ":classifier",
        ":input_generator",
        #internal proto upb dep
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
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
    ],
)

py_test(
    name = "input_generator_test",
    srcs = ["input_generator_test.py"],
    deps = [
        ":input_generator",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)
