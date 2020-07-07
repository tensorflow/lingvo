# Car related tasks.

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

py_test(
    name = "waymo_ap_metric_test",
    srcs = ["waymo_ap_metric_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":waymo_ap_metric",
        ":waymo_metadata",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
        # Implicit Waymo Open Dataset label dependency.
        # Implicit Waymo Open Dataset metrics_proto dependency.
    ],
)

py_test(
    name = "waymo_open_input_generator_test",
    srcs = ["waymo_open_input_generator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":waymo_open_input_generator",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "waymo_ap_metric",
    srcs = [
        "waymo_ap_metric.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/car:ap_metric",
        # Implicit numpy dependency.
        # Implicit Waymo Open Dataset label dependency.
        # Implicit Waymo Open Dataset detection metrics dependency.
        # Implicit Waymo Open Dataset py_metrics dependency.
        # Implicit Waymo Open Dataset config_util_py dependency.
        # Implicit Waymo Open Dataset breakdown_proto dependency.
        # Implicit Waymo Open Dataset metrics_proto dependency.
    ],
)

py_library(
    name = "waymo_decoder",
    srcs = ["waymo_decoder.py"],
    srcs_version = "PY3",
    deps = [
        ":waymo_ap_metric",
        ":waymo_metadata",
        "//lingvo:compat",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/car:base_decoder",
        "//lingvo/tasks/car:detection_3d_metrics",
        "//lingvo/tasks/car:transform_util",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "waymo_metadata",
    srcs = [
        "waymo_metadata.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//lingvo/tasks/car:evaluation_metadata",
    ],
)

py_library(
    name = "waymo_open_input_generator",
    srcs = ["waymo_open_input_generator.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:datasource",
        "//lingvo/core:hyperparams",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/car:input_extractor",
        "//lingvo/tasks/car:input_preprocessors",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "export_to_submission_format_lib",
    srcs = ["export_to_submission_format.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit absl.flags dependency.
        "//lingvo:compat",
        "//lingvo/core/ops:record_py_pb2",
        # Implicit Waymo Open Dataset metrics_proto dependency.
    ],
)

py_binary(
    name = "export_to_submission_format",
    srcs = ["export_to_submission_format.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":export_to_submission_format_lib"],
)
