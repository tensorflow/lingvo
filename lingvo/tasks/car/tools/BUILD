package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "kitti_exporter",
    srcs = ["kitti_exporter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":kitti_data",
        # Implicit network file system dependency.
        # Implicit PIL dependency.
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit absl.logging dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_binary(
    name = "create_kitti_crop_dataset",
    srcs = [
        "create_kitti_crop_dataset.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit apache_beam dependency.
        "//lingvo:compat",
        "//lingvo:model_registry",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/car:geometry",
        "//lingvo/tasks/car:input_extractor",
        "//lingvo/tasks/car/params",
        "//lingvo/tools:beam_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "kitti_data",
    srcs = ["kitti_data.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "kitti_data_test",
    srcs = ["kitti_data_test.py"],
    data = [
        "//lingvo/tasks/car/testdata:kitti_raw",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":kitti_data",
        "//lingvo:compat",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
        "//lingvo/tasks/car:geometry",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "export_kitti_detection_lib",
    srcs = ["export_kitti_detection.py"],
    srcs_version = "PY3",
    deps = [
        ":kitti_data",
        # Implicit absl.flags dependency.
        "//lingvo:compat",
        "//lingvo/core/ops:record_py_pb2",
        "//lingvo/tasks/car:kitti_metadata",
        # Implicit numpy dependency.
    ],
)

py_binary(
    name = "export_kitti_detection",
    srcs = ["export_kitti_detection.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":export_kitti_detection_lib"],
)

py_binary(
    name = "compare_params",
    srcs = ["compare_params_main.py"],
    main = "compare_params_main.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # Implicit absl.flags dependency.
        "//lingvo:compat",
        "//lingvo/tasks/car/params:kitti",
        "//lingvo/tools:compare_params_lib",
    ],
)
