package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "waymo_proto_to_tfe",
    srcs = [
        "waymo_proto_to_tfe.py",
    ],
    srcs_version = "PY3",
    deps = [
        # Implicit apache_beam dependency.
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/car:geometry",
        # Implicit numpy dependency.
        # Implicit Waymo Open Dataset proto dependency.
        # Implicit Waymo Open Dataset range_image_utils dependency.
        # Implicit Waymo Open Dataset transform_utils dependency.
    ],
)

py_binary(
    name = "generate_waymo_tf",
    srcs = [
        "generate_waymo_tf.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":waymo_proto_to_tfe",
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit apache_beam dependency.
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/tools:beam_utils",
        # Implicit Waymo Open Dataset proto dependency.
    ],
)
