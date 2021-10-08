load(
    "//lingvo:lingvo.bzl",
    "lingvo_py_binary",
)

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "download_lm1b",
    srcs = ["download_lm1b.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
    ],
)

py_library(
    name = "gshard_lm_decode_lib",
    srcs = ["gshard_lm_decode.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo:model_registry",
        "//lingvo/core:gshard_decode",
        # Implicit numpy dependency.
    ],
)

lingvo_py_binary(
    name = "gshard_lm_decode",
    srcs = [":gshard_lm_decode_lib"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gshard_lm_decode_lib",
        "//lingvo:model_imports",
        # Implicit tensorflow grpc dependency.
    ],
)
