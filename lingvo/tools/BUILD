package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "keras2ckpt",
    srcs = ["keras2ckpt.py"],
    deps = [
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "wpm_encoder",
    srcs = ["wpm_encoder.py"],
    deps = [
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "wpm_encoder_test",
    srcs = ["wpm_encoder_test.py"],
    deps = [
        ":wpm_encoder",
        # Implicit tensorflow dependency.
    ],
)

py_binary(
    name = "wpm_encode_file",
    srcs = ["wpm_encode_file.py"],
    deps = [
        ":wpm_encoder",
        # Implicit tensorflow dependency.
    ],
)
