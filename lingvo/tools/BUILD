load("//lingvo:lingvo.bzl", "py2and3_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//lingvo:lingvo.bzl",
    "lingvo_cc_binary",
)

py_binary(
    name = "keras2ckpt",
    srcs = ["keras2ckpt.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        # Implicit tensorflow dependency.
    ],
)

py_binary(
    name = "bpe_word_tokenizer",
    srcs = ["bpe_word_tokenizer.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_binary(
    name = "wpm_encode_file",
    srcs = ["wpm_encode_file.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":wpm_encode_file_lib",
    ],
)

py_library(
    name = "wpm_encode_file_lib",
    srcs = ["wpm_encode_file.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo/core:wpm_encoder",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_binary(
    name = "print_tf_records",
    srcs = ["print_tf_records.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_binary(
    name = "compute_stats",
    srcs = ["compute_stats.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "audio_lib",
    srcs = ["audio_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        # Additional FFT kernels dependency.
        "//lingvo/core:py_utils",
        "//lingvo/tasks/asr:frontend",
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "audio_lib_test",
    srcs = ["audio_lib_test.py"],
    data = [
        "//lingvo/tools/testdata:audio_data",
    ],
    deps = [
        ":audio_lib",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
        # Implicit tensorflow dependency.
    ],
)

py_binary(
    name = "create_asr_features",
    srcs = ["create_asr_features.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":audio_lib",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

lingvo_cc_binary(
    name = "generate_proto_def",
    srcs = ["generate_proto_def.cc"],
    deps = [
        # Implicit proto dependency.
        # Implicit tensorflow proto dependency.
    ],
)

sh_binary(
    name = "generate_tf_dot_protos",
    srcs = ["generate_tf_dot_protos.sh"],
    data = [":generate_proto_def"],
)
