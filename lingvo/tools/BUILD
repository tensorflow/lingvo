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
    deps = [
        # Implicit tensorflow dependency.
    ],
)

py_binary(
    name = "bpe_word_tokenizer",
    srcs = ["bpe_word_tokenizer.py"],
    deps = [
        # Implicit tensorflow dependency.
    ],
)

py_binary(
    name = "wpm_encode_file",
    srcs = ["wpm_encode_file.py"],
    deps = [
        # Implicit tensorflow dependency.
        "//lingvo/core:wpm_encoder",
    ],
)

py_binary(
    name = "print_tf_records",
    srcs = ["print_tf_records.py"],
    deps = [
        # Implicit tensorflow dependency.
    ],
)

py_binary(
    name = "compute_stats",
    srcs = ["compute_stats.py"],
    deps = [
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "audio_lib",
    srcs = ["audio_lib.py"],
    deps = [
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
        # Implicit tensorflow dependency.
        "//lingvo/core:test_helper",
    ],
)

py_binary(
    name = "create_asr_features",
    srcs = ["create_asr_features.py"],
    deps = [
        ":audio_lib",
        # Implicit tensorflow dependency.
        "//lingvo/core:asr_frontend",
    ],
)

lingvo_cc_binary(
    name = "generate_proto_def",
    srcs = ["generate_proto_def.cc"],
    deps = [
        # Implicit tensorflow proto dependency.
    ],
)

sh_binary(
    name = "generate_tf_dot_protos",
    srcs = ["generate_tf_dot_protos.sh"],
    data = [":generate_proto_def"],
)
