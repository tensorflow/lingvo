load(
    "//lingvo:lingvo.bzl",
    "lingvo_cc_binary",
)

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "gke_launch",
    srcs = ["gke_launch.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # Implicit absl.app dependency.
        # Implicit absl.flags dependency.
        # Implicit six dependency.
        # Implicit yaml dependency.
    ],
)

py_binary(
    name = "keras2ckpt",
    srcs = ["keras2ckpt.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
    ],
)

py_binary(
    name = "bpe_word_tokenizer",
    srcs = ["bpe_word_tokenizer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//lingvo:compat"],
)

py_binary(
    name = "wpm_encode_file",
    srcs = ["wpm_encode_file.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":wpm_encode_file_lib",
        # Implicit six dependency.
    ],
)

py_library(
    name = "wpm_encode_file_lib",
    srcs = ["wpm_encode_file.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:wpm_encoder",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_binary(
    name = "print_tf_records",
    srcs = ["print_tf_records.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_binary(
    name = "compute_stats",
    srcs = ["compute_stats.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "audio_lib",
    srcs = ["audio_lib.py"],
    srcs_version = "PY3",
    deps = [
        # Additional FFT kernels dependency.
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/asr:frontend",
    ],
)

py_library(
    name = "beam_utils",
    srcs = ["beam_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "beam_utils_test",
    srcs = ["beam_utils_test.py"],
    data = ["//lingvo/tasks/mt/testdata:wmt14_ende_tfexample"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":beam_utils",
        # Implicit apache_beam dependency.
        "//lingvo:compat",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

py_binary(
    name = "count_records",
    srcs = ["count_records.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":beam_utils",
        # Implicit network file system dependency.
        # Implicit apache_beam dependency.
    ],
)

py_test(
    name = "audio_lib_test",
    srcs = ["audio_lib_test.py"],
    data = [
        "//lingvo/tools/testdata:audio_data",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":audio_lib",
        "//lingvo:compat",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "compare_params_lib",
    srcs = ["compare_params.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo:model_registry",
        # Implicit six dependency.
    ],
)

py_test(
    name = "compare_params_test",
    srcs = ["compare_params_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":compare_params_lib",
        "//lingvo:compat",
        "//lingvo/core:hyperparams",
        "//lingvo/core:test_utils",
    ],
)

py_binary(
    name = "create_asr_features",
    srcs = ["create_asr_features.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":audio_lib",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
    ],
)

lingvo_cc_binary(
    name = "generate_proto_def",
    srcs = ["generate_proto_def.cc"],
    deps = [
        # Implicit proto dependency.
        # Implicit tensorflow C++ proto dependency.
    ],
)

sh_binary(
    name = "generate_tf_dot_protos",
    srcs = ["generate_tf_dot_protos.sh"],
    data = [":generate_proto_def"],
)
