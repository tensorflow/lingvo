# buildifier: disable=same-origin-load
# Placeholder: load py_binary
# Placeholder: load py_library
# Placeholder: load py_test
# Placeholder: load sh_binary
load(
    "//lingvo:lingvo.bzl",
    "lingvo_cc_binary",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "gke_launch",
    srcs = ["gke_launch.py"],
    strict_deps = False,
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
    strict_deps = False,
    deps = [
        "//lingvo:compat",
    ],
)

py_binary(
    name = "bpe_word_tokenizer",
    srcs = ["bpe_word_tokenizer.py"],
    strict_deps = False,
    deps = ["//lingvo:compat"],
)

py_binary(
    name = "wpm_encode_file",
    srcs = ["wpm_encode_file.py"],
    strict_deps = False,
    deps = [
        ":wpm_encode_file_lib",
        # Implicit six dependency.
    ],
)

py_library(
    name = "wpm_encode_file_lib",
    srcs = ["wpm_encode_file.py"],
    strict_deps = False,
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
    strict_deps = False,
    deps = [
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_binary(
    name = "compute_stats",
    srcs = ["compute_stats.py"],
    strict_deps = False,
    deps = [
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "audio_lib",
    srcs = ["audio_lib.py"],
    strict_deps = False,
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
    strict_deps = False,
)

py_test(
    name = "beam_utils_test",
    srcs = ["beam_utils_test.py"],
    data = ["//lingvo/tasks/mt/testdata:wmt14_ende_tfexample"],
    strict_deps = False,
    deps = [
        ":beam_utils",
        # Implicit apache_beam dependency.
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

py_binary(
    name = "count_records",
    srcs = ["count_records.py"],
    strict_deps = False,
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
    strict_deps = False,
    deps = [
        ":audio_lib",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "compare_params_lib",
    srcs = ["compare_params.py"],
    strict_deps = False,
    deps = [
        "//lingvo:compat",
        "//lingvo:model_registry",
        # Implicit six dependency.
    ],
)

py_test(
    name = "compare_params_test",
    srcs = ["compare_params_test.py"],
    strict_deps = False,
    deps = [
        ":compare_params_lib",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:hyperparams",
        "//lingvo/core:test_utils",
    ],
)

py_binary(
    name = "create_asr_features",
    srcs = ["create_asr_features.py"],
    strict_deps = False,
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
        # Implicit absl.container.flat_hash_set dependency.
        "//third_party/protobuf",
        # Implicit tensorflow C++ proto dependency.
    ],
)

sh_binary(
    name = "generate_tf_dot_protos",
    srcs = ["generate_tf_dot_protos.sh"],
    data = [":generate_proto_def"],
)
