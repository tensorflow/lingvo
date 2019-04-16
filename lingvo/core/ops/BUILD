package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//lingvo:lingvo.bzl",
    "custom_kernel_library",
    "gen_op_cclib",
    "gen_op_pylib",
    "lingvo_cc_library",
    "lingvo_cc_test",
    "lingvo_proto_cc",
    "lingvo_proto_py",
)

cc_library(
    name = "x_ops_helper",
    hdrs = ["x_ops_helper.h"],
)

# Op definitions to be used from python.
gen_op_cclib(
    name = "x_ops",
    srcs = ["x_ops.cc"],
    deps = [":x_ops_helper"],
)

gen_op_pylib(
    name = "py_x_ops",
    srcs = ["py_x_ops.py"],
    cc_lib_name = "x_ops",
    kernel_deps = [":op_kernels"],
)

# Op kernel deps.
cc_library(
    name = "op_kernels",
    deps = [
        ":assert_kernels",
        ":beam_search_step_op_kernels",
        ":best_step_op_kernels",
        ":functional_ops_kernels",
        ":generic_input_op_kernels",
        ":random_ops_kernels",
        ":tokenizer_ops_kernels",
    ],
)

lingvo_cc_library(
    name = "mutex",
    hdrs = ["mutex.h"],
    deps = [
        "@nsync_includes//:includes",
    ],
)

lingvo_cc_library(
    name = "rope",
    hdrs = ["rope.h"],
    deps = [
        # Implicit rope dependency.
    ],
)

lingvo_cc_library(
    name = "ascii_tokenizer",
    srcs = ["ascii_tokenizer.cc"],
    hdrs = ["ascii_tokenizer.h"],
)

lingvo_cc_library(
    name = "simple_vocab",
    srcs = ["simple_vocab.cc"],
    hdrs = ["simple_vocab.h"],
)

py_test(
    name = "simple_vocab_test",
    srcs = ["simple_vocab_test.py"],
    deps = [
        ":py_x_ops",
        # Implicit tensorflow dependency.
        "//lingvo/core:test_utils",
    ],
)

lingvo_cc_library(
    name = "record",
    srcs = [
        "record_batcher.cc",
        "record_debug.cc",
        "record_yielder.cc",
        "sequential_record_yielder.cc",
        "weighted_mix_record_yielder.cc",
    ],
    hdrs = [
        "record_batcher.h",
        "record_yielder.h",
        "sequential_record_yielder.h",
        "weighted_mix_record_yielder.h",
    ],
    deps = [
        ":mutex",
        ":rope",
    ],
)

lingvo_cc_test(
    name = "record_yielder_test",
    srcs = ["record_yielder_test.cc"],
    deps = [
        ":input_common",
        ":record",
    ],
)

lingvo_cc_test(
    name = "weighted_mix_record_yielder_test",
    srcs = ["weighted_mix_record_yielder_test.cc"],
    deps = [
        ":input_common",
        ":record",
    ],
)

lingvo_cc_test(
    name = "record_batcher_test",
    srcs = ["record_batcher_test.cc"],
    deps = [
        ":input_common",
        ":record",
    ],
)

lingvo_cc_library(
    name = "input_common",
    srcs = ["input_common.cc"],
    hdrs = ["input_common.h"],
    deps = [
        ":record",
    ],
)

###################### Op kernel implementations.

custom_kernel_library(
    name = "assert_kernels",
    srcs = ["assert_kernels.cc"],
    op_def_lib = [":x_ops"],
)

py_test(
    name = "assert_ops_test",
    srcs = ["assert_ops_test.py"],
    deps = [
        ":py_x_ops",
        # Implicit tensorflow dependency.
        "//lingvo/core:test_utils",
    ],
)

custom_kernel_library(
    name = "beam_search_step_op_kernels",
    srcs = ["beam_search_step_op_kernels.cc"],
    hdrs = ["beam_search_step_op_kernels.h"],
    op_def_lib = [":x_ops"],
    deps = [
        ":hyps_proto",
        ":simple_vocab",
    ],
)

py_test(
    name = "beam_search_step_op_test",
    srcs = ["beam_search_step_op_test.py"],
    deps = [
        ":hyps_py_pb2",
        ":py_x_ops",
        # Implicit python proto dependency.
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:test_utils",
    ],
)

lingvo_cc_test(
    name = "beam_search_step_op_top_k_test",
    srcs = ["beam_search_step_op_top_k_test.cc"],
    deps = [
        ":beam_search_step_op_kernels",
    ],
)

custom_kernel_library(
    name = "best_step_op_kernels",
    srcs = ["best_step_op_kernels.cc"],
    op_def_lib = [":x_ops"],
    deps = [
        # Implicit tensorflow proto dependency.
    ],
)

py_test(
    name = "best_step_op_test",
    srcs = ["best_step_op_test.py"],
    data = [
        "//lingvo/core/ops/testdata:best_step_testdata",
    ],
    deps = [
        ":py_x_ops",
        # Implicit tensorflow dependency.
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

custom_kernel_library(
    name = "functional_ops_kernels",
    srcs = ["functional_ops_kernels.cc"],
    op_def_lib = [":x_ops"],
    deps = [
        ":mutex",
    ],
)

py_test(
    name = "functional_ops_test",
    srcs = ["functional_ops_test.py"],
    deps = [
        ":py_x_ops",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow python framework_for_generated_wrappers dependency.
        "//lingvo/core:test_utils",
    ],
)

custom_kernel_library(
    name = "generic_input_op_kernels",
    srcs = ["generic_input_op_kernels.cc"],
    op_def_lib = [":x_ops"],
    deps = [":input_common"],
)

py_test(
    name = "generic_input_op_test",
    srcs = ["generic_input_op_test.py"],
    deps = [
        ":py_x_ops",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:test_utils",
    ],
)

custom_kernel_library(
    name = "random_ops_kernels",
    srcs = ["random_ops_kernels.cc"],
    op_def_lib = [":x_ops"],
    deps = [
        ":mutex",
    ],
)

py_test(
    name = "random_ops_test",
    srcs = ["random_ops_test.py"],
    deps = [
        ":py_x_ops",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core:test_utils",
    ],
)

lingvo_cc_library(
    name = "tokenizer_op_headers",
    hdrs = ["tokenizer_op_headers.h"],
)

custom_kernel_library(
    name = "tokenizer_ops_kernels",
    srcs = ["tokenizer_ops_kernels.cc"],
    hdrs = ["tokenizer_op_headers.h"],
    op_def_lib = [":x_ops"],
    deps = [
        ":ascii_tokenizer",
        ":simple_vocab",
    ],
)

py_test(
    name = "tokenizer_ops_test",
    srcs = ["tokenizer_ops_test.py"],
    data = [
        "//lingvo/core/ops/testdata:bpe_codes_vocab",
        "//lingvo/core/ops/testdata:bpe_words_vocab",
        "//lingvo/core/ops/testdata:test_ngrams",
        "//lingvo/core/ops/testdata:test_vocab",
    ],
    deps = [
        ":py_x_ops",
        # Implicit tensorflow dependency.
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
    ],
)

lingvo_proto_cc(
    name = "hyps_proto",
    src = "hyps.proto",
)

lingvo_proto_py(
    name = "hyps_py_pb2",
    src = "hyps.proto",
    deps = [":hyps_proto"],
)
