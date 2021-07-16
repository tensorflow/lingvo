load(
    "//lingvo:lingvo.bzl",
    "custom_kernel_library",
    "gen_op_cclib",
    "gen_op_pylib",
    "lingvo_cc_library",
    "lingvo_cc_test",
    "lingvo_cc_test_library",
    "lingvo_proto_cc",
    "lingvo_proto_py",
)

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(
    [
        "beam_search_step_op_kernels.cc",
        "beam_search_step_op_kernels.h",
    ],
)

cc_library(
    name = "x_ops_helper",
    hdrs = ["x_ops_helper.h"],
)

# Op definitions to be used from python.
gen_op_cclib(
    name = "x_ops",
    srcs = ["x_ops.cc"],
    deps = [
        ":x_ops_helper",
        # Implicit tensorflow C++ proto dependency.
    ],
)

gen_op_pylib(
    name = "py_x_ops",
    srcs = ["__init__.py"],
    cc_lib_name = "x_ops",
    kernel_deps = [
        ":op_kernels",
    ],
    py_deps = [
        "//lingvo:compat",
    ],
    visibility = ["//visibility:private"],
)

py_library(
    name = "ops",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":py_x_ops",  # buildcleaner: keep
        "//lingvo:compat",
    ],
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
        ":mass_op",
        ":ml_perf_subword_op",
        ":pack_ops",
        ":preconditioner_op_kernels",
        ":random_ops_kernels",
        ":static_map_op",
        ":tokenizer_ops_kernels",
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

custom_kernel_library(
    name = "ml_perf_subword_op",
    srcs = ["ml_perf_subword_op.cc"],
    hdrs = ["ml_perf_subword_op.h"],
    op_def_lib = [":x_ops"],
    deps = [
        "@icu//:common",
    ],
)

py_test(
    name = "simple_vocab_test",
    srcs = ["simple_vocab_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        "//lingvo/core:test_utils",
        # Implicit tensorflow dependency.
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
        ":rope",
        ":versioned_file_set_proto_cc",
        # Implicit absl.strings dependency.
        # Implicit absl.synchronization dependency.
    ],
)

lingvo_cc_test(
    name = "record_yielder_test",
    srcs = ["record_yielder_test.cc"],
    deps = [
        ":input_common",
        ":record",
        ":yielder_test_helper",
        # Implicit absl.flags:flag dependency.
    ],
)

lingvo_cc_test(
    name = "weighted_mix_record_yielder_test",
    srcs = ["weighted_mix_record_yielder_test.cc"],
    deps = [
        ":input_common",
        ":record",
        ":yielder_test_helper",
        # Implicit absl.flags:flag dependency.
    ],
)

lingvo_cc_test(
    name = "record_batcher_test",
    srcs = ["record_batcher_test.cc"],
    deps = [
        ":input_common",
        ":record",
        # Implicit absl.flags:flag dependency.
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

lingvo_cc_test(
    name = "input_common_test",
    srcs = ["input_common_test.cc"],
    deps = [
        ":input_common",
        ":yielder_test_helper",
        # Implicit absl.flags:flag dependency.
    ],
)

lingvo_cc_test_library(
    name = "yielder_test_helper",
    srcs = [
        "yielder_test_helper.cc",
    ],
    hdrs = [
        "yielder_test_helper.h",
    ],
    deps = [
        ":record",
        # Implicit absl.flags:flag dependency.
    ],
)

lingvo_cc_library(
    name = "text_packing",
    srcs = ["text_packing.cc"],
    hdrs = ["text_packing.h"],
)

lingvo_cc_test(
    name = "text_packing_test",
    srcs = ["text_packing_test.cc"],
    deps = [
        ":text_packing",
    ],
)

###################### Op kernel implementations.

custom_kernel_library(
    name = "static_map_op",
    srcs = ["static_map_op.cc"],
    op_def_lib = [":x_ops"],
)

py_test(
    name = "static_map_op_test",
    srcs = ["static_map_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)

custom_kernel_library(
    name = "assert_kernels",
    srcs = ["assert_kernels.cc"],
    op_def_lib = [":x_ops"],
)

py_test(
    name = "assert_ops_test",
    srcs = ["assert_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        "//lingvo/core:test_utils",
        # Implicit tensorflow dependency.
    ],
)

custom_kernel_library(
    name = "beam_search_step_op_kernels",
    srcs = ["beam_search_step_op_kernels.cc"],
    hdrs = ["beam_search_step_op_kernels.h"],
    op_def_lib = [":x_ops"],
    deps = [
        ":hyps_proto_cc",
        # Implicit absl.strings dependency.
    ],
)

lingvo_cc_test(
    name = "beam_search_step_op_top_k_test",
    srcs = ["beam_search_step_op_top_k_test.cc"],
    deps = [
        ":beam_search_step_op_kernels",
    ],
)

py_test(
    name = "beam_search_step_op_test",
    srcs = ["beam_search_step_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hyps_py_pb2",
        ":ops",
        # Implicit python proto dependency.
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

custom_kernel_library(
    name = "best_step_op_kernels",
    srcs = ["best_step_op_kernels.cc"],
    op_def_lib = [":x_ops"],
    deps = [
        # Implicit tensorflow C++ proto dependency.
    ],
)

py_test(
    name = "best_step_op_test",
    srcs = ["best_step_op_test.py"],
    data = [
        "//lingvo/core/ops/testdata:best_step_testdata",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        "//lingvo:compat",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
        # Implicit tensorflow dependency.
    ],
)

custom_kernel_library(
    name = "functional_ops_kernels",
    srcs = ["functional_ops_kernels.cc"],
    op_def_lib = [":x_ops"],
    deps = [
        # Implicit absl.synchronization dependency.
    ],
)

py_test(
    name = "functional_ops_test",
    srcs = ["functional_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow python framework_for_generated_wrappers dependency.
    ],
)

lingvo_cc_library(
    name = "thread_local_runner",
    srcs = ["thread_local_runner.cc"],
    hdrs = ["thread_local_runner.h"],
)

custom_kernel_library(
    name = "generic_input_op_kernels",
    srcs = ["generic_input_op_kernels.cc"],
    op_def_lib = [":x_ops"],
    deps = [
        ":input_common",
        ":thread_local_runner",
        # Implicit absl.memory dependency.
    ],
)

custom_kernel_library(
    name = "random_ops_kernels",
    srcs = ["random_ops_kernels.cc"],
    op_def_lib = [":x_ops"],
    deps = [
        # Implicit absl.synchronization dependency.
    ],
)

py_test(
    name = "random_ops_test",
    srcs = ["random_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit tensorflow dependency.
    ],
)

lingvo_cc_library(
    name = "preconditioner_captain",
    srcs = ["preconditioner_captain.cc"],
    hdrs = ["preconditioner_captain.h"],
    deps = [
        # Implicit tensorflow core_cpu dependency.
        # Implicit tensorflow C++ proto dependency.
    ],
)

custom_kernel_library(
    name = "preconditioner_op_kernels",
    srcs = ["preconditioner_op_kernels.cc"],
    op_def_lib = [":x_ops"],
    deps = [
        ":preconditioner_captain",
    ],
)

py_test(
    name = "preconditioner_op_kernels_test",
    srcs = ["preconditioner_op_kernels_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
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
        "//lingvo/core/ops/testdata:mlperf_vocab",
        "//lingvo/core/ops/testdata:test_ngrams",
        "//lingvo/core/ops/testdata:test_vocab",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        "//lingvo/core:test_helper",
        "//lingvo/core:test_utils",
        # Implicit tensorflow dependency.
    ],
)

custom_kernel_library(
    name = "pack_ops",
    srcs = ["pack_ops.cc"],
    op_def_lib = [":x_ops"],
    deps = [
        ":text_packing",
        # Implicit absl.container.flat_hash_map dependency.
        # Implicit absl.strings dependency.
        # Implicit absl.synchronization dependency.
    ],
)

py_test(
    name = "pack_ops_test",
    srcs = ["pack_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

custom_kernel_library(
    name = "mass_op",
    srcs = ["mass_op.cc"],
    op_def_lib = [":x_ops"],
)

py_test(
    name = "mass_op_test",
    srcs = ["mass_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ops",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
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

lingvo_proto_cc(
    name = "record_proto",
    src = "record.proto",
    deps = [
        # Implicit tensorflow proto dependency.
    ],
)

lingvo_proto_cc(
    name = "versioned_file_set_proto",
    src = "versioned_file_set.proto",
)

lingvo_proto_py(
    name = "versioned_file_set_py_pb2",
    src = "versioned_file_set.proto",
    deps = [":versioned_file_set_proto"],
)

lingvo_proto_py(
    name = "record_py_pb2",
    src = "record.proto",
    deps = [":record_proto"],
)
