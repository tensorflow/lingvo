load(
    "//lingvo:lingvo.bzl",
    "lingvo_cuda_py_test",
    "lingvo_proto_cc",
    "lingvo_proto_py",
    "lingvo_py_binary",
)
load("//lingvo:lingvo.bzl", "pytype_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "activations",
    srcs = ["activations.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)

lingvo_cuda_py_test(
    name = "activations_test",
    srcs = ["activations_test.py"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":activations",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "attention",
    srcs = ["attention.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":layers",
        ":py_utils",
        ":quant_utils",
        ":symbolic",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

lingvo_cuda_py_test(
    name = "attention_test",
    srcs = ["attention_test.py"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":attention",
        ":py_utils",
        ":quant_utils",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "base_decoder",
    srcs = ["base_decoder.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":beam_search_helper",
        ":py_utils",
        ":target_sequence_sampler",
        "//lingvo:compat",
    ],
)

pytype_library(
    name = "batch_major_attention",
    srcs = ["batch_major_attention.py"],
    srcs_version = "PY3",
    deps = [
        ":attention_util",
        ":base_layer",
        ":builder",
        ":computation_cost",
        ":conv_layers_builder",
        ":favor_attention",
        ":gpipe",
        ":gshard_layers",
        ":gshard_utils",
        ":hyperparams",
        ":layers",
        ":layers_with_attention",
        ":py_utils",
        ":quant_utils",
        ":repeat_layer",
        ":scatter_update",
        ":symbolic",
        ":tshape",
        # Implicit absl.logging dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "batch_utils",
    srcs = ["batch_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        ":py_utils",
    ],
)

py_test(
    name = "batch_utils_test",
    srcs = ["batch_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":batch_utils",
        ":cluster_factory",
        # Implicit absl.testing.flagsaver dependency.
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
    ],
)

py_library(
    name = "beam_search_helper",
    srcs = [
        "beam_search_helper.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        "//lingvo:compat",
        "//lingvo/core/ops",
    ],
)

py_test(
    name = "beam_search_helper_test",
    srcs = ["beam_search_helper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":beam_search_helper",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "bfloat16_variables",
    srcs = ["bfloat16_variables.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
    ],
)

py_test(
    name = "bfloat16_variables_test",
    srcs = ["bfloat16_variables_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":bfloat16_variables",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY3",
)

py_library(
    name = "conformer_layer",
    srcs = ["conformer_layer.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":base_layer",
        ":batch_major_attention",
        ":bn_layers",
        ":conv_layers_with_time_padding",
        ":gshard_builder",
        ":gshard_utils",
        ":hyperparams",
        ":layers",
        ":layers_with_attention",
        ":plot",
        ":py_utils",
        ":recurrent",
        ":summary_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "conformer_layer_test",
    timeout = "long",
    srcs = ["conformer_layer_test.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY3",
    deps = [
        ":conformer_layer",
        ":py_utils",
        ":stream_step_test_base",
        # Implicit absl.testing.flagsaver dependency.
        # Implicit absl.testing.parameterized dependency.
        "//lingvo/core:bn_layers",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "stream_step_test_base",
    srcs = ["stream_step_test_base.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        # Implicit absl.testing.flagsaver dependency.
        # Implicit absl.testing.parameterized dependency.
        "//lingvo/core:bn_layers",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "target_sequence_sampler",
    srcs = [
        "target_sequence_sampler.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        ":recurrent",
        "//lingvo:compat",
    ],
)

py_test(
    name = "target_sequence_sampler_test",
    srcs = ["target_sequence_sampler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        ":target_sequence_sampler",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "base_input_generator",
    srcs = ["base_input_generator.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":batch_utils",
        ":cluster",
        ":cluster_factory",
        ":datasource",
        ":hyperparams",
        ":input_generator_helper",
        ":inspect_utils",
        ":py_utils",
        ":tokenizers",
        ":tpu_embedding_layers",
        "//lingvo:compat",
        "//lingvo/core/ops",
    ],
)

py_test(
    name = "base_input_generator_test",
    srcs = ["base_input_generator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_input_generator",
        ":cluster_factory",
        ":hyperparams",
        ":py_utils",
        ":summary_utils",
        ":test_utils",
        # Implicit absl.testing.flagsaver dependency.
        "//lingvo:compat",
        # Implicit mock dependency.
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "batch_major_attention_test",
    srcs = ["batch_major_attention_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        ":attention",
        ":attention_util",
        ":base_layer",
        ":batch_major_attention",
        ":hyperparams",
        ":py_utils",
        ":stream_step_test_base",
        ":test_utils",
        # Implicit absl.testing.flagsaver dependency.
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

pytype_library(
    name = "base_layer",
    srcs = ["base_layer.py"],
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        ":hyperparams",
        ":py_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "bn_layers",
    srcs = ["bn_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        ":summary_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "base_layer_test",
    size = "small",
    srcs = ["base_layer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":hyperparams",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "favor_attention",
    srcs = ["favor_attention.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "favor_attention_test",
    size = "medium",
    srcs = [
        "favor_attention_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":favor_attention",
        ":test_utils",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "flat_beam_search_helper",
    srcs = ["flat_beam_search_helper.py"],
    srcs_version = "PY3",
    deps = [
        ":tpu_summary",
        "//lingvo:compat",
    ],
)

py_test(
    name = "flat_beam_search_helper_test",
    srcs = ["flat_beam_search_helper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":flat_beam_search_helper",
        ":test_utils",
        ":tpu_summary",
        # Implicit absl.testing.parameterized dependency.
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "reversible_layers",
    srcs = ["reversible_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "reversible_layers_test",
    srcs = ["reversible_layers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":layers",
        ":py_utils",
        ":reversible_layers",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "decoder_lib",
    srcs = ["decoder_lib.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_library(
    name = "base_model",
    srcs = ["base_model.py"],
    srcs_version = "PY3",
    deps = [
        ":base_input_generator",
        ":base_layer",
        ":build_data",
        ":cluster_factory",
        ":early_stop",
        ":hyperparams",
        ":learner",
        ":optimizer",
        ":pruning_utils",
        ":py_utils",
        ":schedule",
        ":summary_utils",
        ":task_scheduler",
        ":tpu_embedding_layers",
        # Implicit model_pruning.pruning dependency.
        "//lingvo:compat",
        "//lingvo/core:decoder_lib",
        "//lingvo/core:input_policy",
    ],
)

py_test(
    name = "base_model_test",
    srcs = ["base_model_test.py"],
    args = ["--noenable_eager_execution"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_decoder",
        ":base_input_generator",
        ":base_layer",
        ":base_model",
        ":base_model_params",
        ":distillation_task",
        ":hyperparams",
        ":layers",
        ":learner",
        ":py_utils",
        ":task_scheduler",
        ":test_utils",
        # Implicit absl.testing.flagsaver dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "base_model_eager_test",
    srcs = ["base_model_test.py"],
    args = ["--enable_eager_execution"],
    main = "base_model_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_decoder",
        ":base_input_generator",
        ":base_layer",
        ":base_model",
        ":base_model_params",
        ":distillation_task",
        ":hyperparams",
        ":layers",
        ":learner",
        ":py_utils",
        ":task_scheduler",
        ":test_utils",
        # Implicit absl.testing.flagsaver dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "base_model_params",
    srcs = ["base_model_params.py"],
    srcs_version = "PY3",
    deps = [
        ":base_input_generator",
        ":base_model",
        ":hyperparams",
        "//lingvo:datasets_lib",
    ],
)

py_test(
    name = "base_model_params_test",
    srcs = ["base_model_params_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_model_params",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "build_data",
    srcs = ["build_data.py"],
    srcs_version = "PY3",
    deps = [
        ":hyperparams",
        # Implicit build_data dependency.
    ],
)

py_library(
    name = "builder",
    srcs = ["builder.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":builder_layers",
        ":hyperparams",
        ":layers",
        ":py_utils",
        ":tshape",
        "//lingvo:compat",
    ],
)

py_library(
    name = "builder_layers",
    srcs = ["builder_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":computation_cost",
        ":py_utils",
        ":quant_utils",
        ":recurrent",
        ":summary_utils",
        ":symbolic",
        ":tshape",
        "//lingvo:compat",
    ],
)

py_test(
    name = "builder_layers_test",
    srcs = ["builder_layers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":builder_layers",
        ":layers",
        ":py_utils",
        ":test_utils",
        ":tshape",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "builder_test",
    srcs = ["builder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":builder",
        ":cluster_factory",
        ":layers",
        ":py_utils",
        ":test_utils",
        ":tshape",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit sympy dependency.
    ],
)

py_library(
    name = "cluster",
    srcs = ["cluster.py"],
    srcs_version = "PY3",
    deps = [
        ":hyperparams",
        ":nested_map",
        ":thread_local_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "cluster_factory",
    srcs = ["cluster_factory.py"],
    srcs_version = "PY3",
    deps = ["//lingvo/core:cluster"],
)

py_test(
    name = "cluster_test",
    size = "small",
    srcs = ["cluster_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cluster",
        ":cluster_factory",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "datasource",
    srcs = ["datasource.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":batch_utils",
        ":cluster",
        ":py_utils",
        "//lingvo:compat",
        # Implicit tensorflow_datasets dependency.
    ],
)

py_test(
    name = "datasource_test",
    size = "small",
    srcs = ["datasource_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_input_generator",
        ":cluster_factory",
        ":datasource",
        ":generic_input",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit tensorflow_datasets dependency.
    ],
)

py_library(
    name = "matrix_functions",
    srcs = ["matrix_functions.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "matrix_functions_test",
    size = "small",
    srcs = ["matrix_functions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":distributed_shampoo",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "distributed_shampoo",
    srcs = ["distributed_shampoo.py"],
    srcs_version = "PY3",
    deps = [
        ":matrix_functions",
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "distributed_shampoo_test",
    size = "small",
    srcs = ["distributed_shampoo_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":distributed_shampoo",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "early_stop",
    srcs = ["early_stop.py"],
    srcs_version = "PY3",
    deps = [
        ":hyperparams",
        ":py_utils",
        "//lingvo:compat",
        "//lingvo/core/ops",
    ],
)

py_test(
    name = "early_stop_test",
    size = "small",
    srcs = ["early_stop_test.py"],
    data = [
        "//lingvo/core/ops/testdata:best_step_testdata",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":early_stop",
        ":hyperparams",
        ":test_helper",
        ":test_utils",
        "//lingvo:compat",
    ],
)

lingvo_proto_cc(
    name = "inference_graph_proto",
    src = "inference_graph.proto",
)

lingvo_proto_py(
    name = "inference_graph_py_pb2",
    src = "inference_graph.proto",
    deps = [":inference_graph_proto"],
)

pytype_library(
    name = "hyperparams",
    srcs = ["hyperparams.py"],
    srcs_version = "PY3",
    deps = [
        ":hyperparams_py_pb2",
        ":symbolic",
        # Implicit python proto dependency.
        "//lingvo:compat",
        # Implicit typing_extensions dependency.
    ],
)

lingvo_proto_cc(
    name = "hyperparams_proto",
    src = "hyperparams.proto",
)

lingvo_proto_py(
    name = "hyperparams_py_pb2",
    src = "hyperparams.proto",
    deps = [":hyperparams_proto"],
)

py_test(
    name = "hyperparams_test",
    srcs = ["hyperparams_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hyperparams",
        ":hyperparams_py_pb2",
        ":symbolic",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "inference_graph_exporter",
    srcs = ["inference_graph_exporter.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model",
        ":bfloat16_variables",
        ":inference_graph_py_pb2",
        ":py_utils",
        # Implicit python proto dependency.
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "inference_graph_exporter_test",
    srcs = ["inference_graph_exporter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_input_generator",
        ":base_model",
        ":base_model_params",
        ":inference_graph_exporter",
        ":inference_graph_py_pb2",
        ":predictor_lib",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
        "//lingvo:model_registry",
    ],
)

py_library(
    name = "input_generator_helper",
    srcs = ["input_generator_helper.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "input_generator_helper_test",
    srcs = ["input_generator_helper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":input_generator_helper",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "inspect_utils",
    srcs = ["inspect_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "inspect_utils_test",
    srcs = ["inspect_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hyperparams",
        ":inspect_utils",
        ":test_utils",
    ],
)

pytype_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":base_layer",
        ":bn_layers",
        ":builder_layers",
        ":computation_cost",
        ":conv_layers_with_time_padding",
        ":gshard_utils",
        ":pruning_utils",
        ":py_utils",
        ":quant_utils",
        ":recurrent",
        ":schedule",
        ":summary_utils",
        ":symbolic",
        ":tshape",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit sympy dependency.
    ],
)

py_library(
    name = "learner",
    srcs = ["learner.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":gradient_combiner",
        ":optimizer",
        ":py_utils",
        ":schedule",
        ":summary_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "learner_test",
    srcs = ["learner_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":layers",
        ":learner",
        ":optimizer",
        ":py_utils",
        ":schedule",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "conv_layers_with_time_padding",
    srcs = ["conv_layers_with_time_padding.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":base_layer",
        ":bn_layers",
        ":py_utils",
        ":quant_utils",
        ":symbolic",
        ":tshape",
        "//lingvo:compat",
    ],
)

py_library(
    name = "conv_layers_builder",
    srcs = ["conv_layers_builder.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":base_layer",
        ":builder",
        ":builder_layers",
        ":conv_layers_with_time_padding",
        ":py_utils",
        # Implicit absl.flags dependency.
        "//lingvo:compat",
    ],
)

lingvo_cuda_py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    args = ["--noenable_eager_execution"],
    python_version = "PY3",
    shard_count = 20,
    deps = [":layers_test_lib"],
)

lingvo_cuda_py_test(
    name = "conv_layers_builder_test",
    srcs = ["conv_layers_builder_test.py"],
    python_version = "PY3",
    deps = [
        ":bn_layers",
        ":conv_layers_builder",
        ":conv_layers_with_time_padding",
        ":layers",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

lingvo_cuda_py_test(
    name = "conv_layers_with_time_padding_test",
    srcs = ["conv_layers_with_time_padding_test.py"],
    python_version = "PY3",
    deps = [
        ":layers_test_lib",
        ":stream_step_test_base",
        # Implicit absl.testing.flagsaver dependency.
    ],
)

py_library(
    name = "layers_test_lib",
    testonly = 1,
    srcs = ["layers_test.py"],
    srcs_version = "PY3",
    deps = [
        ":bn_layers",
        ":cluster_factory",
        ":gpipe",
        ":layers",
        ":py_utils",
        ":quant_utils",
        ":symbolic",
        ":test_utils",
        ":tshape",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "saver",
    srcs = ["saver.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        # Implicit python proto dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "schedule",
    srcs = ["schedule.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":early_stop",
        ":py_utils",
        "//lingvo:compat",
        "//lingvo/core/ops",
    ],
)

py_test(
    name = "saver_test",
    srcs = ["saver_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        ":py_utils",
        ":saver",
        ":test_utils",
        "//lingvo:compat",
        "//lingvo/tasks/image/params:mnist",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "schedule_test",
    size = "small",
    timeout = "moderate",
    srcs = ["schedule_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        ":early_stop",
        ":py_utils",
        ":schedule",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY3",
    deps = [
        ":hyperparams",
        ":plot",
        ":py_utils",
        ":scorers",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit sklearn dependency.  # buildcleaner: keep
    ],
)

py_library(
    name = "ml_perf_bleu_metric",
    srcs = ["ml_perf_bleu_metric.py"],
    srcs_version = "PY3",
    deps = [
        ":metrics",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_test(
    name = "ml_perf_bleu_metric_test",
    srcs = ["ml_perf_bleu_metric_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ml_perf_bleu_metric",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metrics",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "multitask_model",
    srcs = ["multitask_model.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model",
        ":py_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "multitask_model_test",
    srcs = ["multitask_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_input_generator",
        ":base_layer",
        ":base_model",
        ":base_model_params",
        ":hyperparams",
        ":multitask_model",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "adagraft",
    srcs = [
        "adagraft.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "adagraft_test",
    size = "small",
    srcs = ["adagraft_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adagraft",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    srcs_version = "PY3",
    deps = [
        ":adagraft",
        ":base_layer",
        ":distributed_shampoo",
        ":gshard_utils",
        ":py_utils",
        ":summary_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "optimizer_test",
    size = "small",
    srcs = ["optimizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        ":layers",
        ":optimizer",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "plot",
    srcs = ["plot.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        # Implicit Tkinter dependency.  # buildcleaner: keep
        "//lingvo:compat",
        # Implicit matplotlib dependency.
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_test(
    name = "plot_test",
    srcs = ["plot_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":plot",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "predictor_lib",
    srcs = ["predictor.py"],
    srcs_version = "PY3",
    deps = [
        ":inference_graph_py_pb2",
        ":py_utils",
        # Implicit python proto dependency.
        # Implicit IPython dependency.
        "//lingvo:compat",
        "//lingvo:model_imports_no_params",
    ],
)

lingvo_py_binary(
    name = "predictor",
    srcs = ["predictor.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":inference_graph_py_pb2",
        ":py_utils",
        # Implicit python proto dependency.
        "//lingvo:compat",
        "//lingvo:model_imports_no_params",
    ],
)

py_test(
    name = "predictor_test",
    size = "medium",
    srcs = ["predictor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_input_generator",
        ":base_model",
        ":inference_graph_exporter",
        ":inference_graph_py_pb2",
        ":predictor_lib",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "predictor_runner_base",
    srcs = ["predictor_runner_base.py"],
    srcs_version = "PY3",
    deps = [
        ":predictor_lib",
        ":py_utils",
        # Implicit absl.flags dependency.
        # Implicit concurrent.futures dependency.
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_library(
    name = "pruning_utils",
    srcs = ["pruning_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        # Implicit model_pruning.pruning dependency.
        "//lingvo:compat",
    ],
)

py_library(
    name = "py_utils_flags",
    srcs = ["py_utils_flags.py"],
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        "//lingvo:compat",
    ],
)

pytype_library(
    name = "py_utils",
    srcs = ["py_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        ":gshard_utils",
        ":hyperparams",
        ":nested_map",
        ":py_utils_flags",
        ":retry",
        ":symbolic",
        ":thread_local_utils",
        ":tshape",
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow py proto dependency.
    ],
)

py_test(
    name = "py_utils_test",
    size = "medium",
    srcs = ["py_utils_test.py"],
    args = ["--noenable_eager_execution"],
    data = [
        "//lingvo/core/testdata:lenet_test_model",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":builder_layers",
        ":cluster_factory",
        ":hyperparams",
        ":py_utils",
        ":py_utils_flags",
        ":recurrent",
        ":symbolic",
        ":test_helper",
        ":test_utils",
        # Implicit absl.testing.flagsaver dependency.
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        "//lingvo:model_registry",
        "//lingvo/tasks/image/params:mnist",
        # Implicit mock dependency.
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "py_utils_eager_test",
    srcs = ["py_utils_eager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":layers",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

pytype_library(
    name = "quant_utils",
    srcs = ["quant_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":hyperparams",
        ":py_utils",
        ":summary_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "quant_test_lib",
    srcs = ["quant_test_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        ":quant_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "quant_utils_test",
    size = "small",
    srcs = ["quant_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        ":quant_test_lib",
        ":quant_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "layers_with_gpipe",
    srcs = ["layers_with_gpipe.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":batch_major_attention",
        ":gpipe",
        ":layers",
        ":layers_with_attention",
        ":py_utils",
        ":tshape",
        "//lingvo:compat",
    ],
)

py_test(
    name = "layers_with_gpipe_test",
    size = "large",
    srcs = ["layers_with_gpipe_test.py"],
    python_version = "PY3",
    shard_count = 15,
    srcs_version = "PY3",
    deps = [
        ":layers_with_gpipe",
        ":py_utils",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "spectrum_augmenter",
    srcs = ["spectrum_augmenter.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "spectrum_augmenter_test",
    srcs = ["spectrum_augmenter_test.py"],
    python_version = "PY3",
    shard_count = 1,
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        ":spectrum_augmenter",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "spectrum_augmenter_on_device",
    srcs = ["spectrum_augmenter_on_device.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        ":spectrum_augmenter",
        "//lingvo:compat",
    ],
)

py_test(
    name = "spectrum_augmenter_on_device_test",
    srcs = ["spectrum_augmenter_on_device_test.py"],
    python_version = "PY3",
    shard_count = 1,
    srcs_version = "PY3",
    deps = [
        ":spectrum_augmenter",
        ":spectrum_augmenter_on_device",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "generic_input",
    srcs = ["generic_input.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit tensorflow python framework_for_generated_wrappers dependency.
    ],
)

py_test(
    name = "generic_input_test",
    srcs = ["generic_input_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":generic_input",
        ":py_utils",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "gpipe",
    srcs = ["gpipe.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":builder_layers",
        ":py_utils",
        ":recurrent",
        ":tshape",
        "//lingvo:compat",
    ],
)

py_test(
    name = "gpipe_test",
    srcs = ["gpipe_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":gpipe",
        ":layers",
        ":py_utils",
        ":test_utils",
        ":tshape",
        "//lingvo:compat",
    ],
)

py_library(
    name = "recurrent",
    srcs = ["recurrent.py"],
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        ":py_utils",
        ":scatter_update",
        ":sendrecv",
        ":symbolic",
        ":thread_local_utils",
        "//lingvo:compat",
    ],
)

lingvo_cuda_py_test(
    name = "recurrent_test",
    srcs = ["recurrent_test.py"],
    python_version = "PY3",
    shard_count = 6,
    srcs_version = "PY3",
    deps = [":recurrent_test_lib"],
)

py_library(
    name = "recurrent_test_lib",
    testonly = 1,
    srcs = ["recurrent_test.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":layers",
        ":py_utils",
        ":recurrent",
        ":symbolic",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "attention_util",
    srcs = ["attention_util.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":layers",
        ":py_utils",
        ":summary_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "retry",
    srcs = ["retry.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit retry dependency.
    ],
)

py_test(
    name = "attention_util_test",
    srcs = ["attention_util_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":attention_util",
        ":py_utils",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "retry_test",
    srcs = ["retry_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":retry",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "self_attention_layer",
    srcs = ["self_attention_layer.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":batch_major_attention",
        ":py_utils",
    ],
)

py_library(
    name = "sendrecv",
    srcs = ["sendrecv.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
    ],
)

py_test(
    name = "self_attention_layer_test",
    timeout = "long",
    srcs = ["self_attention_layer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":batch_major_attention",
        ":py_utils",
        ":self_attention_layer",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "sendrecv_test",
    srcs = ["sendrecv_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sendrecv",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "summary_utils",
    srcs = ["summary_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":base_input_generator",
        ":cluster_factory",
        ":plot",
        ":py_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "task_scheduler",
    srcs = ["task_scheduler.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":early_stop",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "task_scheduler_test",
    srcs = ["task_scheduler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":early_stop",
        ":task_scheduler",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "test_helper",
    srcs = ["test_helper.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        ":py_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "trainer_test_utils",
    srcs = ["trainer_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":base_input_generator",
        ":base_model",
        ":py_utils",
        ":summary_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "test_trainer_utils",
    srcs = ["test_trainer_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        "//lingvo:trainer_impl",
    ],
)

py_library(
    name = "wpm_encoder",
    srcs = ["wpm_encoder.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit six dependency.
    ],
)

py_test(
    name = "wpm_encoder_test",
    srcs = ["wpm_encoder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":test_utils",
        ":wpm_encoder",
        "//lingvo:compat",
    ],
)

py_library(
    name = "tokenizers",
    srcs = ["tokenizers.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        ":wpm_encoder",
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit numpy dependency.
        # Implicit tensorflow_text dependency.
    ],
)

py_test(
    name = "tokenizers_test",
    srcs = ["tokenizers_test.py"],
    data = [
        "//lingvo/core/testdata:en_1k_spm",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":test_helper",
        ":test_utils",
        ":tokenizers",
        "//lingvo:compat",
    ],
)

py_library(
    name = "tpu_summary",
    srcs = ["tpu_summary.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
    ],
)

py_test(
    name = "wpm_tokenizer_test",
    srcs = ["wpm_tokenizer_test.py"],
    data = [
        "//lingvo/tasks/mt:wpm_ende",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":test_helper",
        ":test_utils",
        ":tokenizers",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "routing_config_helper",
    srcs = ["routing_config_helper.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo/core:batch_major_attention",
    ],
)

py_test(
    name = "routing_config_helper_test",
    srcs = ["routing_config_helper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":routing_config_helper",
    ],
)

py_library(
    name = "rnn_cell",
    srcs = ["rnn_cell.py"],
    srcs_version = "PY3",
    deps = [
        ":hyperparams",
        ":layers",
        ":pruning_utils",
        ":py_utils",
        ":quant_utils",
        ":summary_utils",
        "//lingvo:compat",
        "//lingvo/core/steps:embedding_steps",
    ],
)

lingvo_cuda_py_test(
    name = "rnn_cell_test",
    srcs = ["rnn_cell_test.py"],
    python_version = "PY3",
    deps = [
        ":layers",
        ":py_utils",
        ":quant_utils",
        ":rnn_cell",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        "//lingvo/core/steps:embedding_steps",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "layers_with_attention",
    srcs = ["layers_with_attention.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":attention",
        ":base_layer",
        ":gshard_builder",
        ":gshard_layers",
        ":gshard_utils",
        ":hyperparams",
        ":layers",
        ":py_utils",
        ":symbolic",
        "//lingvo:compat",
    ],
)

py_test(
    name = "layers_with_attention_test",
    size = "large",
    srcs = ["layers_with_attention_test.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    deps = [
        ":gshard_builder",
        ":layers",
        ":layers_with_attention",
        ":py_utils",
        ":symbolic",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "rnn_layers",
    srcs = ["rnn_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":attention",
        ":base_layer",
        ":layers",
        ":py_utils",
        ":quant_utils",
        ":recurrent",
        ":rnn_cell",
        "//lingvo:compat",
    ],
)

lingvo_cuda_py_test(
    name = "rnn_layers_test",
    size = "large",
    srcs = ["rnn_layers_test.py"],
    python_version = "PY3",
    shard_count = 15,
    deps = [":rnn_layers_test_lib"],
)

py_library(
    name = "rnn_layers_test_lib",
    testonly = 1,
    srcs = ["rnn_layers_test.py"],
    srcs_version = "PY3",
    deps = [
        ":attention",
        ":base_layer",
        ":cluster_factory",
        ":layers_with_attention",
        ":py_utils",
        ":rnn_cell",
        ":rnn_layers",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "lstm_frnn_layer",
    srcs = ["lstm_frnn_layer.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":recurrent",
        ":rnn_cell",
        ":rnn_layers",
        # Implicit absl.flags dependency.
        "//lingvo:compat",
    ],
)

py_test(
    name = "lstm_frnn_layer_test",
    srcs = ["lstm_frnn_layer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":lstm_frnn_layer",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "scorers",
    srcs = ["scorers.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit six dependency.
    ],
)

py_test(
    name = "scorers_test",
    srcs = ["scorers_test.py"],
    data = ["//lingvo/core/ops/testdata:wmt"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":scorers",
        ":test_helper",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "entmax",
    srcs = ["entmax.py"],
    srcs_version = "PY3",
    visibility = [
        "//lingvo/core:__subpackages__",
    ],
    deps = [
        ":test_utils",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
    ],
)

py_test(
    name = "entmax_test",
    srcs = ["entmax_test.py"],
    python_version = "PY3",
    deps = [
        ":entmax",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "model_helper",
    srcs = ["model_helper.py"],
    srcs_version = "PY3",
    deps = [
        ":rnn_layers",
        "//lingvo:compat",
    ],
)

py_library(
    name = "gshard_builder",
    srcs = ["gshard_builder.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model",
        ":builder",
        ":flat_beam_search_helper",
        ":gshard_layers",
        ":layers",
        ":py_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "gshard_builder_test",
    srcs = ["gshard_builder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gshard_builder",
        ":gshard_layers",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "gshard_layers",
    srcs = ["gshard_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":base_layer",
        ":conv_layers_with_time_padding",
        ":gshard_utils",
        ":py_utils",
        ":recurrent",
        ":tpu_summary",
        ":var_tmp_wrappers",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "gshard_layers_test",
    srcs = ["gshard_layers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gshard_builder",
        ":gshard_layers",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

pytype_library(
    name = "gshard_utils",
    srcs = ["gshard_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils_flags",
        ":thread_local_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit sentencepiece dependency.
    ],
)

py_library(
    name = "gshard_decode",
    srcs = ["gshard_decode.py"],
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        ":gshard_utils",
        ":tpu_summary",
        "//lingvo:compat",
    ],
)

py_library(
    name = "tshape",
    srcs = ["tshape.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        # Implicit sympy dependency.
    ],
)

py_test(
    name = "tpu_summary_test",
    srcs = ["tpu_summary_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":test_utils",
        ":tpu_summary",
        "//lingvo:compat",
    ],
)

py_test(
    name = "tshape_test",
    srcs = ["tshape_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":test_utils",
        ":tshape",
        "//lingvo:compat",
    ],
)

py_library(
    name = "symbolic",
    srcs = ["symbolic.py"],
    srcs_version = "PY3",
    deps = [
        # Implicit sympy dependency.
    ],
)

py_test(
    name = "symbolic_test",
    srcs = ["symbolic_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":symbolic",
        ":test_utils",
        "//lingvo:compat",
        # Implicit sympy dependency.
    ],
)

py_library(
    name = "insertion",
    srcs = ["insertion.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        "//lingvo:compat",
    ],
)

lingvo_cuda_py_test(
    name = "insertion_test",
    srcs = ["insertion_test.py"],
    python_version = "PY3",
    deps = [
        ":insertion",
        ":test_utils",
    ],
)

py_library(
    name = "computation_cost",
    srcs = ["computation_cost.py"],
    srcs_version = "PY3",
    deps = [
        ":bn_layers",
        "//lingvo:compat",
    ],
)

py_library(
    name = "checkpointer_lib",
    srcs = ["checkpointer.py"],
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        ":py_utils",
        ":saver",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_library(
    name = "program_lib",
    srcs = ["program.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model",
        ":checkpointer_lib",
        ":cluster_factory",
        ":hyperparams",
        ":metrics",
        ":ml_perf_log",
        ":py_utils",
        ":summary_utils",
        "//lingvo:base_trial",
        "//lingvo:compat",
        # Implicit tensorflow grpc dependency.
    ],
)

py_library(
    name = "step",
    srcs = ["step.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":builder_layers",
        ":py_utils",
        ":recurrent",
        "//lingvo:compat",
    ],
)

py_test(
    name = "step_test",
    size = "small",
    srcs = ["step_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":builder_layers",
        ":py_utils",
        ":step",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "summary_utils_test",
    srcs = ["summary_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cluster_factory",
        ":summary_utils",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "checkpointer_test",
    srcs = ["checkpointer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_input_generator",
        ":base_model",
        ":checkpointer_lib",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "ema_test",
    srcs = ["ema_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_decoder",
        ":base_input_generator",
        ":base_model",
        ":checkpointer_lib",
        ":layers",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "ml_perf_tokenizer",
    srcs = ["ml_perf_tokenizer.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        ":tokenizers",
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "ml_perf_log",
    srcs = ["ml_perf_log.py"],
    srcs_version = "PY3",
)

py_test(
    name = "ml_perf_log_test",
    srcs = ["ml_perf_log_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ml_perf_log",
        "//lingvo:compat",
    ],
)

py_library(
    name = "egdd",
    srcs = [
        "egdd.py",
    ],
    srcs_version = "PY3",
    deps = [
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "egdd_test",
    size = "large",
    srcs = ["egdd_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":egdd",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "distillation_task",
    srcs = ["distillation_task.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model",
        ":py_utils",
        ":schedule",
        "//lingvo:compat",
    ],
)

py_library(
    name = "gradient_combiner",
    srcs = ["gradient_combiner.py"],
    srcs_version = "PY3",
    deps = [":base_layer"],
)

py_library(
    name = "tpu_embedding_layers",
    srcs = ["tpu_embedding_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":hyperparams",
        ":py_utils",
        ":schedule",
        "//lingvo:compat",
    ],
)

py_library(
    name = "graddrop",
    srcs = ["graddrop.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
        "//lingvo/core:summary_utils",
    ],
)

py_test(
    name = "graddrop_test",
    srcs = ["graddrop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":graddrop",
        "//lingvo:compat",
        "//lingvo/core:builder_layers",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "input_policy",
    srcs = ["input_policy.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
    ],
)

py_library(
    name = "thread_local_utils",
    srcs = ["thread_local_utils.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

pytype_library(
    name = "nested_map",
    srcs = ["nested_map.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
        # Implicit typing_extensions dependency.
    ],
)

py_library(
    name = "repeat_layer",
    srcs = ["repeat_layer.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":py_utils",
        ":recurrent",
        "//lingvo:compat",
    ],
)

py_library(
    name = "scatter_update",
    srcs = ["scatter_update.py"],
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        ":thread_local_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "var_tmp_wrappers",
    srcs = ["var_tmp_wrappers.py"],
    srcs_version = "PY3",
    deps = [
        "//lingvo:compat",
    ],
)

py_test(
    name = "var_tmp_wrappers_test",
    srcs = ["var_tmp_wrappers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":py_utils",
        ":var_tmp_wrappers",
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)
