load("//lingvo:lingvo.bzl", "py2and3_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//lingvo:lingvo.bzl",
    "lingvo_cuda_py2and3_test",
    "lingvo_proto_cc",
    "lingvo_proto_py",
    "lingvo_py_binary",
)

py_library(
    name = "attention",
    srcs = ["attention.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":layers",
        ":py_utils",
        ":quant_utils",
        ":summary_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

lingvo_cuda_py2and3_test(
    name = "attention_test",
    srcs = ["attention_test.py"],
    shard_count = 5,
    deps = [
        ":attention",
        ":py_utils",
        ":quant_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "base_decoder",
    srcs = ["base_decoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":beam_search_helper",
        ":target_sequence_sampler",
    ],
)

py_library(
    name = "beam_search_helper",
    srcs = [
        "beam_search_helper.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":py_utils",
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit six dependency.
    ],
)

py_test(
    name = "beam_search_helper_test",
    srcs = ["beam_search_helper_test.py"],
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
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
    ],
)

py_test(
    name = "bfloat16_variables_test",
    srcs = ["bfloat16_variables_test.py"],
    deps = [
        ":bfloat16_variables",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "target_sequence_sampler",
    srcs = [
        "target_sequence_sampler.py",
    ],
    srcs_version = "PY2AND3",
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
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":datasource",
        ":hyperparams",
        ":input_generator_helper",
        ":py_utils",
        ":tokenizers",
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit six dependency.
    ],
)

py_test(
    name = "base_input_generator_test",
    srcs = ["base_input_generator_test.py"],
    deps = [
        ":base_input_generator",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "base_layer",
    srcs = ["base_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cluster_factory",
        ":hyperparams",
        ":py_utils",
        ":symbolic",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_library(
    name = "bn_layers",
    srcs = ["bn_layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":py_utils",
        ":summary_utils",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "base_layer_test",
    size = "small",
    srcs = ["base_layer_test.py"],
    deps = [
        ":base_layer",
        ":hyperparams",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "decoder_lib",
    srcs = ["decoder_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
    ],
)

py_library(
    name = "base_model",
    srcs = ["base_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_input_generator",
        ":base_layer",
        ":build_data",
        ":cluster_factory",
        ":early_stop",
        ":hyperparams",
        ":learner",
        ":optimizer",
        ":py_utils",
        ":schedule",
        ":summary_utils",
        ":task_scheduler",
        "//lingvo/core:decoder_lib",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "base_model_test",
    srcs = ["base_model_test.py"],
    deps = [
        ":base_decoder",
        ":base_input_generator",
        ":base_layer",
        ":base_model",
        ":base_model_params",
        ":hyperparams",
        ":layers",
        ":learner",
        ":py_utils",
        ":task_scheduler",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "base_model_params",
    srcs = ["base_model_params.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_input_generator",
        ":base_model",
        ":hyperparams",
    ],
)

py_test(
    name = "base_model_params_test",
    srcs = ["base_model_params_test.py"],
    deps = [
        ":base_model_params",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "build_data",
    srcs = ["build_data.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":hyperparams",
        # Implicit build_data dependency.
    ],
)

py_library(
    name = "builder_layers",
    srcs = ["builder_layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":computation_cost",
        ":py_utils",
        ":recurrent",
        ":summary_utils",
        ":symbolic",
        ":tshape",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "builder_layers_test",
    srcs = ["builder_layers_test.py"],
    deps = [
        ":builder_layers",
        ":layers",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "cluster",
    srcs = ["cluster.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":hyperparams",
        ":py_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "cluster_factory",
    srcs = ["cluster_factory.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo/core:cluster",
    ],
)

py_test(
    name = "cluster_test",
    size = "small",
    srcs = ["cluster_test.py"],
    deps = [
        ":cluster_factory",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "cudnn_rnn_utils",
    srcs = ["cudnn_rnn_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit tensorflow contrib cudnn_rnn_py dependency.
    ],
)

lingvo_cuda_py2and3_test(
    name = "cudnn_rnn_utils_test",
    srcs = ["cudnn_rnn_utils_test.py"],
    deps = [
        ":cudnn_rnn_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_library(
    name = "datasource",
    srcs = ["datasource.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":hyperparams",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "datasource_test",
    size = "small",
    srcs = ["datasource_test.py"],
    deps = [
        ":datasource",
        ":hyperparams",
        ":test_helper",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "early_stop",
    srcs = ["early_stop.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":hyperparams",
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

py_library(
    name = "hyperparams",
    srcs = ["hyperparams.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "hyperparams_test",
    srcs = ["hyperparams_test.py"],
    deps = [
        ":hyperparams",
        ":test_utils",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_library(
    name = "inference_graph_exporter",
    srcs = ["inference_graph_exporter.py"],
    srcs_version = "PY2AND3",
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
    srcs_version = "PY2AND3",
    deps = [
        ":py_utils",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "input_generator_helper_test",
    srcs = ["input_generator_helper_test.py"],
    deps = [
        ":input_generator_helper",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":bn_layers",
        ":computation_cost",
        ":conv_layers_with_time_padding",
        ":py_utils",
        ":quant_utils",
        ":recurrent",
        ":summary_utils",
        ":symbolic",
        ":tshape",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit sympy dependency.
    ],
)

py_library(
    name = "learner",
    srcs = ["learner.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
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
    deps = [
        ":base_layer",
        ":learner",
        ":optimizer",
        ":py_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "conv_layers_with_time_padding",
    srcs = ["conv_layers_with_time_padding.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":bn_layers",
        ":py_utils",
        ":tshape",
        "//lingvo:compat",
    ],
)

lingvo_cuda_py2and3_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    shard_count = 20,
    deps = [
        ":layers_test_lib",
        # Implicit six dependency.
    ],
)

lingvo_cuda_py2and3_test(
    name = "conv_layers_with_time_padding_test",
    srcs = ["conv_layers_with_time_padding_test.py"],
    deps = [
        ":layers_test_lib",
        # Implicit six dependency.
    ],
)

py_library(
    name = "layers_test_lib",
    testonly = 1,
    srcs = ["layers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":gpipe",
        ":layers",
        ":py_utils",
        ":quant_utils",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "schedule",
    srcs = ["schedule.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":early_stop",
        ":py_utils",
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit six dependency.
    ],
)

py_test(
    name = "schedule_test",
    size = "small",
    timeout = "moderate",
    srcs = ["schedule_test.py"],
    deps = [
        ":cluster_factory",
        ":early_stop",
        ":schedule",
        ":test_utils",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":plot",
        ":py_utils",
        ":scorers",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit sklearn dependency.
    ],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":metrics",
        ":test_utils",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_library(
    name = "multitask_model",
    srcs = ["multitask_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":base_model",
        ":py_utils",
    ],
)

py_test(
    name = "multitask_model_test",
    srcs = ["multitask_model_test.py"],
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
    name = "optimizer",
    srcs = ["optimizer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":py_utils",
        ":summary_utils",
        "//lingvo:compat",
    ],
)

py_test(
    name = "optimizer_test",
    size = "small",
    srcs = ["optimizer_test.py"],
    deps = [
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
    srcs_version = "PY2AND3",
    deps = [
        # Implicit Tkinter dependency.
        "//lingvo:compat",
        # Implicit matplotlib dependency.
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_test(
    name = "plot_test",
    srcs = ["plot_test.py"],
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
    srcs_version = "PY2AND3",
    deps = [
        ":inference_graph_py_pb2",
        ":py_utils",
        # Implicit python proto dependency.
        # Implicit IPython dependency.
        "//lingvo:compat",
        "//lingvo:model_imports_no_params",
        # Implicit six dependency.
        # Implicit tensorflow py proto dependency.
    ],
)

lingvo_py_binary(
    name = "predictor",
    srcs = ["predictor.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":predictor_lib",
        ":py_utils",
        "//lingvo:compat",
        "//lingvo:model_imports_no_params",
        # Implicit six dependency.
        # Implicit tensorflow py proto dependency.
    ],
)

py_test(
    name = "predictor_test",
    size = "medium",
    srcs = ["predictor_test.py"],
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
    deps = [
        ":predictor_lib",
        ":py_utils",
        # Implicit absl.flags dependency.
        # Implicit concurrent.futures dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "py_utils",
    srcs = ["py_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":hyperparams",
        ":retry",
        ":symbolic",
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
    data = [
        "//lingvo/core/testdata:lenet_test_model",
    ],
    deps = [
        ":base_layer",
        ":cluster_factory",
        ":py_utils",
        ":recurrent",
        ":test_helper",
        ":test_utils",
        "//lingvo:compat",
        "//lingvo:model_registry",
        "//lingvo/tasks/image/params:mnist",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "quant_utils",
    srcs = ["quant_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":hyperparams",
        ":py_utils",
        ":summary_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "quant_utils_test",
    size = "small",
    srcs = ["quant_utils_test.py"],
    deps = [
        ":base_layer",
        ":py_utils",
        ":quant_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "layers_with_gpipe",
    srcs = ["layers_with_gpipe.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":gpipe",
        ":layers",
        ":layers_with_attention",
        ":py_utils",
        ":tshape",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "layers_with_gpipe_test",
    size = "large",
    srcs = ["layers_with_gpipe_test.py"],
    shard_count = 15,
    deps = [
        ":layers_with_gpipe",
        ":py_utils",
        ":test_utils",
        # Implicit absl.testing.parameterized dependency.
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "spectrum_augmenter",
    srcs = ["spectrum_augmenter.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":py_utils",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "spectrum_augmenter_test",
    srcs = ["spectrum_augmenter_test.py"],
    shard_count = 1,
    deps = [
        ":spectrum_augmenter",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "generic_input",
    srcs = ["generic_input.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":py_utils",
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit tensorflow python framework_for_generated_wrappers dependency.
    ],
)

py_library(
    name = "gpipe",
    srcs = ["gpipe.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":builder_layers",
        ":py_utils",
        ":recurrent",
        ":tshape",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "gpipe_test",
    srcs = ["gpipe_test.py"],
    deps = [
        ":base_layer",
        ":gpipe",
        ":layers",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_library(
    name = "recurrent",
    srcs = ["recurrent.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cluster_factory",
        ":py_utils",
        ":sendrecv",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "recurrent_test",
    srcs = ["recurrent_test.py"],
    shard_count = 6,
    deps = [
        ":recurrent_test_lib",
    ],
)

py_library(
    name = "recurrent_test_lib",
    testonly = 1,
    srcs = ["recurrent_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":py_utils",
        ":recurrent",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow python framework_for_generated_wrappers dependency.
    ],
)

lingvo_cuda_py2and3_test(
    name = "recurrent_gpu_test",
    srcs = ["recurrent_gpu_test.py"],
    shard_count = 6,
    deps = [
        ":py_utils",
        ":recurrent",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "retry",
    srcs = ["retry.py"],
    srcs_version = "PY2AND3",
    deps = [
        # Implicit retry dependency.
    ],
)

py_test(
    name = "retry_test",
    srcs = ["retry_test.py"],
    deps = [
        ":retry",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "sendrecv",
    srcs = ["sendrecv.py"],
    srcs_version = "PY2AND3",
    deps = [
        # Implicit python proto dependency.
        # Implicit tensorflow py proto dependency.
    ],
)

py_test(
    name = "sendrecv_test",
    srcs = ["sendrecv_test.py"],
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
    srcs_version = "PY2AND3",
    deps = [
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
    srcs_version = "PY2AND3",
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
    deps = [
        ":early_stop",
        ":task_scheduler",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "test_helper",
    srcs = ["test_helper.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":py_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    deps = [
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "trainer_test_utils",
    srcs = ["trainer_test_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_input_generator",
        ":base_layer",
        ":base_model",
        ":py_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "wpm_encoder",
    srcs = ["wpm_encoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit six dependency.
    ],
)

py_test(
    name = "wpm_encoder_test",
    srcs = ["wpm_encoder_test.py"],
    deps = [
        ":test_utils",
        ":wpm_encoder",
        "//lingvo:compat",
    ],
)

py_library(
    name = "tokenizers",
    srcs = ["tokenizers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":py_utils",
        ":wpm_encoder",
        "//lingvo:compat",
        "//lingvo/core/ops",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "tokenizers_test",
    srcs = ["tokenizers_test.py"],
    data = [
        "//lingvo/tasks/mt:wpm_ende",
    ],
    deps = [
        ":test_helper",
        ":test_utils",
        ":tokenizers",
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "rnn_cell",
    srcs = ["rnn_cell.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":cudnn_rnn_utils",
        ":hyperparams",
        ":py_utils",
        ":quant_utils",
        ":summary_utils",
        "//lingvo:compat",
        # Implicit six dependency.
        # Implicit tensorflow contrib cudnn_rnn_py dependency.
    ],
)

lingvo_cuda_py2and3_test(
    name = "rnn_cell_test",
    srcs = ["rnn_cell_test.py"],
    deps = [
        ":cudnn_rnn_utils",
        ":py_utils",
        ":quant_utils",
        ":rnn_cell",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit tensorflow contrib cudnn_rnn_py dependency.
    ],
)

py_library(
    name = "layers_with_attention",
    srcs = ["layers_with_attention.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":attention",
        ":base_layer",
        ":layers",
        ":py_utils",
        "//lingvo:compat",
        # Implicit six dependency.
    ],
)

py_test(
    name = "layers_with_attention_test",
    size = "large",
    srcs = ["layers_with_attention_test.py"],
    deps = [
        ":attention",
        ":layers",
        ":layers_with_attention",
        ":py_utils",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "rnn_layers",
    srcs = ["rnn_layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":attention",
        ":base_layer",
        ":cudnn_rnn_utils",
        ":hyperparams",
        ":layers",
        ":layers_with_attention",
        ":py_utils",
        ":quant_utils",
        ":recurrent",
        ":rnn_cell",
        "//lingvo:compat",
        # Implicit six dependency.
        # Implicit tensorflow contrib cudnn_rnn_py dependency.
    ],
)

lingvo_cuda_py2and3_test(
    name = "rnn_layers_test",
    size = "large",
    srcs = ["rnn_layers_test.py"],
    shard_count = 15,
    deps = [
        ":rnn_layers_test_lib",
        # Implicit six dependency.
    ],
)

py_library(
    name = "rnn_layers_test_lib",
    testonly = 1,
    srcs = ["rnn_layers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":attention",
        ":base_layer",
        ":cluster_factory",
        ":layers_with_attention",
        ":py_utils",
        ":rnn_cell",
        ":rnn_layers",
        ":test_utils",
        "//lingvo:compat",
        # Implicit numpy dependency.
        # Implicit six dependency.
    ],
)

py_library(
    name = "scorers",
    srcs = ["scorers.py"],
    srcs_version = "PY2AND3",
    deps = [
        # Implicit six dependency.
    ],
)

py_test(
    name = "scorers_test",
    srcs = ["scorers_test.py"],
    data = ["//lingvo/core/ops/testdata:wmt"],
    deps = [
        ":scorers",
        ":test_helper",
        ":test_utils",
        "//lingvo:compat",
    ],
)

py_library(
    name = "model_helper",
    srcs = ["model_helper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":rnn_layers",
        "//lingvo:compat",
    ],
)

py_library(
    name = "tshape",
    srcs = ["tshape.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lingvo:compat",
        # Implicit six dependency.
        # Implicit sympy dependency.
    ],
)

py_test(
    name = "tshape_test",
    srcs = ["tshape_test.py"],
    deps = [
        ":test_utils",
        ":tshape",
        "//lingvo:compat",
    ],
)

py_library(
    name = "symbolic",
    srcs = ["symbolic.py"],
    srcs_version = "PY2AND3",
    deps = [
        # Implicit six dependency.
        # Implicit sympy dependency.
    ],
)

py_test(
    name = "symbolic_test",
    srcs = ["symbolic_test.py"],
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
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":py_utils",
        "//lingvo:compat",
    ],
)

lingvo_cuda_py2and3_test(
    name = "insertion_test",
    srcs = ["insertion_test.py"],
    deps = [
        ":insertion",
        ":test_utils",
        # Implicit six dependency.
    ],
)

py_library(
    name = "computation_cost",
    srcs = ["computation_cost.py"],
    deps = [
        ":bn_layers",
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "checkpointer_lib",
    srcs = ["checkpointer.py"],
    deps = [
        ":py_utils",
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "program_lib",
    srcs = ["program.py"],
    deps = [
        ":checkpointer_lib",
        ":hyperparams",
        ":metrics",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow grpc dependency.
    ],
)
