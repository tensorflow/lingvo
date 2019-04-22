package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//lingvo:lingvo.bzl",
    "lingvo_cuda_py_test",
    "lingvo_proto_cc",
    "lingvo_proto_py",
    "lingvo_py_binary",
)

py_library(
    name = "attention",
    srcs = ["attention.py"],
    deps = [
        ":base_layer",
        ":layers",
        ":py_utils",
        ":quant_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

lingvo_cuda_py_test(
    name = "attention_test",
    srcs = ["attention_test.py"],
    shard_count = 5,
    deps = [
        ":attention",
        ":py_utils",
        ":quant_utils",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "base_encoder",
    srcs = ["base_encoder.py"],
    deps = [":base_layer"],
)

py_library(
    name = "base_decoder",
    srcs = ["base_decoder.py"],
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
    deps = [
        ":base_layer",
        ":py_utils",
        # Implicit tensorflow dependency.
        "//lingvo/core/ops:py_x_ops",
    ],
)

py_test(
    name = "beam_search_helper_test",
    srcs = ["beam_search_helper_test.py"],
    deps = [
        ":beam_search_helper",
        ":py_utils",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "bfloat16_variables",
    srcs = ["bfloat16_variables.py"],
    deps = [
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "bfloat16_variables_test",
    srcs = ["bfloat16_variables_test.py"],
    deps = [
        ":bfloat16_variables",
        ":test_utils",
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "target_sequence_sampler",
    srcs = [
        "target_sequence_sampler.py",
    ],
    deps = [
        ":base_layer",
        ":py_utils",
        ":recurrent",
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "target_sequence_sampler_test",
    srcs = ["target_sequence_sampler_test.py"],
    deps = [
        ":py_utils",
        ":target_sequence_sampler",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "base_input_generator",
    srcs = ["base_input_generator.py"],
    deps = [
        ":base_layer",
        ":input_generator_helper",
        ":py_utils",
        ":tokenizers",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core/ops:py_x_ops",
    ],
)

py_test(
    name = "base_input_generator_test",
    srcs = ["base_input_generator_test.py"],
    deps = [
        ":base_input_generator",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "base_layer",
    srcs = ["base_layer.py"],
    deps = [
        ":cluster_factory",
        ":hyperparams",
        ":py_utils",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "bn_layers",
    srcs = ["bn_layers.py"],
    deps = [
        ":base_layer",
        ":cluster_factory",
        ":hyperparams",
        ":py_utils",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "base_layer_test",
    size = "small",
    srcs = ["base_layer_test.py"],
    deps = [
        ":base_layer",
        ":py_utils",
        ":test_utils",
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "base_model",
    srcs = ["base_model.py"],
    deps = [
        ":base_input_generator",
        ":base_layer",
        ":build_data",
        ":cluster_factory",
        ":early_stop",
        ":hyperparams",
        ":lr_schedule",
        ":optimizer",
        ":py_utils",
        ":summary_utils",
        ":task_scheduler",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "base_model_test",
    srcs = ["base_model_test.py"],
    deps = [
        ":base_decoder",
        ":base_encoder",
        ":base_input_generator",
        ":base_layer",
        ":base_model",
        ":base_model_params",
        ":hyperparams",
        ":layers",
        ":py_utils",
        ":task_scheduler",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "base_model_params",
    srcs = ["base_model_params.py"],
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
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "build_data",
    srcs = ["build_data.py"],
    deps = [
        ":hyperparams",
        # Implicit build_data dependency.
    ],
)

py_library(
    name = "cluster",
    srcs = ["cluster.py"],
    deps = [
        ":hyperparams",
        ":py_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "cluster_factory",
    srcs = ["cluster_factory.py"],
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
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "cudnn_rnn_utils",
    srcs = ["cudnn_rnn_utils.py"],
    deps = [
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow contrib cudnn_rnn_py dependency.
    ],
)

lingvo_cuda_py_test(
    name = "cudnn_rnn_utils_test",
    srcs = ["cudnn_rnn_utils_test.py"],
    deps = [
        ":cudnn_rnn_utils",
        ":test_utils",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "early_stop",
    srcs = ["early_stop.py"],
    deps = [
        ":hyperparams",
        # Implicit tensorflow dependency.
        "//lingvo/core/ops:py_x_ops",
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
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "hyperparams",
    srcs = ["hyperparams.py"],
    deps = [
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "hyperparams_test",
    srcs = ["hyperparams_test.py"],
    deps = [
        ":hyperparams",
        ":test_utils",
        # Implicit tensorflow dependency.
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
    name = "inference_graph_exporter",
    srcs = ["inference_graph_exporter.py"],
    deps = [
        ":base_model",
        ":bfloat16_variables",
        ":inference_graph_py_pb2",
        ":py_utils",
        # Implicit python proto dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
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
        # Implicit tensorflow dependency.
        "//lingvo:model_registry",
    ],
)

py_library(
    name = "input_generator_helper",
    srcs = ["input_generator_helper.py"],
    deps = [
        ":py_utils",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "input_generator_helper_test",
    srcs = ["input_generator_helper_test.py"],
    deps = [
        ":input_generator_helper",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    deps = [
        ":base_layer",
        ":bn_layers",
        ":conv_layers_with_time_padding",
        ":py_utils",
        ":quant_utils",
        ":recurrent",
        ":summary_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "conv_layers_with_time_padding",
    srcs = ["conv_layers_with_time_padding.py"],
    deps = [
        ":base_layer",
        ":bn_layers",
        ":py_utils",
        ":quant_utils",
        ":recurrent",
        ":summary_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

lingvo_cuda_py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    shard_count = 20,
    deps = [":layers_test_main_lib"],
)

lingvo_cuda_py_test(
    name = "conv_layers_with_time_padding_test",
    srcs = ["conv_layers_with_time_padding_test.py"],
    deps = [":layers_test_main_lib"],
)

py_library(
    name = "layers_test_main_lib",
    testonly = True,
    srcs = ["layers_test.py"],
    deps = [
        ":conv_layers_with_time_padding",
        ":gpipe",
        ":layers",
        ":py_utils",
        ":quant_utils",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "lr_schedule",
    srcs = ["lr_schedule.py"],
    deps = [
        ":base_layer",
        ":early_stop",
        ":py_utils",
        # Implicit tensorflow dependency.
        "//lingvo/core/ops:py_x_ops",
    ],
)

py_test(
    name = "lr_schedule_test",
    size = "small",
    timeout = "moderate",
    srcs = ["lr_schedule_test.py"],
    deps = [
        ":cluster_factory",
        ":early_stop",
        ":lr_schedule",
        ":test_utils",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
        ":plot",
        ":py_utils",
        ":scorers",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit sklearn dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":metrics",
        ":test_utils",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "multitask_model",
    srcs = ["multitask_model.py"],
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
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    deps = [
        ":base_layer",
        ":hyperparams",
        ":py_utils",
        ":summary_utils",
        # Implicit tensorflow dependency.
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
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "plot",
    srcs = ["plot.py"],
    deps = [
        # Implicit Tkinter dependency.
        # Implicit matplotlib dependency.
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "plot_test",
    srcs = ["plot_test.py"],
    deps = [
        ":plot",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "predictor_lib",
    srcs = ["predictor.py"],
    deps = [
        ":inference_graph_py_pb2",
        ":py_utils",
        # Implicit python proto dependency.
        # Implicit IPython dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow py proto dependency.
        "//lingvo:model_imports_no_params",
    ],
)

lingvo_py_binary(
    name = "predictor",
    srcs = ["predictor.py"],
    deps = [":predictor_main_lib"],
)

py_library(
    name = "predictor_main_lib",
    srcs = ["predictor.py"],
    deps = [
        ":inference_graph_py_pb2",
        ":py_utils",
        # Implicit python proto dependency.
        # Implicit IPython dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow py proto dependency.
        "//lingvo:model_imports",
    ],
)

py_library(
    name = "py_utils",
    srcs = ["py_utils.py"],
    deps = [
        ":hyperparams",
        ":retry",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow py proto dependency.
        "//lingvo/core/ops:py_x_ops",
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
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        "//lingvo:model_registry",
        "//lingvo/tasks/image/params:mnist",
    ],
)

py_library(
    name = "quant_utils",
    srcs = ["quant_utils.py"],
    deps = [
        ":base_layer",
        ":hyperparams",
        ":py_utils",
        ":summary_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
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
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "layers_with_gpipe",
    srcs = ["layers_with_gpipe.py"],
    deps = [
        ":base_layer",
        ":gpipe",
        ":layers",
        ":layers_with_attention",
        ":py_utils",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "layers_with_gpipe_test",
    size = "large",
    srcs = ["layers_with_gpipe_test.py"],
    shard_count = 8,
    deps = [
        ":layers_with_gpipe",
        ":py_utils",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "gpipe",
    srcs = ["gpipe.py"],
    deps = [
        ":base_layer",
        ":py_utils",
        ":recurrent",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
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
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "recurrent",
    srcs = ["recurrent.py"],
    deps = [
        ":py_utils",
        ":sendrecv",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "recurrent_test",
    srcs = ["recurrent_test.py"],
    shard_count = 6,
    deps = [":recurrent_test_lib"],
)

py_library(
    name = "recurrent_test_lib",
    testonly = True,
    srcs = ["recurrent_test.py"],
    deps = [
        ":base_layer",
        ":py_utils",
        ":recurrent",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow python framework_for_generated_wrappers dependency.
    ],
)

lingvo_cuda_py_test(
    name = "recurrent_gpu_test",
    srcs = ["recurrent_gpu_test.py"],
    shard_count = 6,
    deps = [
        ":py_utils",
        ":recurrent",
        ":test_utils",
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "retry",
    srcs = ["retry.py"],
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
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "sendrecv",
    srcs = ["sendrecv.py"],
    deps = [
        # Implicit python proto dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "sendrecv_test",
    srcs = ["sendrecv_test.py"],
    deps = [
        ":sendrecv",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "summary_utils",
    srcs = ["summary_utils.py"],
    deps = [
        ":cluster_factory",
        ":plot",
        ":py_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "task_scheduler",
    srcs = ["task_scheduler.py"],
    deps = [
        ":base_layer",
        ":early_stop",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "task_scheduler_test",
    srcs = ["task_scheduler_test.py"],
    deps = [
        ":early_stop",
        ":task_scheduler",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "test_helper",
    srcs = ["test_helper.py"],
    deps = [
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        ":py_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    deps = [
        ":test_utils",
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "trainer_test_utils",
    srcs = ["trainer_test_utils.py"],
    deps = [
        ":base_input_generator",
        ":base_layer",
        ":base_model",
        ":optimizer",
        ":py_utils",
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "wpm_encoder",
    srcs = ["wpm_encoder.py"],
    deps = [
        # Implicit tensorflow dependency.
        "//lingvo/core/ops:py_x_ops",
    ],
)

py_test(
    name = "wpm_encoder_test",
    srcs = ["wpm_encoder_test.py"],
    deps = [
        ":test_utils",
        ":wpm_encoder",
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "tokenizers",
    srcs = ["tokenizers.py"],
    deps = [
        ":base_layer",
        ":py_utils",
        ":wpm_encoder",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        "//lingvo/core/ops:py_x_ops",
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
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "rnn_cell",
    srcs = ["rnn_cell.py"],
    deps = [
        ":base_layer",
        ":cudnn_rnn_utils",
        ":hyperparams",
        ":py_utils",
        ":quant_utils",
        ":summary_utils",
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow contrib cudnn_rnn_py dependency.
    ],
)

lingvo_cuda_py_test(
    name = "rnn_cell_test",
    srcs = ["rnn_cell_test.py"],
    deps = [
        ":cudnn_rnn_utils",
        ":py_utils",
        ":quant_utils",
        ":rnn_cell",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow contrib cudnn_rnn_py dependency.
    ],
)

py_library(
    name = "layers_with_attention",
    srcs = ["layers_with_attention.py"],
    deps = [
        ":attention",
        ":base_layer",
        ":layers",
        ":py_utils",
        # Implicit tensorflow dependency.
    ],
)

py_test(
    name = "layers_with_attention_test",
    size = "medium",
    srcs = ["layers_with_attention_test.py"],
    deps = [
        ":attention",
        ":layers",
        ":layers_with_attention",
        ":py_utils",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "rnn_layers",
    srcs = ["rnn_layers.py"],
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
        # Implicit six dependency.
        # Implicit tensorflow dependency.
        # Implicit tensorflow contrib cudnn_rnn_py dependency.
    ],
)

lingvo_cuda_py_test(
    name = "rnn_layers_test",
    size = "large",
    srcs = ["rnn_layers_test.py"],
    shard_count = 15,
    deps = [":rnn_layers_test_main_lib"],
)

py_library(
    name = "rnn_layers_test_main_lib",
    testonly = True,
    srcs = ["rnn_layers_test.py"],
    deps = [
        ":attention",
        ":base_layer",
        ":cluster_factory",
        ":layers_with_attention",
        ":py_utils",
        ":rnn_cell",
        ":rnn_layers",
        ":test_utils",
        # Implicit numpy dependency.
        # Implicit six dependency.
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "scorers",
    srcs = ["scorers.py"],
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
        # Implicit tensorflow dependency.
    ],
)

py_library(
    name = "model_helper",
    srcs = ["model_helper.py"],
    deps = [
        ":rnn_layers",
    ],
)
