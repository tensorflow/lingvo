# Car related tasks.

# Placeholder: load py_library
# Placeholder: load py_test

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])

py_library(
    name = "car_layers",
    srcs = ["car_layers.py"],
    deps = [
        ":car_lib",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
    ],
)

py_library(
    name = "detection_3d_lib",
    srcs = ["detection_3d_lib.py"],
    deps = [
        ":geometry",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/car/ops",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "detection_decoder",
    srcs = ["detection_decoder.py"],
    deps = [
        ":detection_3d_lib",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
    ],
)

py_test(
    name = "detection_decoder_test",
    srcs = ["detection_decoder_test.py"],
    deps = [
        ":detection_decoder",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "detection_3d_metrics",
    srcs = ["detection_3d_metrics.py"],
    deps = [
        ":summary",
        ":transform_util",
        # Implicit PIL dependency.
        "//lingvo:compat",
        "//lingvo/core:metrics",
        "//lingvo/core:plot",
        "//lingvo/core:py_utils",
        # Implicit matplotlib dependency.
        # Implicit numpy dependency.
        # Implicit tensorboard/plugins/mesh:summary dependency.
    ],
)

py_test(
    name = "detection_3d_metrics_test",
    srcs = ["detection_3d_metrics_test.py"],
    deps = [
        ":detection_3d_metrics",
        ":transform_util",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "detection_3d_lib_test",
    srcs = ["detection_3d_lib_test.py"],
    deps = [
        ":detection_3d_lib",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "car_layers_test",
    srcs = ["car_layers_test.py"],
    deps = [
        ":car_layers",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "ap_metric",
    srcs = [
        "ap_metric.py",
    ],
    deps = [
        ":breakdown_metric",
        "//lingvo:compat",
        "//lingvo/core:hyperparams",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "calibration_processing",
    srcs = [
        "calibration_processing.py",
    ],
    deps = [
        "//lingvo:compat",
        "//lingvo/core:plot",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "calibration_processing_test",
    srcs = ["calibration_processing_test.py"],
    deps = [
        ":calibration_processing",
        ":kitti_metadata",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "breakdown_metric",
    srcs = [
        "breakdown_metric.py",
    ],
    deps = [
        ":calibration_processing",
        "//lingvo:compat",
        "//lingvo/core:hyperparams",
        "//lingvo/core:plot",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "breakdown_metric_test",
    srcs = ["breakdown_metric_test.py"],
    deps = [
        ":breakdown_metric",
        ":kitti_ap_metric",
        ":kitti_metadata",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "builder_lib",
    srcs = ["builder_lib.py"],
    deps = [
        ":car_layers",
        "//lingvo:compat",
        "//lingvo/core:activations",
        "//lingvo/core:builder_layers",
        "//lingvo/core:layers",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "builder_lib_test",
    srcs = ["builder_lib_test.py"],
    shard_count = 2,
    deps = [
        ":builder_lib",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "evaluation_metadata",
    srcs = [
        "evaluation_metadata.py",
    ],
    deps = [
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "car_lib",
    srcs = ["car_lib.py"],
    deps = [
        ":geometry",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "geometry",
    srcs = ["geometry.py"],
    deps = [
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "input_extractor",
    srcs = ["input_extractor.py"],
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/car:base_extractor",
    ],
)

py_test(
    name = "input_extractor_test",
    srcs = ["input_extractor_test.py"],
    deps = [
        ":base_extractor",
        ":input_extractor",
        ":input_preprocessors",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:hyperparams",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "base_decoder",
    srcs = ["base_decoder.py"],
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
        "//lingvo/core/ops:record_py_pb2",
    ],
)

py_library(
    name = "base_extractor",
    srcs = ["base_extractor.py"],
    deps = [
        "//lingvo:compat",
        "//lingvo/core:base_input_generator",
        "//lingvo/core:datasource",
        "//lingvo/core:generic_input",
        "//lingvo/core:hyperparams",
        "//lingvo/core:py_utils",
    ],
)

py_library(
    name = "kitti_ap_metric",
    srcs = [
        "kitti_ap_metric.py",
    ],
    deps = [
        ":ap_metric",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/tasks/car/ops",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "kitti_decoder",
    srcs = ["kitti_decoder.py"],
    deps = [
        ":base_decoder",
        ":detection_3d_lib",
        ":detection_3d_metrics",
        ":geometry",
        ":kitti_ap_metric",
        ":kitti_metadata",
        ":transform_util",
        "//lingvo:compat",
        "//lingvo/core:metrics",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "kitti_metadata",
    srcs = [
        "kitti_metadata.py",
    ],
    deps = [":evaluation_metadata"],
)

py_library(
    name = "lr_util",
    srcs = ["lr_util.py"],
    deps = [
        "//lingvo:compat",
        "//lingvo/core:cluster_factory",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
    ],
)

py_test(
    name = "lr_util_test",
    size = "small",
    srcs = ["lr_util_test.py"],
    deps = [
        ":kitti_input_generator",
        ":lr_util",
        ":pillars",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_library(
    name = "pillars",
    srcs = ["pillars.py"],
    deps = [
        ":builder_lib",
        ":car_layers",
        ":detection_3d_lib",
        ":geometry",
        ":point_detector",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:layers",
        "//lingvo/core:optimizer",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "point_detector",
    srcs = ["point_detector.py"],
    deps = [
        ":detection_decoder",
        ":kitti_decoder",
        "//lingvo:compat",
        "//lingvo/core:base_model",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "pointnet",
    srcs = ["pointnet.py"],
    deps = [
        ":builder_lib",
        "//lingvo:compat",
        "//lingvo/core:builder_layers",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "starnet",
    srcs = ["starnet.py"],
    deps = [
        ":builder_lib",
        ":detection_3d_lib",
        ":geometry",
        ":point_detector",
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "summary",
    srcs = ["summary.py"],
    deps = [
        ":transform_util",
        # Implicit PIL dependency.
        "//lingvo:compat",
        "//lingvo/core:plot",
        # Implicit matplotlib dependency.
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "pointnet_test",
    srcs = ["pointnet_test.py"],
    deps = [
        ":pointnet",
        # Implicit absl.testing.parameterized dependency.
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
    ],
)

py_test(
    name = "summary_test",
    srcs = ["summary_test.py"],
    deps = [
        ":summary",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "transform_util",
    srcs = [
        "transform_util.py",
    ],
    deps = [
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "transform_util_test",
    size = "small",
    srcs = [
        "transform_util_test.py",
    ],
    deps = [
        ":transform_util",
        #internal proto upb dep
        "//lingvo:compat",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "car_lib_test",
    srcs = ["car_lib_test.py"],
    deps = [
        ":car_lib",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "geometry_test",
    srcs = ["geometry_test.py"],
    deps = [
        ":geometry",
        # Implicit absl.testing.parameterized dependency.
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "input_preprocessors",
    srcs = ["input_preprocessors.py"],
    deps = [
        ":car_lib",
        ":detection_3d_lib",
        ":geometry",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
        "//lingvo/tasks/car/ops",
        # Implicit numpy dependency.
    ],
)

py_test(
    name = "input_preprocessors_test",
    srcs = ["input_preprocessors_test.py"],
    deps = [
        ":input_preprocessors",
        #internal proto upb dep
        "//lingvo:compat",
        "//lingvo/core:py_utils",
        "//lingvo/core:schedule",
        "//lingvo/core:test_utils",
        # Implicit numpy dependency.
    ],
)

py_library(
    name = "kitti_input_generator",
    srcs = [
        "kitti_input_generator.py",
    ],
    deps = [
        ":geometry",
        ":input_extractor",
        ":input_preprocessors",
        ":kitti_metadata",
        "//lingvo:compat",
        "//lingvo/core:datasource",
        "//lingvo/core:hyperparams",
        "//lingvo/core:py_utils",
        "//lingvo/core/ops",
    ],
)

py_library(
    name = "pillars_anchor_free",
    srcs = ["pillars_anchor_free.py"],
    deps = [
        ":detection_3d_lib",
        ":detection_decoder",
        ":pillars",
        ":point_detector",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
        # Implicit tensorflow_probability dependency.
    ],
)

py_library(
    name = "deep_fusion",
    srcs = ["deep_fusion.py"],
    deps = [
        ":car_lib",
        ":pillars",
        "//lingvo:compat",
        "//lingvo/core:base_layer",
        "//lingvo/core:py_utils",
        # Implicit numpy dependency.
    ],
)
