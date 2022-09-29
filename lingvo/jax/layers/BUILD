# Description:
#   Lingvo Jax layers. The public API is defined in __init__.py.

load("//lingvo:lingvo.bzl", "py_strict_test")
load("//lingvo:lingvo.bzl", "pytype_strict_library")
load("//lingvo/jax:build-visibility.bzl", "JAX_VISIBILITY")

package(default_visibility = JAX_VISIBILITY)

licenses(["notice"])

pytype_strict_library(
    name = "layers",
    srcs = ["__init__.py"],
    deps = [
        ":activations",
        ":attentions",
        ":augmentations",
        ":conformers",
        ":convolutions",
        ":embedding_softmax",
        ":linears",
        ":losses",
        ":ngrammer",
        ":normalizations",
        ":objectives",
        ":pipeline",
        ":poolings",
        ":quantizer",
        ":recurrent",
        ":repeats",
        ":resnets",
        ":rnn_cell",
        ":spectrum_augmenter",
        ":stochastics",
        ":transformer_models",
        ":transformers",
        ":vanillanets",
        ":vit",
    ],
)

pytype_strict_library(
    name = "activations",
    srcs = ["activations.py"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "attentions",
    srcs = ["attentions.py"],
    deps = [
        ":embedding_softmax",
        ":stochastics",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "augmentations",
    srcs = ["augmentations.py"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "convolutions",
    srcs = ["convolutions.py"],
    deps = [
        ":activations",
        ":linears",
        ":normalizations",
        ":stochastics",
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "embedding_softmax",
    srcs = ["embedding_softmax.py"],
    deps = [
        ":linears",
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "linears",
    srcs = ["linears.py"],
    deps = [
        ":activations",
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "losses",
    srcs = ["losses.py"],
    deps = [
        # Implicit jax_bitempered_loss dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "ngrammer",
    srcs = ["ngrammer.py"],
    deps = [
        ":embedding_softmax",
        ":normalizations",
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
        # Implicit sympy dependency.
    ],
)

pytype_strict_library(
    name = "normalizations",
    srcs = ["normalizations.py"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:asserts",
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "objectives",
    srcs = [
        "ctc_objectives.py",
        "quantizer_objectives.py",
    ],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:asserts",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "stats",
    srcs = ["stats.py"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

py_strict_test(
    name = "stats_test",
    srcs = ["stats_test.py"],
    deps = [
        ":stats",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        ":recurrent",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "poolings",
    srcs = ["poolings.py"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "quantizer",
    srcs = ["quantizer.py"],
    deps = [
        ":objectives",
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "recurrent",
    srcs = ["recurrent.py"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "repeats",
    srcs = ["repeats.py"],
    deps = [
        ":recurrent",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "resnets",
    srcs = ["resnets.py"],
    deps = [
        ":activations",
        ":convolutions",
        ":poolings",
        ":stochastics",
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "spectrum_augmenter",
    srcs = ["spectrum_augmenter.py"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

py_strict_test(
    name = "spectrum_augmenter_test",
    srcs = ["spectrum_augmenter_test.py"],
    deps = [
        ":spectrum_augmenter",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "rnn_cell",
    srcs = ["rnn_cell.py"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:asserts",
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "vanillanets",
    srcs = ["vanillanets.py"],
    deps = [
        ":convolutions",
        ":poolings",
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "stochastics",
    srcs = ["stochastics.py"],
    deps = [
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "test_layers",
    testonly = True,
    srcs = ["test_layers.py"],
    deps = [
        ":linears",
        ":normalizations",
        ":transformers",
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:base_model",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

py_strict_test(
    name = "conformers_test",
    srcs = ["conformers_test.py"],
    deps = [
        ":conformers",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:cluster_factory",
        "//lingvo/core:conformer_layer",
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

pytype_strict_library(
    name = "conformers",
    srcs = ["conformers.py"],
    deps = [
        ":attentions",
        ":convolutions",
        ":normalizations",
        ":stochastics",
        ":transformers",
        "//lingvo/jax:asserts",
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
    ],
)

pytype_strict_library(
    name = "transformers",
    srcs = ["transformers.py"],
    deps = [
        ":activations",
        ":attentions",
        ":linears",
        ":normalizations",
        ":pipeline",
        ":recurrent",
        ":repeats",
        ":stats",
        ":stochastics",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:gshard_utils",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
        # Implicit numpy dependency.
    ],
)

pytype_strict_library(
    name = "vit",
    srcs = ["vit.py"],
    deps = [
        ":embedding_softmax",
        ":linears",
        ":normalizations",
        ":poolings",
        ":stochastics",
        ":transformers",
        # Implicit einops dependency.
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
    ],
)

pytype_strict_library(
    name = "transformer_models",
    srcs = ["transformer_models.py"],
    deps = [
        ":activations",
        ":attentions",
        ":embedding_softmax",
        ":linears",
        ":normalizations",
        ":pipeline",
        ":recurrent",
        ":repeats",
        ":stats",
        ":stochastics",
        ":transformers",
        # Implicit absl.logging dependency.
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:gshard_utils",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:pytypes",
        # Implicit numpy dependency.
    ],
)

py_strict_test(
    name = "vit_test",
    srcs = ["vit_test.py"],
    deps = [
        ":vit",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "attentions_test",
    srcs = ["attentions_test.py"],
    deps = [
        ":attentions",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:batch_major_attention",
        "//lingvo/jax:base_layer",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "augmentations_test",
    srcs = ["augmentations_test.py"],
    deps = [
        ":augmentations",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
    ],
)

py_strict_test(
    name = "ctc_objectives_test",
    srcs = ["ctc_objectives_test.py"],
    deps = [
        ":objectives",
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        # Implicit jax/experimental/jax2tf dependency.
        "//lingvo/jax:pytypes",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "embedding_softmax_test",
    srcs = ["embedding_softmax_test.py"],
    shard_count = 2,
    deps = [
        ":embedding_softmax",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:layers",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "linears_test",
    srcs = ["linears_test.py"],
    deps = [
        ":linears",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:layers",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "ngrammer_test",
    srcs = ["ngrammer_test.py"],
    deps = [
        ":ngrammer",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:attention_util",
        "//lingvo/jax:base_layer",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "normalizations_test",
    srcs = ["normalizations_test.py"],
    deps = [
        ":normalizations",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:bn_layers",
        "//lingvo/core:layers",
        "//lingvo/jax:base_layer",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "poolings_test",
    srcs = ["poolings_test.py"],
    deps = [
        ":poolings",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:layers",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "quantizer_test",
    srcs = ["quantizer_test.py"],
    deps = [
        ":quantizer",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "quantizer_objectives_test",
    srcs = ["quantizer_objectives_test.py"],
    deps = [
        ":objectives",
        # Implicit absl.testing.absltest dependency.
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
    ],
)

py_strict_test(
    name = "recurrent_test",
    srcs = ["recurrent_test.py"],
    deps = [
        ":recurrent",
        ":stochastics",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "rnn_cell_test",
    srcs = ["rnn_cell_test.py"],
    deps = [
        ":rnn_cell",
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:rnn_cell",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "stochastics_test",
    srcs = ["stochastics_test.py"],
    deps = [
        ":stochastics",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit jax dependency.
        "//lingvo/jax:base_layer",
        "//lingvo/jax:test_utils",
    ],
)

py_strict_test(
    name = "transformers_test",
    size = "large",
    srcs = ["transformers_test.py"],
    shard_count = 20,
    deps = [
        ":attentions",
        ":transformers",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:batch_major_attention",
        "//lingvo/core:layers_with_attention",
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)

py_strict_test(
    name = "transformer_models_test",
    srcs = ["transformer_models_test.py"],
    shard_count = 25,
    deps = [
        ":embedding_softmax",
        ":ngrammer",
        ":transformer_models",
        ":transformers",
        # Implicit absl.logging dependency.
        # Implicit absl.testing.absltest dependency.
        # Implicit absl.testing.parameterized dependency.
        # Implicit jax dependency.
        "//lingvo/core:gshard_builder",
        "//lingvo/jax:base_layer",
        "//lingvo/jax:py_utils",
        "//lingvo/jax:test_utils",
        # Implicit numpy dependency.
        # Implicit tensorflow dependency.
    ],
)
