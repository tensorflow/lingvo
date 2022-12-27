"""Workspace file for lingvo."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//lingvo:repo.bzl", "cc_tf_configure", "icu", "lingvo_protoc_deps", "lingvo_testonly_deps")

http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-2.10.0",
    sha256 = "d79a95ede8305f14a10dd0409a1e5a228849039c19ccfb90dfe8367295fd04e0",
    urls = ["https://github.com/tensorflow/tensorflow/archive/v2.10.0.zip"],
)

# This import (along with the org_tensorflow archive) is necessary to provide the devtoolset-9 toolchain
load("@org_tensorflow//tensorflow/tools/toolchains/remote_config:configs.bzl", "initialize_rbe_configs")  # buildifier: disable=load-on-top

initialize_rbe_configs()

cc_tf_configure()

lingvo_testonly_deps()

lingvo_protoc_deps()

icu()
