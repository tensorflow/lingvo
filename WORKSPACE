"""Workspace file for lingvo."""

load(
    "//lingvo:repo.bzl",
    "cc_tf_configure",
    "lingvo_protoc_deps",
    "lingvo_testonly_deps",
)

git_repository(
    name = "subpar",
    remote = "https://github.com/google/subpar",
    tag = "1.3.0",
)

cc_tf_configure()

lingvo_testonly_deps()

lingvo_protoc_deps()
