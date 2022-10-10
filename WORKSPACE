"""Workspace file for lingvo."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load(
    "//lingvo:repo.bzl",
    "cc_tf_configure",
    "icu",
    "lingvo_protoc_deps",
    "lingvo_testonly_deps",
)

git_repository(
    name = "subpar",
    remote = "https://github.com/google/subpar",
    commit = "35bb9f0092f71ea56b742a520602da9b3638a24f",
    shallow_since = "1557863961 -0400",
)

cc_tf_configure()

lingvo_testonly_deps()

lingvo_protoc_deps()

icu()
