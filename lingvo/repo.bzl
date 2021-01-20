"""Setup autoconf repo for tensorflow."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party:repo.bzl", "third_party_http_archive")

def _find_tf_include_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            "python3",
            "-c",
            "import tensorflow as tf; import sys; " +
            "sys.stdout.write(tf.sysconfig.get_include())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate tensorflow installation path.")
    return exec_result.stdout.splitlines()[-1]

def _find_tf_lib_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            "python3",
            "-c",
            "import tensorflow as tf; import sys; " +
            "sys.stdout.write(tf.sysconfig.get_lib())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate tensorflow installation path.")
    return exec_result.stdout.splitlines()[-1]

def _eigen_archive_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tf_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["tf_includes/Eigen/**/*.h",
                 "tf_includes/Eigen/**",
                 "tf_includes/unsupported/Eigen/**/*.h",
                 "tf_includes/unsupported/Eigen/**"]),
    # https://groups.google.com/forum/#!topic/bazel-discuss/HyyuuqTxKok
    includes = ["tf_includes"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _absl_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(
        tf_include_path + "/absl",
        "absl",
    )
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["absl/**/*.h",
                 "absl/**/*.inc"]),
    includes = ["absl"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _zlib_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(
        tf_include_path + "/external/zlib",
        "zlib",
    )
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["zlib/**/*.h"]),
    includes = ["zlib"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _protobuf_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tf_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["tf_includes/google/protobuf/*.h",
                 "tf_includes/google/protobuf/*.inc",
                 "tf_includes/google/protobuf/**/*.h",
                 "tf_includes/google/protobuf/**/*.inc"]),
    includes = ["tf_includes"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _tensorflow_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tensorflow_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["tensorflow_includes/**/*.h",
                 "tensorflow_includes/third_party/eigen3/**"]),
    includes = ["tensorflow_includes"],
    deps = ["@absl_includes//:includes",
            "@eigen_archive//:includes",
            "@protobuf_archive//:includes",
            "@zlib_includes//:includes",],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _tensorflow_solib_repo_impl(repo_ctx):
    tf_lib_path = _find_tf_lib_path(repo_ctx)
    repo_ctx.symlink(tf_lib_path, "tensorflow_solib")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "framework_lib",
    srcs = ["tensorflow_solib/libtensorflow_framework.so.2"],
    visibility = ["//visibility:public"],
)
""",
    )

def cc_tf_configure():
    """Autoconf pre-installed tensorflow repo."""
    make_eigen_repo = repository_rule(implementation = _eigen_archive_repo_impl)
    make_eigen_repo(name = "eigen_archive")
    make_absl_repo = repository_rule(
        implementation = _absl_includes_repo_impl,
    )
    make_absl_repo(name = "absl_includes")
    make_zlib_repo = repository_rule(
        implementation = _zlib_includes_repo_impl,
    )
    make_zlib_repo(name = "zlib_includes")
    make_protobuf_repo = repository_rule(
        implementation = _protobuf_includes_repo_impl,
    )
    make_protobuf_repo(name = "protobuf_archive")
    make_tfinc_repo = repository_rule(
        implementation = _tensorflow_includes_repo_impl,
    )
    make_tfinc_repo(name = "tensorflow_includes")
    make_tflib_repo = repository_rule(
        implementation = _tensorflow_solib_repo_impl,
    )
    make_tflib_repo(name = "tensorflow_solib")

def lingvo_testonly_deps():
    if not native.existing_rule("com_google_googletest"):
        http_archive(
            name = "com_google_googletest",
            build_file_content = """
cc_library(
    name = "gtest",
    srcs = [
          "googletest/src/gtest-all.cc",
          "googlemock/src/gmock-all.cc",
    ],
    copts = ["-D_GLIBCXX_USE_CXX11_ABI=0"],
    hdrs = glob([
        "**/*.h",
        "googletest/src/*.cc",
        "googlemock/src/*.cc",
    ]),
    includes = [
        "googlemock",
        "googletest",
        "googletest/include",
        "googlemock/include",
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gtest_main",
    copts = ["-D_GLIBCXX_USE_CXX11_ABI=0"],
    srcs = ["googlemock/src/gmock_main.cc"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
    deps = [":gtest"],
)
""",
            urls = [
                "https://github.com/google/googletest/archive/release-1.10.0.tar.gz",
            ],
            sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
            strip_prefix = "googletest-release-1.10.0",
        )

def lingvo_protoc_deps():
    http_archive(
        name = "protobuf_protoc",
        build_file_content = """
filegroup(
    name = "protoc_bin",
    srcs = ["bin/protoc"],
    visibility = ["//visibility:public"],
)
""",
        urls = [
            "https://github.com/protocolbuffers/protobuf/releases/download/v3.9.2/protoc-3.9.2-linux-x86_64.zip",
        ],
        sha256 = "0d9034a3b02bd77edf5ef926fb514819a0007f84252c5e6a6391ddfc4189b904",
    )

def icu():
    third_party_http_archive(
        name = "icu",
        strip_prefix = "icu-release-64-2",
        sha256 = "dfc62618aa4bd3ca14a3df548cd65fe393155edd213e49c39f3a30ccd618fc27",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/unicode-org/icu/archive/release-64-2.zip",
            "https://github.com/unicode-org/icu/archive/release-64-2.zip",
        ],
        build_file = "//third_party/icu:BUILD.bazel",
        system_build_file = "//third_party/icu:BUILD.system",
        patch_file = "//third_party/icu:udata.patch",
    )
