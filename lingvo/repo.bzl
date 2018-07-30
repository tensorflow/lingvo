"""Setup autoconf repo for tensorflow."""

def _find_tf_include_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            "python",
            "-c",
            "import tensorflow as tf; import sys; " +
            "sys.stdout.write(tf.sysconfig.get_include())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate tensorflow installation path.")
    return exec_result.stdout

def _find_tf_lib_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            "python",
            "-c",
            "import tensorflow as tf; import sys; " +
            "sys.stdout.write(tf.sysconfig.get_lib())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate tensorflow installation path.")
    return exec_result.stdout

def _eigen_archive_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(
        tf_include_path + "/external/eigen_archive",
        "eigen_archive",
    )
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["eigen_archive/**/*.h", "eigen_archive/**"]),
    # https://groups.google.com/forum/#!topic/bazel-discuss/HyyuuqTxKok
    includes = ["eigen_archive"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _nsync_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path + "/external", "nsync_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["nsync_includes/nsync/public/*.h"]),
    includes = ["nsync_includes"],
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
    deps = ["@eigen_archive//:includes"],
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
    srcs = ["tensorflow_solib/libtensorflow_framework.so"],
    visibility = ["//visibility:public"],
)
""",
    )

def cc_tf_configure():
    """Autoconf pre-installed tensorflow repo."""
    make_eigen_repo = repository_rule(implementation = _eigen_archive_repo_impl)
    make_eigen_repo(name = "eigen_archive")
    make_nsync_repo = repository_rule(
        implementation = _nsync_includes_repo_impl,
    )
    make_nsync_repo(name = "nsync_includes")
    make_tfinc_repo = repository_rule(
        implementation = _tensorflow_includes_repo_impl,
    )
    make_tfinc_repo(name = "tensorflow_includes")
    make_tflib_repo = repository_rule(
        implementation = _tensorflow_solib_repo_impl,
    )
    make_tflib_repo(name = "tensorflow_solib")

def lingvo_testonly_deps():
    if "com_google_googletest" not in native.existing_rules():
        native.new_http_archive(
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
                "https://github.com/google/googletest/archive/release-1.8.0.tar.gz",
            ],
            sha256 = "58a6f4277ca2bc8565222b3bbd58a177609e9c488e8a72649359ba51450db7d8",
            strip_prefix = "googletest-release-1.8.0",
        )


def lingvo_protoc_deps():
  native.new_http_archive(
      name = "protobuf_protoc",
      build_file_content = """
filegroup(
    name = "protoc_bin",
    srcs = ["bin/protoc"],
    visibility = ["//visibility:public"],
)
""",
      urls = [
          "https://github.com/google/protobuf/releases/download/v3.6.0/protoc-3.6.0-linux-x86_64.zip",
      ],
      sha256 = "84e29b25de6896c6c4b22067fb79472dac13cf54240a7a210ef1cac623f5231d",
  )
