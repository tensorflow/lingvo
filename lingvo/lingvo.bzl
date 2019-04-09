"""Implements custom rules for Lingvo."""

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@subpar//:subpar.bzl", "par_binary")

def tf_copts():
    # TODO(drpng): autoconf this.
    return ["-D_GLIBCXX_USE_CXX11_ABI=0", "-Wno-sign-compare"] + select({
        "//lingvo:cuda": ["-DGOOGLE_CUDA=1"],
        "//conditions:default": [],
    })

def lingvo_cc_library(name, srcs = [], hdrs = [], deps = []):
    native.cc_library(
        name = name,
        copts = tf_copts(),
        srcs = srcs,
        hdrs = hdrs,
        deps = [
            "@tensorflow_includes//:includes",
        ] + deps,
    )

def lingvo_cc_binary(name, srcs = [], deps = []):
    native.cc_binary(
        name = name,
        copts = tf_copts(),
        srcs = srcs,
        deps = [
            "@tensorflow_includes//:includes",
            "@tensorflow_solib//:framework_lib",
        ] + deps,
    )

def lingvo_cc_test(name, srcs, deps = []):
    native.cc_test(
        name = name,
        copts = tf_copts(),
        srcs = srcs,
        deps = [
            "@tensorflow_includes//:includes",
            "@tensorflow_solib//:framework_lib",
            "@com_google_googletest//:gtest_main",
        ] + deps,
    )

def lingvo_py_binary(*args, **kwargs):
    # Extract all files to a temporary directory is required to allow access to
    # the shared object containing lingvo ops.
    kwargs["zip_safe"] = False
    par_binary(*args, **kwargs)

def custom_kernel_library(name, op_def_lib, srcs, hdrs = [], deps = []):
    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        copts = tf_copts(),
        deps = [
            "@tensorflow_includes//:includes",
        ] + deps + op_def_lib,
        alwayslink = 1,
    )

def gen_op_cclib(name, srcs, deps):
    # TODO(drpng): gpu.
    native.cc_library(
        name = name,
        srcs = srcs,
        deps = [
            "@tensorflow_includes//:includes",
            "@tensorflow_solib//:framework_lib",
        ] + deps,
        alwayslink = 1,
        copts = tf_copts(),
    )

def gen_op_pylib(name, cc_lib_name, srcs, kernel_deps):
    native.cc_binary(
        name = cc_lib_name + ".so",
        deps = [cc_lib_name] + kernel_deps,
        linkshared = 1,
        copts = tf_copts(),
    )

    native.py_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY2AND3",
        data = [cc_lib_name + ".so"],
    )

def lingvo_cuda_py_test(name, tags = [], deps = [], **kwargs):
    native.py_test(
        name = name,
        tags = tags + ["requires-gpu"],
        deps = deps,
        **kwargs
    )

def _proto_gen_cc_src(name, basename):
    native.genrule(
        name = name,
        srcs = [basename + ".proto"],
        outs = [basename + ".pb.cc", basename + ".pb.h"],
        tools = [
            "@protobuf_protoc//:protoc_bin",
            "//lingvo:tf_dot_protos",
        ],
        # TODO(drpng): only unpack if tf_proto dependency is requested.
        cmd = """
          mkdir -p $(@D)/tf_proto.$$$$;
          tar -C $(@D)/tf_proto.$$$$ -xf $(location //lingvo:tf_dot_protos);
          $(location @protobuf_protoc//:protoc_bin) --proto_path=$(@D)/tf_proto.$$$$  --proto_path=. --cpp_out=$(GENDIR) $(<);
          rm -rf $(@D)/tf_proto.$$$$
        """,
    )

def _proto_gen_py_src(name, basename):
    native.genrule(
        name = name,
        srcs = [basename + ".proto"],
        outs = [basename + "_pb2.py"],
        tools = [
            "@protobuf_protoc//:protoc_bin",
            "//lingvo:tf_dot_protos",
        ],
        # TODO(drpng): only unpack if tf_proto dependency is requested.
        cmd = """
          mkdir -p $(@D)/tf_proto.$$$$;
          tar -C $(@D)/tf_proto.$$$$ -xf $(location //lingvo:tf_dot_protos);
          $(location @protobuf_protoc//:protoc_bin) --proto_path=$(@D)/tf_proto.$$$$ --proto_path=. --python_out=$(GENDIR) $(<);
          rm -rf $(@D)/tf_proto.$$$$
        """,
    )

def lingvo_proto_cc(name, src, deps = []):
    # TODO(drpng): only works with proto with no deps within lingvo.
    _unused = [deps]
    basename = src.replace(".proto", "")
    _proto_gen_cc_src(name + "_gencc", basename)
    lingvo_cc_library(
        name = name,
        srcs = [basename + ".pb.cc"],
        hdrs = [basename + ".pb.h"],
    )

def lingvo_proto_py(name, src, deps = []):
    # TODO(drpng): only works with proto with no deps within lingvo.
    _unused = [deps]
    basename = src.replace(".proto", "")
    _proto_gen_py_src(name + "_genpy", basename)
    native.py_library(
        name = name,
        srcs = [basename + "_pb2.py"],
    )
