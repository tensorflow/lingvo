"""Implements custom rules for Lingvo."""

load("@rules_python//python:proto.bzl", "py_proto_library")

def tf_copts():
    #  "-Wno-sign-compare", "-mavx" removed for compat with aarch64
    return ["-std=c++17"] + select({
        "//lingvo:cuda": ["-DGOOGLE_CUDA=1"],
        "//conditions:default": [],
    })

def lingvo_cc_library(name, srcs = [], hdrs = [], deps = [], testonly = 0):
    native.cc_library(
        name = name,
        copts = tf_copts(),
        srcs = srcs,
        hdrs = hdrs,
        deps = [
            "@tensorflow_includes//:includes",
        ] + deps,
        testonly = testonly,
    )

def lingvo_cc_test_library(name, srcs = [], hdrs = [], deps = []):
    lingvo_cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps + ["@com_google_googletest//:gtest"],
        testonly = 1,
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

def lingvo_cc_test(name, srcs, deps = [], **kwargs):
    native.cc_test(
        name = name,
        copts = tf_copts(),
        srcs = srcs,
        deps = [
            "@tensorflow_includes//:includes",
            "@tensorflow_solib//:framework_lib",
            "@com_google_benchmark//:benchmark",
            "@com_google_googletest//:gtest_main",
        ] + deps,
        **kwargs
    )

# TODO(b/263806511): Determine if this alias breaks any existing OSS use-cases.
lingvo_py_binary = native.py_binary

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

def gen_op_cclib(name, srcs, deps = [], nonportable_deps = []):
    # TODO(drpng): gpu.
    native.cc_library(
        name = name,
        srcs = srcs,
        deps = [
            "@tensorflow_includes//:includes",
            "@tensorflow_solib//:framework_lib",
        ] + deps + nonportable_deps,
        alwayslink = 1,
        copts = tf_copts(),
    )

def gen_op_pylib(name, cc_lib_name, srcs, kernel_deps, py_deps = [], **kwargs):
    native.cc_binary(
        name = cc_lib_name + ".so",
        deps = [cc_lib_name] + kernel_deps,
        linkshared = 1,
        copts = tf_copts(),
        **kwargs
    )

    native.py_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY3",
        data = [cc_lib_name + ".so"],
        deps = py_deps,
        **kwargs
    )

def lingvo_cuda_py_test(name, tags = [], deps = [], **kwargs):
    native.py_test(
        name = name,
        tags = tags + ["requires-gpu"],
        deps = deps,
        **kwargs
    )

WELL_KNOWN_PROTO_LIBS = [
    "@com_google_protobuf//:any_proto",
    "@com_google_protobuf//:api_proto",
    "@com_google_protobuf//:compiler_plugin_proto",
    "@com_google_protobuf//:descriptor_proto",
    "@com_google_protobuf//:duration_proto",
    "@com_google_protobuf//:empty_proto",
    "@com_google_protobuf//:field_mask_proto",
    "@com_google_protobuf//:source_context_proto",
    "@com_google_protobuf//:struct_proto",
    "@com_google_protobuf//:timestamp_proto",
    "@com_google_protobuf//:type_proto",
    "@com_google_protobuf//:wrappers_proto",
]

def lingvo_proto_cc(name, src, deps = []):
    # TODO(drpng): only works with proto with no deps within lingvo.
    _unused = [deps]
    basename = src.replace(".proto", "")
    native.proto_library(
        name = name,
        srcs = [src],
        deps = [
             "//lingvo:tf_protos",
        ] + WELL_KNOWN_PROTO_LIBS,
    )
    native.cc_proto_library(
        name = name + "_cc",
        deps = [":" + name]
    )


def lingvo_proto_py(name, src, deps = []):
    # TODO(drpng): only works with proto with no deps within lingvo.
    _unused = [deps]
    basename = src.replace(".proto", "")
    native.proto_library(
        name = name + "_pyproto",
        srcs = [src],
        deps = [
             "//lingvo:tf_protos",
        ] + WELL_KNOWN_PROTO_LIBS,
    )
    py_proto_library(
        name = name,
        deps = [name + "_pyproto"],
    )

# Placeholders to use until bazel supports pytype_{,strict_}{library,test,binary}.
def pytype_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

def pytype_strict_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

def pytype_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

def py_strict_test(name, **kwargs):
    native.py_test(name = name, **kwargs)

def pytype_strict_test(name, **kwargs):
    native.py_test(name = name, **kwargs)

def lingvo_portable_pytype_library(name, deps = [], nonportable_deps = [], **kwargs):
    pytype_library(
        name = name,
        deps = deps + nonportable_deps,
        **kwargs
    )
