package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# First 100 sentences of the news-test 2013 from WMT'14 development set.
# http://www.statmt.org/wmt14/dev.tgz
filegroup(
    name = "wmt14_ende_tfexample",
    data = [
        "wmt14_ende_wpm_32k_test.tfrecord",
        "wmt14_ende_wpm_32k_test.vocab",
    ],
)

filegroup(
    name = "mlperf_tfexample",
    data = [
        "translate_ende_mlperf.packed.tfrecord",
        "translate_ende_wmt32k-train-00511-of-00512",
    ],
)

filegroup(
    name = "en_fr",
    data = [
        "en_fr.tfrecord",
    ],
)
