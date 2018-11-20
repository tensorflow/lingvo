package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

# First 100 training sentences from the 1 Billion Word Language Model Benchmark.
# http://www.statmt.org/lm-benchmark
filegroup(
    name = "lm1b_100",
    data = ["lm1b_100.txt"],
)

# Small word-level vocab file (taken from a list of stopwords).
filegroup(
    name = "small_word_vocab",
    data = ["small_word_vocab.txt"],
)
