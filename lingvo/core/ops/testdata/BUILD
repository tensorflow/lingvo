package(default_visibility = ["//lingvo/core/ops:__pkg__"])

licenses(["notice"])  # Apache 2.0

package_group(
    name = "friends",
    packages = ["//lingvo/core/..."],
)

filegroup(
    name = "best_step_testdata",
    data = [
        "events.out.tfevents.test",
        "history.txt",
        "history_bleu.txt",
    ],
    visibility = [":friends"],
)

filegroup(
    name = "test_vocab",
    data = ["test_vocab.txt"],
)

filegroup(
    name = "test_ngrams",
    data = ["test_ngrams.txt"],
)

# Data from "Findings of the 2014 Workshop on Statistical Machine Translation"
# http://www.aclweb.org/anthology/W14-3302
filegroup(
    name = "wmt",
    data = glob(["wmt/*"]),
    visibility = [":friends"],
)
