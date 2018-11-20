package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "best_step_testdata",
    data = [
        "events.out.tfevents.test",
        "history.txt",
        "history_bleu.txt",
    ],
)

filegroup(
    name = "test_vocab",
    data = ["test_vocab.txt"],
)

filegroup(
    name = "test_ngrams",
    data = ["test_ngrams.txt"],
)

filegroup(
    name = "bpe_words_vocab",
    data = ["bpe_words.vocab"],
)

filegroup(
    name = "bpe_codes_vocab",
    data = ["bpe_codes.vocab"],
)

# Data from "Findings of the 2014 Workshop on Statistical Machine Translation"
# http://www.aclweb.org/anthology/W14-3302
filegroup(
    name = "wmt",
    data = glob(["wmt/*"]),
)
