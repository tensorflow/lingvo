package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "all_params",
    # LINT.IfChange(task_dirs)
    deps = [
        "//lingvo/tasks/asr/params",
        "//lingvo/tasks/car/params",
        "//lingvo/tasks/image/params",
        "//lingvo/tasks/lm/params",
        "//lingvo/tasks/milan/params",
        "//lingvo/tasks/mt/params",
        "//lingvo/tasks/punctuator/params",
    ],
    # LINT.ThenChange(../model_imports.py:task_dirs)
)
