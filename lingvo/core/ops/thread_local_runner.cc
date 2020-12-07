/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "lingvo/core/ops/thread_local_runner.h"

namespace tensorflow {
namespace lingvo {

typedef std::function<void()> Closure;
typedef std::function<void(Closure)> Runner;

ThreadLocalRunner& ThreadLocalRunner::PerThread() {
  thread_local ThreadLocalRunner tl_runner;
  return tl_runner;
}

ThreadLocalRunner::ThreadLocalRunner() : pool_(Env::Default(), "single", 1) {
  runner_ = [this](Closure c) { pool_.Schedule(Wrapper(c)); };
}

}  // namespace lingvo
}  // namespace tensorflow
