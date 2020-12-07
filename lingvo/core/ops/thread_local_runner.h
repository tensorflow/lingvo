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

#ifndef THIRD_PARTY_PY_LINGVO_CORE_OPS_THREAD_LOCAL_RUNNER_H_
#define THIRD_PARTY_PY_LINGVO_CORE_OPS_THREAD_LOCAL_RUNNER_H_

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace lingvo {

typedef std::function<void()> Closure;
typedef std::function<void(Closure)> Runner;

// ThreadLocalRunner::PerThread() is a thread local object which owns a thread
// pool with one thread. That thread is configured to disable as much
// TensorFlow runtime parallelism as we can.
//
// NOTE: Maybe a cpu-local object will work better, and the thread in
// ThreadLocalRunner can be affined to one cpu.
class ThreadLocalRunner {
 public:
  static ThreadLocalRunner& PerThread();

  ThreadLocalRunner();

  Runner* runner() { return &runner_; }

 private:
  thread::ThreadPool pool_;
  Runner runner_;

  class Wrapper : Closure {
   public:
    explicit Wrapper(Closure c) : c_(std::move(c)) {}

    void operator()() const {
      ScopedPerThreadMaxParallelism scope(1);
      c_();
    }

   private:
    Closure c_;
  };
};

}  // namespace lingvo
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_LINGVO_CORE_OPS_THREAD_LOCAL_RUNNER_H_
