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
#ifndef LINGVO_CORE_OPS_PRECONDITIONER_CAPTAIN_H_
#define LINGVO_CORE_OPS_PRECONDITIONER_CAPTAIN_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace lingvo {

// PreconditionerCaptain service options.
struct PreconditionerCaptainOptions {
  // Number of threads to run;
  int32 num_compute_threads = 64;
  // Graph that computes the inverse pth root.
  string preconditioner_compute_graphdef;
};

struct StatisticsValue {
  // Global step for entry for the statistics.
  int global_step;
  // Value of the tensor statistics.
  Tensor value;
};

class PreconditionerCaptain {
 public:
  explicit PreconditionerCaptain(const PreconditionerCaptainOptions& options);
  ~PreconditionerCaptain();

  // Disable copy (and move) semantics.
  PreconditionerCaptain(const PreconditionerCaptain&) = delete;
  PreconditionerCaptain& operator=(const PreconditionerCaptain&) = delete;

  // Insert gradient statistics.
  void InsertGradientStatistics(const std::string& key, Tensor statistics,
                                Tensor exponent, int global_step, bool sync);

  // Get preconditioner with status.
  Tensor GetPreconditioner(const std::string& key, bool* ok);

 private:
  // Options for the captain.
  const PreconditionerCaptainOptions options_;

  // Options to throttle preconditioning.
  int32 active_preconditioners_ = 0;

  // Executor used to serve the requests and compute preconditioners.
  std::unique_ptr<tensorflow::thread::ThreadPool> workers_;
  // Mutex protecting the statistics, and preconditioners.
  mutex mu_;
  // A map name to preconditioners
  std::unordered_map<string, StatisticsValue> gradient_statistics_;
  std::unordered_map<string, Tensor> preconditioners_;
  // Internal sessions used to compute preconditioners.
  std::vector<std::unique_ptr<tensorflow::Session>> sessions_;
};

}  // namespace lingvo
}  // namespace tensorflow

#endif  // LINGVO_CORE_OPS_PRECONDITIONER_CAPTAIN_H_
