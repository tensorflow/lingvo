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
#include "lingvo/core/ops/preconditioner_captain.h"

#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {
namespace lingvo {
namespace {

tensorflow::Session* CreateSessionForPreconditioning(
    const PreconditionerCaptainOptions& options) {
  tensorflow::SessionOptions session_options;
  session_options.target = "";
  session_options.config.set_log_device_placement(true);
  auto* session = NewSession(session_options);
  GraphDef gdef;
  CHECK(!options.preconditioner_compute_graphdef.empty());
  gdef.ParseFromString(options.preconditioner_compute_graphdef);
  TF_CHECK_OK(session->Create(gdef));
  return session;
}

const float kEpsilon = 5e-2;

}  // namespace

PreconditionerCaptain::PreconditionerCaptain(
    const PreconditionerCaptainOptions& options)
    : options_(options) {
  workers_ = absl::make_unique<tensorflow::thread::ThreadPool>(
      tensorflow::Env::Default(), "preconditioners-threads",
      options.num_compute_threads);
  // NOTE: For distributing svd across all host machines, you may add many
  // sessions against other CPU hosts.
  sessions_.emplace_back(CreateSessionForPreconditioning(options));
}

PreconditionerCaptain::~PreconditionerCaptain() {}

Tensor PreconditionerCaptain::GetPreconditioner(const std::string& key,
                                                bool* ok) {
  mutex_lock l(mu_);
  if (preconditioners_.find(key) == preconditioners_.end()) {
    *ok = false;
    return {};
  }
  *ok = true;
  return preconditioners_[key];
}

void PreconditionerCaptain::InsertGradientStatistics(const std::string& key,
                                                     Tensor statistics,
                                                     Tensor exponent,
                                                     int global_step,
                                                     bool sync) {
  const int session_to_use = std::hash<std::string>{}(key) % sessions_.size();
  bool should_calculate_preconditioner = true;
  {
    mutex_lock l(mu_);
    if (gradient_statistics_.find(key) != gradient_statistics_.end()) {
      if (gradient_statistics_[key].global_step == global_step) {
        should_calculate_preconditioner = false;
      }
    }
    if (should_calculate_preconditioner) {
      gradient_statistics_[key] = {global_step, statistics};
    }
  }
  if (should_calculate_preconditioner) {
    auto run_preconditioner = [this, key, session_to_use, global_step,
                               statistics, exponent] {
      {
        mutex_lock l(mu_);
        ++active_preconditioners_;
      }

      Status status;
      std::vector<Tensor> outputs;
      std::vector<std::pair<string, Tensor>> inputs;
      inputs.push_back(std::make_pair("input", statistics));
      inputs.push_back(std::make_pair("exponent", exponent));
      do {
        outputs.clear();
        LOG(INFO) << "START: inverse pth root for " << key << " @ "
                  << global_step << " " << statistics.shape().DebugString();
        status = sessions_[session_to_use]->Run(inputs, {"output", "diff"}, {},
                                                &outputs);
        LOG(INFO) << "DONE: Inverse pth root for " << key << " @ "
                  << global_step << " " << statistics.shape().DebugString();
        LOG(INFO) << status.error_message();
      } while (!status.ok());

      // Certain matrices cause SVD to have less precision with its calculation
      // of inverse pth root. We handle that case by ignoring preconditioners
      // for those updates.
      if (outputs[1].scalar<float>()() < kEpsilon) {
        mutex_lock l(mu_);
        preconditioners_[key] = outputs[0];
        LOG(INFO) << "For " << key << " @ " << global_step
                  << " with diff (u-v):" << outputs[1].scalar<float>()();
      } else {
        LOG(INFO) << "Skipping preconditioner update for " << key << " @ "
                  << global_step
                  << " with diff (u-v):" << outputs[1].scalar<float>()();
      }
      {
        mutex_lock l(mu_);
        --active_preconditioners_;
      }
    };
    if (!sync) {
      workers_->Schedule(run_preconditioner);
    } else {
      run_preconditioner();
    }
  }
}

}  // namespace lingvo
}  // namespace tensorflow
