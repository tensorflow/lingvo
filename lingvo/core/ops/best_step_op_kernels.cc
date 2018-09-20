/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace lingvo {
using ::tensorflow::io::RecordReader;
namespace {

// Reads a text file containing 'step value' records, and finds the step that
// corresponds to the lowest-value record, within a given tolerance.
class BestStepOp : public OpKernel {
 public:
  explicit BestStepOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hist_file", &hist_file_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tol", &tol_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("minimize", &minimize_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("metric", &metric_));
    CHECK_GE(tol_, 0.0);
  }

  void ExtractValueFromOneTfEvent(OpKernelContext* ctx, const string& filename,
                                  std::map<int, float>* step_value,
                                  const string& metric, bool minimize) {
    const Status status = ctx->env()->FileExists(filename);
    if (status.ok()) {
      ::std::unique_ptr<RandomAccessFile> file;
      OP_REQUIRES_OK(ctx, ctx->env()->NewRandomAccessFile(filename, &file));
      ::std::unique_ptr<RecordReader> reader(new RecordReader(file.get()));

      uint64 offset = 0;
      string raw_proto;
      while (reader->ReadRecord(&offset, &raw_proto).ok()) {
        Event event;
        CHECK(::tensorflow::ParseProtoUnlimited(&event, raw_proto));
        if (event.what_case() != Event::WhatCase::kSummary) {
          continue;
        }
        if (event.has_summary()) {
          for (const auto& value : event.summary().value()) {
            // Look for the tag that matches the metric.
            if (value.tag() == metric) {
              if (minimize) {
                step_value->insert(
                    std::pair<int, float>(event.step(), value.simple_value()));
              } else {
                step_value->insert(
                    std::pair<int, float>(event.step(), -value.simple_value()));
              }
              break;
            }
          }
        }
      }
    } else {
      LOG(WARNING) << "tf events file '" << filename << "' doesn't exist.";
    }
  }

  void ExtractValueFromTfEvents(OpKernelContext* ctx, const string& filename,
                                std::map<int, float>* step_value) {
    std::vector<string> tf_events;
    const Status status = ctx->env()->GetMatchingPaths(filename, &tf_events);
    if (!tf_events.empty()) {
      for (const auto& fname : tf_events) {
        // Loop through all found tf events files.
        ExtractValueFromOneTfEvent(ctx, fname, step_value, metric_, minimize_);
      }
    } else {
      LOG(WARNING) << "Couldn't find tf events files that match pattern: '"
                   << filename;
    }
  }

  void ExtractValueFromTxt(OpKernelContext* ctx, const string& filename,
                           std::map<int, float>* step_value) {
    const Status status = ctx->env()->FileExists(filename);
    if (status.ok()) {
      std::unique_ptr<RandomAccessFile> file;
      OP_REQUIRES_OK(ctx, ctx->env()->NewRandomAccessFile(filename, &file));
      std::unique_ptr<io::RandomAccessInputStream> input_stream(
          new io::RandomAccessInputStream(file.get()));
      io::BufferedInputStream in(input_stream.get(), 4 << 10);
      string line;
      std::vector<float> rec;
      while (true) {
        const Status s = in.ReadLine(&line);
        if (errors::IsOutOfRange(s)) break;
        TF_CHECK_OK(s);
        CHECK(str_util::SplitAndParseAsFloats(line, ' ', &rec));
        CHECK_EQ(rec.size(), 2);
        if (minimize_) {
          step_value->insert(std::pair<int, float>(rec[0], rec[1]));
        } else {  // Negate the value if it's the larger the better.
          step_value->insert(std::pair<int, float>(rec[0], -rec[1]));
        }
      }
    } else {
      LOG(WARNING) << "hist_file '" << &filename << "' doesn't exist.";
    }
  }

  void Compute(OpKernelContext* ctx) override {
    int64 best_step = 0, last_step = 0;
    float best_val = 0.0;
    std::map<int, float> step_value;
    if (hist_file_.find("events.out.tfevents") != std::string::npos) {
      // History file are tf events.
      ExtractValueFromTfEvents(ctx, hist_file_, &step_value);
    } else {  // History file is a txt.
      ExtractValueFromTxt(ctx, hist_file_, &step_value);
    }
    std::map<int, float>::iterator itr;
    for (itr = step_value.begin(); itr != step_value.end(); ++itr) {
      last_step = itr->first;
      const float val = itr->second;
      if (best_step == 0 || val + tol_ < best_val) {
        best_step = last_step;
        best_val = val;
      }
    }

    Tensor* res;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({2}), &res));
    res->vec<int64>()(0) = best_step;
    res->vec<int64>()(1) = last_step;
  }

 private:
  string hist_file_;
  string metric_;
  float tol_ = 0.0;
  bool minimize_ = true;
};

REGISTER_KERNEL_BUILDER(Name("BestStep").Device(DEVICE_CPU), BestStepOp);

}  // namespace
}  // namespace lingvo
}  // namespace tensorflow
