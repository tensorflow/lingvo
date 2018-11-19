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

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#include "google/protobuf/descriptor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"

namespace {
void WriteDotProto(const google::protobuf::FileDescriptor* dot_proto,
                   const char* output_dirpath) {
  std::string output_filepath(output_dirpath);
  output_filepath += "/";
  output_filepath += dot_proto->name();
  std::ofstream output_file;
  // Assumes the directory tree is already there.
  output_file.open(output_filepath);
  output_file << dot_proto->DebugString();
  output_file.close();
}

void GenerateProtoDef(const google::protobuf::FileDescriptor* dot_proto,
                      const char* output_dirpath,
                      std::unordered_set<std::string>* printed_files) {
  if (printed_files->find(dot_proto->name()) != printed_files->end()) {
    return;
  }
  printed_files->insert(dot_proto->name());
  WriteDotProto(dot_proto, output_dirpath);
  for (int k = 0; k < dot_proto->dependency_count(); ++k)
    GenerateProtoDef(dot_proto->dependency(k), output_dirpath, printed_files);
}

// Regurgitate the text definitions from binary.
void GenerateProtoDefs(const char* output_dirpath) {
  std::unordered_set<std::string> printed_files;
  GenerateProtoDef(tensorflow::GraphDef::descriptor()->file(), output_dirpath,
                   &printed_files);
  GenerateProtoDef(tensorflow::DataType_descriptor()->file(), output_dirpath,
                   &printed_files);
  GenerateProtoDef(tensorflow::SaverDef::descriptor()->file(), output_dirpath,
                   &printed_files);
}
}  // namespace

int main(const int argc, const char** argv) {
  GenerateProtoDefs(argv[1]);
  return 0;
}
