// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <Foundation/Foundation.h>

#include "tensorflow_utils.h"
#include "ios_image_load.h"

#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <queue>
#include <sstream>
#include <string>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

namespace {

// Helper class used to load protobufs efficiently.
class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
 public:
  explicit IfstreamInputStream(const std::string& file_name)
      : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
  ~IfstreamInputStream() { ifs_.close(); }

  int Read(void* buffer, int size) {
    if (!ifs_) {
      return -1;
    }
    ifs_.read(static_cast<char*>(buffer), size);
    return ifs_.gcount();
  }

 private:
  std::ifstream ifs_;
};
}  // namespace

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
void GetTopN(const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                                    Eigen::Aligned>& prediction,
             const int num_results, const float threshold,
             std::vector<std::pair<float, int> >* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>,
                      std::vector<std::pair<float, int> >,
                      std::greater<std::pair<float, int> > >
      top_result_pq;

  const int count = prediction.size();
  for (int i = 0; i < count; ++i) {
    const float value = prediction(i);

    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
  ::google::protobuf::io::CopyingInputStreamAdaptor stream(
      new IfstreamInputStream(file_name));
  stream.SetOwnsCopyingStream(true);
  ::google::protobuf::io::CodedInputStream coded_stream(&stream);
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

tensorflow::Status LoadModel(NSString* model_path,
                             std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::SessionOptions options;

  tensorflow::Session* session_pointer = nullptr;
  tensorflow::Status session_status =
      tensorflow::NewSession(options, &session_pointer);
  if (!session_status.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Session: " << session_status;
    return session_status;
  }
  session->reset(session_pointer);

  tensorflow::GraphDef tensorflow_graph;

  const bool read_proto_succeeded =
      PortableReadFileToProto([model_path UTF8String], &tensorflow_graph);
  if (!read_proto_succeeded) {
    LOG(ERROR) << "Failed to load model proto from" << [model_path UTF8String];
    return tensorflow::errors::NotFound([model_path UTF8String]);
  }

  tensorflow::Status create_status = (*session)->Create(tensorflow_graph);
  if (!create_status.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Graph: " << create_status;
    return create_status;
  }

  return tensorflow::Status::OK();
}

tensorflow::Status LoadMemoryMappedModel(
    NSString* network_path,
    std::unique_ptr<tensorflow::Session>* session,
    std::unique_ptr<tensorflow::MemmappedEnv>* memmapped_env) {
  memmapped_env->reset(
      new tensorflow::MemmappedEnv(tensorflow::Env::Default()));
  tensorflow::Status mmap_status =
      (memmapped_env->get())->InitializeFromFile([network_path UTF8String]);
  if (!mmap_status.ok()) {
    LOG(ERROR) << "MMap failed with " << mmap_status.error_message();
    return mmap_status;
  }

  tensorflow::GraphDef tensorflow_graph;
  tensorflow::Status load_graph_status = ReadBinaryProto(
      memmapped_env->get(),
      tensorflow::MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
      &tensorflow_graph);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << "MMap load graph failed with "
               << load_graph_status.error_message();
    return load_graph_status;
  }

  tensorflow::SessionOptions options;
  // Disable optimizations on this graph so that constant folding doesn't
  // increase the memory footprint by creating new constant copies of the weight
  // parameters.
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(::tensorflow::OptimizerOptions::L0);
  options.env = memmapped_env->get();

  tensorflow::Session* session_pointer = nullptr;
  tensorflow::Status session_status =
      tensorflow::NewSession(options, &session_pointer);
  if (!session_status.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Session: " << session_status;
    return session_status;
  }

  tensorflow::Status create_status = session_pointer->Create(tensorflow_graph);
  if (!create_status.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Graph: " << create_status;
    return create_status;
  }

  session->reset(session_pointer);

  return tensorflow::Status::OK();
}

tensorflow::Status LoadLabels(NSString* labels_path,
                              std::vector<std::string>* label_strings) {
  // Read the label list
  std::ifstream t;
  t.open([labels_path UTF8String]);
  std::string line;
  while (t) {
    std::getline(t, line);
    label_strings->push_back(line);
  }
  t.close();
  return tensorflow::Status::OK();
}

tensorflow::Status RunInferenceOnImage(NSString* image, int input_size, float input_mean, float input_std, std::string input_layer, std::string output_layer, std::unique_ptr<tensorflow::Session>* session, std::vector<std::string>* labels, std::vector<Result>* results) {

  int image_width;
  int image_height;
  int image_channels;
  std::vector<tensorflow::uint8> image_data = LoadImageFromBase64(
        image, &image_width, &image_height, &image_channels);
  const int wanted_width = input_size;
  const int wanted_height = input_size;
  const int wanted_channels = 3;

  assert(image_channels >= wanted_channels);
  tensorflow::Tensor image_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({
          1, wanted_height, wanted_width, wanted_channels}));
  auto image_tensor_mapped = image_tensor.tensor<float, 4>();
  tensorflow::uint8* in = image_data.data();
  tensorflow::uint8* in_end = (in + (image_height * image_width * image_channels));
  float* out = image_tensor_mapped.data();
  for (int y = 0; y < wanted_height; ++y) {
    const int in_y = (y * image_height) / wanted_height;
    tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
    float* out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const int in_x = (x * image_width) / wanted_width;
      tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
      float* out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
        out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
      }
    }
  }

  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status = (*session)->Run({{input_layer, image_tensor}},
                                               {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return run_status;
  }

  tensorflow::Tensor* output = &outputs[0];
  const int kNumResults = 5;
  const float kThreshold = 0.1f;
  std::vector<std::pair<float, int> > top_results;
  GetTopN(output->flat<float>(), kNumResults, kThreshold, &top_results);
  for (const auto& result : top_results) {
    struct Result res;
    std::string label = labels->at(result.second % labels->size());
    res.label = [NSString stringWithUTF8String:label.c_str()];
    res.confidence = [NSNumber numberWithFloat:result.first];
    results->push_back(res);
  }

  return run_status;
}
