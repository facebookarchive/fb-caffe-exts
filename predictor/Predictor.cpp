/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "Predictor.h"

#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "folly/Memory.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include <mkl.h>

#include "Optimize.h"

namespace caffe { namespace fb {

namespace {

template <class C>
bool vectorContains(const C& container, const typename C::value_type& value) {
  return std::find(container.begin(), container.end(), value) !=
         container.end();
}
}

namespace detail {
void disable_blas_threading() {
  // Disable threading for users of this Predictor.
  // Ideally, we'd be able to just link against either mkl_lp64_gomp
  // or mkl_lp64_seq, but Buck's build system doesn't allow this.
  // Instead, just link to _gomp everywhere (including in tp2, etc),
  // and for users of this library (people who explicitly instantiate
  // Predictor), set mkl_num_threads/omp_num_threads to 1.
  // See t8682905 for details.
  LOG(INFO) << "Setting BLAS (MKL, OMP) threads to 1";
  mkl_set_num_threads(1);
}
}

std::unique_ptr<Predictor> Predictor::Predictor::paths(
    const std::string& prototxt_path,
    const std::string& weights_path,
    Optimization optimization) {
  auto prototxt = folly::make_unique<caffe::NetParameter>();
  CHECK(caffe::ReadProtoFromTextFile(prototxt_path.c_str(), prototxt.get()));
  CHECK(caffe::UpgradeNetAsNeeded(prototxt_path, prototxt.get()));

  auto weights = folly::make_unique<caffe::NetParameter>();
  CHECK(caffe::ReadProtoFromBinaryFile(weights_path, weights.get()));
  CHECK(caffe::UpgradeNetAsNeeded(weights_path, weights.get()));
  // Can't make_unique b/c of private constructor
  return std::unique_ptr<Predictor>(
      new Predictor(*prototxt, *weights, optimization));
}

std::unique_ptr<Predictor> Predictor::Predictor::strings(
    const std::string& text_prototxt,
    const std::string& binary_weights,
    Optimization optimization) {
  auto prototxt = folly::make_unique<caffe::NetParameter>();
  CHECK(google::protobuf::TextFormat::ParseFromString(text_prototxt,
                                                        prototxt.get()));
  CHECK(caffe::UpgradeNetAsNeeded("<memory>", prototxt.get()));
  auto weights = folly::make_unique<caffe::NetParameter>();
  auto input_stream =
      folly::make_unique<google::protobuf::io::ArrayInputStream>(
          binary_weights.data(), binary_weights.size());
  auto stream = folly::make_unique<google::protobuf::io::CodedInputStream>(
      input_stream.get());
  // from caffe/util/io.cpp
  constexpr auto kProtoReadBytesLimit =
      INT_MAX; // Max size of 2 GB minus 1 byte.
  stream->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);
  CHECK(weights->ParseFromCodedStream(stream.get()));
  CHECK(caffe::UpgradeNetAsNeeded("<memory>", weights.get()));
  // Can't make_unique b/c of private constructor
  return std::unique_ptr<Predictor>(
      new Predictor(*prototxt, *weights, optimization));
}

Predictor::Predictor(const caffe::NetParameter& param,
                     const caffe::NetParameter& weights,
                     Optimization optimization)
    : optimization_(optimization) {
  detail::disable_blas_threading();

  // Check that we have some layers - empty strings/files, for
  // example, are forgivingly deserialized.
  CHECK(param.layer().size());
  CHECK(weights.layer().size());
  param_ = std::make_shared<caffe::NetParameter>(param);
  param_->mutable_state()->set_phase(caffe::TEST);
  weights_ = folly::make_unique<caffe::Net<float>>(*param_);
  weights_->CopyTrainedLayersFrom(weights);
}

void Predictor::runForward(
    const std::vector<caffe::Blob<float>*>& input_blobs) {
  if (!predictors_.get()) {
    auto predictor =
        folly::make_unique<caffe::Net<float>>(*param_);
    predictor->ShareTrainedLayersWith(weights_.get());
    if (optimization_ == Optimization::MEMORY) {
      optimizeMemory(predictor.get());
    }
    predictors_.reset(predictor.release());
  }
  auto* predictor = predictors_.get();
  CHECK(predictor);
  CHECK_EQ(input_blobs.size(), predictor->input_blobs().size());
  for (auto i = 0; i < input_blobs.size(); ++i) {
    auto& input_blob = input_blobs[i];
    CHECK(input_blob);
    predictor->input_blobs()[i]->ReshapeLike(*input_blob);
    // mutable_cpu_data b/c the interface demands it, but logically const.
    predictor->input_blobs()[i]->set_cpu_data(input_blob->mutable_cpu_data());
  }
  predictor->Reshape();
  predictor->ForwardPrefilled();
}

void Predictor::forward(
    const std::vector<caffe::Blob<float>*>& input_blobs,
    const std::vector<std::string>& output_layer_names,
    std::vector<caffe::Blob<float>*>* output_blobs) {
  runForward(input_blobs);
  auto* predictor = predictors_.get();
  output_blobs->reserve(output_layer_names.size());
  for (const auto& layer_name: output_layer_names) {
    auto& output_blob = predictor->blob_by_name(layer_name);
    CHECK(output_blob) << "Misspecified layer_name: " << layer_name;
    if (optimization_ == Optimization::MEMORY) {
      CHECK(vectorContains(predictor->output_blobs(), output_blob.get()));
    }
    output_blobs->push_back(output_blob.get());
  }
}

std::vector<caffe::Blob<float>*> Predictor::forward(
    const std::vector<caffe::Blob<float>*>& input_blobs,
    const std::vector<std::string>& output_layer_names) {
  std::vector<caffe::Blob<float>*> output_blobs;
  output_blobs.reserve(input_blobs.size());
  forward(input_blobs, output_layer_names, &output_blobs);
  return output_blobs;
}

std::unordered_map<std::string, caffe::Blob<float>*> Predictor::forward(
    const std::vector<caffe::Blob<float>*>& input_blobs) {
  runForward(input_blobs);
  auto* predictor = predictors_.get();
  auto blob_names = predictor->blob_names();
  std::unordered_map<std::string, caffe::Blob<float>*> output_blobs;
  for (const auto& blob_name: blob_names) {
    auto& output_blob = predictor->blob_by_name(blob_name);
    if (optimization_ == Optimization::MEMORY) {
      CHECK(vectorContains(predictor->output_blobs(), output_blob.get()));
    }
    output_blobs[blob_name] = output_blob.get();
  }
  return output_blobs;
}
}
}
