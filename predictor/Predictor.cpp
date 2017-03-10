/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "Predictor.h"

#include <caffe/net.hpp>
#include <folly/Memory.h>

#include "Optimize.h"
#include "Util.h"

namespace caffe {
namespace fb {

namespace {
template <class C>
bool vectorContains(const C& container, const typename C::value_type& value) {
  return std::find(container.begin(), container.end(), value) !=
         container.end();
}
}

std::unique_ptr<Predictor> Predictor::Predictor::paths(
    const std::string& prototxt_path,
    const std::string& weights_path,
    Optimization optimization,
    const bool flag_disable_blas_threading
  ) {
  auto prototxt = loadNetFromFile(prototxt_path);
  auto weights = loadWeightsFromFile(weights_path);

  // Can't make_unique b/c of private constructor
  return std::unique_ptr<Predictor>(
    new Predictor(*prototxt, *weights, optimization,
                  flag_disable_blas_threading));
}

std::unique_ptr<Predictor> Predictor::Predictor::strings(
    const std::string& text_prototxt,
    const std::string& text_weights,
    Optimization optimization,
    const bool flag_disable_blas_threading) {
  auto prototxt = loadNetFromString(text_prototxt);
  auto weights = loadWeightsFromString(text_weights);

  // Can't make_unique b/c of private constructor
  return std::unique_ptr<Predictor>(
      new Predictor(*prototxt, *weights, optimization,
                    flag_disable_blas_threading));
}

std::unique_ptr<Predictor> Predictor::Predictor::hdf5_paths(
    const std::string& prototxt_path,
    const std::string& hdf5_binary_weights_path,
    Optimization optimization,
    const bool flag_disable_blas_threading) {
  auto prototxt = loadNetFromFile(prototxt_path);
  return std::unique_ptr<Predictor>(new Predictor(
      *prototxt,
      hdf5_binary_weights_path,
      optimization,
      flag_disable_blas_threading));
}

Predictor::Predictor(const caffe::NetParameter& param,
                     const caffe::NetParameter& weights,
                     Optimization optimization,
                     const bool flag_disable_blas_threading)
    : optimization_(optimization) {
  if (flag_disable_blas_threading) {
    disable_blas_threading();
  }
  // Check that we have some layers - empty strings/files, for
  // example, are forgivingly deserialized.
  CHECK_GT(param.layer().size(), 0);
  CHECK_GT(weights.layer().size(), 0);
  param_ = std::make_shared<caffe::NetParameter>(param);
  param_->mutable_state()->set_phase(caffe::TEST);
  net_ = std::make_unique<caffe::Net<float>>(*param_);
  net_->CopyTrainedLayersFrom(weights);
}

Predictor::Predictor(const caffe::NetParameter& param,
                     const std::string& hdf5_binary_weights,
                     Optimization optimization,
                     const bool flag_disable_blas_threading)
    : optimization_(optimization) {
  if (flag_disable_blas_threading) {
    disable_blas_threading();
  }
  // Check that we have some layers - empty strings/files, for
  // example, are forgivingly deserialized.
  CHECK_GT(param.layer().size(), 0);
  param_ = std::make_shared<caffe::NetParameter>(param);
  param_->mutable_state()->set_phase(caffe::TEST);
  net_ = std::make_unique<caffe::Net<float>>(*param_);
  net_->CopyTrainedLayersFromHDF5(hdf5_binary_weights);
}

void Predictor::runForward(
    const std::vector<caffe::Blob<float>*>& input_blobs) {
  if (!predictors_.get()) {
    auto predictor =
        std::make_unique<caffe::Net<float>>(*param_);
    predictor->ShareTrainedLayersWith(net_.get());
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
    CHECK(output_blob) << "Misspecified layer_name";
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
    output_blobs[blob_name] = output_blob.get();
  }
  return output_blobs;
}

}
}
