/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once
#include <folly/Range.h>
#include <folly/ThreadLocal.h>
#include <unordered_map>

namespace caffe {
template <typename Dtype>
class Net;
template <typename Dtype>
class Blob;
class NetParameter;
}

namespace caffe {
namespace fb {

class Predictor {
 public:
  enum Optimization {
    NONE,
    MEMORY
  };
  static std::unique_ptr<Predictor> strings(
      const std::string& text_prototxt,
      const std::string& binary_weights,
      Optimization optimization = Optimization::NONE);

  static std::unique_ptr<Predictor> paths(
      const std::string& prototxt_path,
      const std::string& weights_path,
      Optimization optimization = Optimization::NONE);

  std::vector<caffe::Blob<float>*> forward(
      const std::vector<caffe::Blob<float>*>& input_blobs,
      const std::vector<std::string>& output_layer_names);

  void forward(const std::vector<caffe::Blob<float>*>& input_blobs,
               const std::vector<std::string>& output_layer_names,
               std::vector<caffe::Blob<float>*>* output_blobs);

  std::unordered_map<std::string, caffe::Blob<float>*> forward(
      const std::vector<caffe::Blob<float>*>& input_blobs);

  caffe::Net<float>* canonicalNet() const { return weights_.get(); }

 private:
  Predictor(const caffe::NetParameter& params,
            const caffe::NetParameter& weights,
            Optimization optimization = Optimization::NONE);

  void runForward(
      const std::vector<caffe::Blob<float>*>& input_blobs);

  // Shared for forward declaration
  std::shared_ptr<caffe::NetParameter> param_;
  std::shared_ptr<caffe::Net<float>> weights_;
  const Optimization optimization_;

  folly::ThreadLocalPtr<caffe::Net<float>> predictors_;
};
}
}
