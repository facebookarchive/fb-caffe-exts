/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "Util.h"

#include <caffe/net.hpp>
#include <caffe/util/io.hpp>
#include <caffe/util/upgrade_proto.hpp>
#include <folly/Memory.h>
#include <glog/logging.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <mkl.h>

namespace caffe {
namespace fb {

std::unique_ptr<caffe::NetParameter> loadNetFromFile(
    const std::string& prototxt_path) {
  auto prototxt = std::make_unique<caffe::NetParameter>();
  CHECK(caffe::ReadProtoFromTextFile(prototxt_path.c_str(), prototxt.get()));
  CHECK(caffe::UpgradeNetAsNeeded(prototxt_path, prototxt.get()));
  return prototxt;
}

std::unique_ptr<caffe::NetParameter> loadWeightsFromFile(
    const std::string& weights_path) {
  auto weights = std::make_unique<caffe::NetParameter>();
  CHECK(caffe::ReadProtoFromBinaryFile(weights_path, weights.get()));
  CHECK(caffe::UpgradeNetAsNeeded(weights_path, weights.get()));
  return weights;
}

std::unique_ptr<caffe::NetParameter> loadNetFromString(
    const std::string& text_prototxt) {
  auto prototxt = std::make_unique<caffe::NetParameter>();
  CHECK(google::protobuf::TextFormat::ParseFromString(text_prototxt,
                                                        prototxt.get()));
  CHECK(caffe::UpgradeNetAsNeeded("<memory>", prototxt.get()));
  return prototxt;
}

std::unique_ptr<caffe::NetParameter> loadWeightsFromString(
    const std::string& text_weights) {
  auto weights = std::make_unique<caffe::NetParameter>();
  auto input_stream =
      std::make_unique<google::protobuf::io::ArrayInputStream>(
          text_weights.data(), text_weights.size());
  auto stream = std::make_unique<google::protobuf::io::CodedInputStream>(
      input_stream.get());
  // from caffe/util/io.cpp
  constexpr auto kProtoReadBytesLimit =
      INT_MAX; // Max size of 2 GB minus 1 byte.
  stream->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);
  CHECK(weights->ParseFromCodedStream(stream.get()));
  CHECK(caffe::UpgradeNetAsNeeded("<memory>", weights.get()));
  return weights;
}

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
}
