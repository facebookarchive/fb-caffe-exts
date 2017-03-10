/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "PinnedPooledPredictor.h"

using namespace folly;

namespace caffe {
namespace fb {

PinnedPooledPredictor::PinnedPooledPredictor(
    std::shared_ptr<PooledPredictor> predictor,
    uint32_t netId)
    : predictor_(predictor), netId_(netId) {}

Future<Unit> PinnedPooledPredictor::forward(
    const std::vector<caffe::Blob<float>*>& input_blobs,
    OutputLayers* output) {
  return predictor_->forward(input_blobs, output, netId_);
}

Future<Unit> PinnedPooledPredictor::forward(
    std::vector<caffe::Blob<float>*>&& input_blobs,
    OutputLayers* output) {
  return predictor_->forward(std::move(input_blobs), output, netId_);
}

const caffe::Net<float>* PinnedPooledPredictor::canonicalNet() const {
  return predictor_->canonicalNet(netId_);
}

}
}
