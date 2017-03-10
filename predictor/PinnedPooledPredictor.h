/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once

#include "PooledPredictor.h"

namespace caffe {
namespace fb {

class PinnedPooledPredictor : public BasePooledPredictor {
 public:
  explicit PinnedPooledPredictor(
      std::shared_ptr<PooledPredictor> predictor,
      uint32_t netId);

  folly::Future<folly::Unit> forward(
      const std::vector<caffe::Blob<float>*>& input_blobs,
      OutputLayers* output) override;

  folly::Future<folly::Unit> forward(
      std::vector<caffe::Blob<float>*>&& input_blobs,
      OutputLayers* output) override;

  const caffe::Net<float>* canonicalNet() const override;

 private:
  std::shared_ptr<PooledPredictor> predictor_;
  uint32_t netId_;
};

}
}
