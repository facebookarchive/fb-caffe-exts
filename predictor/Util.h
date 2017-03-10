/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once

#include <memory>
#include <string>

namespace caffe {

// Forward definition of Caffe class.
class NetParameter;

namespace fb {

std::unique_ptr<caffe::NetParameter> loadNetFromFile(
  const std::string& prototxt_path);

std::unique_ptr<caffe::NetParameter> loadWeightsFromFile(
  const std::string& weights_path);

std::unique_ptr<caffe::NetParameter> loadNetFromString(
  const std::string& text_prototxt);

std::unique_ptr<caffe::NetParameter> loadWeightsFromString(
  const std::string& text_weights);

void disable_blas_threading();

}
}
