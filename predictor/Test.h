/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once

#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <vector>

namespace caffe {
namespace fb {

enum InputType {
  PATHS = 0,
  STRINGS = 1,
  HDF5_PATHS = 2,
};

struct ModelSpec {
  std::string prototxt;
  std::string caffemodel;
  std::vector<int> inputDims;
  std::string outputLayer;

  struct Output {
    size_t index;
    double score;
    double epsilon;
  };

  std::vector<Output> outputValues;
};

extern std::vector<ModelSpec> path_specs;
extern std::vector<ModelSpec> hdf5_specs;

void SetCaffeModeForTest();
}
}
