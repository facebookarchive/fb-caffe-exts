/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "Test.h"

#include <caffe/caffe.hpp>

namespace caffe {
namespace fb {

void SetCaffeModeForTest() {
#ifdef TEST_GPU
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count > 0) {
    Caffe::set_mode(Caffe::GPU);
    return;
  }
#endif
  Caffe::set_mode(Caffe::CPU);
}
}
}
