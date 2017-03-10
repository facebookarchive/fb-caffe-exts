/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include <caffe/caffe.hpp>

#include "Test.h"

namespace caffe {
namespace fb {

std::vector<ModelSpec> path_specs = {
    ModelSpec{"bvlc_caffenet/deploy.prototxt",
              "bvlc_caffenet/bvlc_caffenet.caffemodel",
              {1, 3, 227, 227},
              "prob",
              {{5, 0.00015368311, 1e-4}}},
    ModelSpec{"bvlc_googlenet/deploy.prototxt",
              "bvlc_googlenet/bvlc_googlenet.caffemodel",
              {1, 3, 227, 227},
              "prob",
              // Use 1e-3 epsilon to make it work on the GPU side
              {{5, 0.0020543954, 1e-3}}}
};

std::vector<ModelSpec> hdf5_specs = {};

}
}
