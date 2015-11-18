/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "Predictor.h"
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"

#include <folly/FileUtil.h>
#include <gtest/gtest.h>
#include <thread>

namespace caffe {
namespace fb {

enum class InputTy {
  PATHS = 0,
  STRINGS = 1,
};

struct ModelSpec {
  std::string prototxt;
  std::string caffemodel;
  std::vector<int> inputDims;
  std::string outputLayer;
  std::vector<std::pair<size_t, double>> outputValues;
};

using Param = std::tuple<InputTy,                 // paths or string
                         Predictor::Optimization, // optimization level
                         ModelSpec>;

class PredictorTest : public ::testing::TestWithParam<Param> {};

TEST_P(PredictorTest, ConsistentAcrossThreads) {
  const auto &inputTy = std::get<0>(GetParam());
  const auto &optimization = std::get<1>(GetParam());
  const auto &ms = std::get<2>(GetParam());
  Caffe::set_random_seed(1701);

  std::unique_ptr<Predictor> pp;
  if (inputTy == InputTy::PATHS) {
    pp = Predictor::paths(ms.prototxt, ms.caffemodel, optimization);
  } else if (inputTy == InputTy::STRINGS) {
    std::string prototxt_str;
    folly::readFile(ms.prototxt.c_str(), prototxt_str);
    std::string caffemodel_str;
    folly::readFile(ms.caffemodel.c_str(), caffemodel_str);
    pp = Predictor::strings(prototxt_str, caffemodel_str, optimization);
  }
  CHECK(pp);
  auto &p = *pp;
  FillerParameter param;
  param.set_min(-1000);
  param.set_max(1000);
  UniformFiller<float> filler(param);
  Blob<float> blob;
  blob.Reshape(ms.inputDims);
  filler.Fill(&blob);
  auto output_blobs = p.forward({&blob}, {ms.outputLayer});
  // Test output blobs in-place.
  EXPECT_EQ(1, output_blobs.size());
  output_blobs.clear();
  p.forward({&blob}, {ms.outputLayer}, &output_blobs);
  EXPECT_EQ(1, output_blobs.size());
  for (const auto &kv : ms.outputValues) {
    EXPECT_FLOAT_EQ(kv.second, output_blobs[0]->cpu_data()[kv.first]);
  }

  auto output_blobs2 = p.forward({&blob});
  for (const auto &kv : ms.outputValues) {
    EXPECT_FLOAT_EQ(kv.second,
                    output_blobs2[ms.outputLayer]->cpu_data()[kv.first]);
  }

  // True across threads as well.
  std::vector<std::thread> ts;
  for (auto i = 0; i < 3; ++i) {
    ts.emplace_back([&]() {
      auto output_blobs = p.forward({&blob}, {ms.outputLayer});
      EXPECT_EQ(1, output_blobs.size());
      for (const auto &kv : ms.outputValues) {
        EXPECT_FLOAT_EQ(kv.second, output_blobs[0]->cpu_data()[kv.first]);
      }
    });
  }
  for (auto &t : ts) {
    t.join();
  }
}

INSTANTIATE_TEST_CASE_P(
    P,
    PredictorTest,
    ::testing::Combine(
        ::testing::Values(InputTy::PATHS, InputTy::STRINGS),
        ::testing::Values(Predictor::Optimization::MEMORY,
                          Predictor::Optimization::NONE),
        ::testing::Values())
            ModelSpec{
                "bvlc_caffenet/deploy.prototxt",
                "bvlc_caffenet/bvlc_caffenet.caffemodel",
                {1, 3, 227, 227},
                "prob",
                {{5, 0.00015368311}}},
            ModelSpec{
                "bvlc_googlenet/deploy.prototxt",
                "bvlc_googlenet/bvlc_googlenet.caffemodel",
                {1, 3, 227, 227},
                "prob",
                {{5, 0.0020543954}}})));
}
}
