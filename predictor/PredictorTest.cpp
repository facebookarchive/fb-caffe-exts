/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include <caffe/blob.hpp>
#include <caffe/caffe.hpp>
#include <caffe/filler.hpp>
#include <folly/FileUtil.h>
#include <thread>

#include "Optimize.h"
#include "Predictor.h"
#include "Test.h"

namespace caffe {
namespace fb {

using Param = std::tuple<
  InputType,
  Predictor::Optimization,
  ModelSpec
  >;

class PredictorTest : public ::testing::TestWithParam<Param> {
};

TEST_P(PredictorTest, Performance) {
  const auto& inputType = std::get<0>(GetParam());
  const auto& optimization = std::get<1>(GetParam());
  const auto& ms = std::get<2>(GetParam());
  Caffe::set_random_seed(1701);
  SetCaffeModeForTest();

  std::unique_ptr<Predictor> pp;
  if (inputType == InputType::PATHS) {
    pp = Predictor::paths(ms.prototxt, ms.caffemodel, optimization);
  } else if (inputType == InputType::STRINGS) {
    std::string prototxt_str;
    folly::readFile(ms.prototxt.c_str(), prototxt_str);
    std::string caffemodel_str;
    folly::readFile(ms.caffemodel.c_str(), caffemodel_str);
    pp = Predictor::strings(prototxt_str, caffemodel_str, optimization);
  } else if (inputType == InputType::HDF5_PATHS) {
    pp = Predictor::hdf5_paths(ms.prototxt, ms.caffemodel, optimization);
  }
  Blob<float> blob;
  blob.Reshape(ms.inputDims);
  CHECK(pp);
  pp->forward({&blob});
  caffe::Timer timer;
  timer.Start();
  constexpr size_t kIters = 5;
  for (auto i = 0; i < kIters; ++i) {
    pp->forward({&blob});
  }
  LOG(INFO) << "Model: " << ms.prototxt;
  LOG(INFO) << "Optim: " << optimization;
  LOG(INFO) << "Forward took: " << timer.MicroSeconds() / 1000 / kIters << "ms";
}

TEST_P(PredictorTest, ConsistentAcrossThreads) {
  const auto& inputType = std::get<0>(GetParam());
  const auto& optimization = std::get<1>(GetParam());
  const auto& ms = std::get<2>(GetParam());
  Caffe::set_random_seed(1701);
  SetCaffeModeForTest();

  std::unique_ptr<Predictor> pp;
  if (inputType == InputType::PATHS) {
    pp = Predictor::paths(ms.prototxt, ms.caffemodel, optimization);
  } else if (inputType == InputType::STRINGS) {
    std::string prototxt_str;
    folly::readFile(ms.prototxt.c_str(), prototxt_str);
    std::string caffemodel_str;
    folly::readFile(ms.caffemodel.c_str(), caffemodel_str);
    pp = Predictor::strings(prototxt_str, caffemodel_str, optimization);
  } else if (inputType == InputType::HDF5_PATHS) {
    pp = Predictor::hdf5_paths(ms.prototxt, ms.caffemodel, optimization);
  }
  CHECK(pp);
  auto& p = *pp;
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
  for (const auto& kv: ms.outputValues) {
    EXPECT_NEAR(
      kv.score,
      output_blobs[0]->cpu_data()[kv.index],
      kv.epsilon);
  }

  auto output_blobs2 = p.forward({&blob});
  for (const auto& kv : ms.outputValues) {
    EXPECT_NEAR(
        kv.score,
        output_blobs2[ms.outputLayer]->cpu_data()[kv.index],
        kv.epsilon);
  }

  // True across threads as well.
  std::vector<std::thread> ts;
  for (auto i = 0; i < 3; ++i) {
    ts.emplace_back([&](){
        auto output_blobs = p.forward({&blob}, {ms.outputLayer});
        EXPECT_EQ(1, output_blobs.size());
        for (const auto& kv: ms.outputValues) {
          EXPECT_NEAR(
            kv.score,
            output_blobs[0]->cpu_data()[kv.index],
            kv.epsilon);
        }
      });
  }
  for (auto& t: ts) {
    t.join();
  }
}

INSTANTIATE_TEST_CASE_P(
  PROTO,
  PredictorTest,
  ::testing::Combine(
    ::testing::Values(InputType::PATHS, InputType::STRINGS),
    ::testing::Values(
      Predictor::Optimization::MEMORY,
      Predictor::Optimization::NONE
    ),
    ::testing::ValuesIn(path_specs)));

INSTANTIATE_TEST_CASE_P(
  HDF5,
  PredictorTest,
  ::testing::Combine(
    ::testing::Values(InputType::HDF5_PATHS),
    ::testing::Values(Predictor::Optimization::NONE),
    ::testing::ValuesIn(hdf5_specs)));

}
}
