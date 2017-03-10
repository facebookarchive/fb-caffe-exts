/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include <caffe/caffe.hpp>
#include <folly/FileUtil.h>
#include <gtest/gtest.h>
#include <thread>

#include "PooledPredictor.h"
#include "Test.h"

namespace caffe {
namespace fb {

using Config = PooledPredictor::Config;
using Optimization = PooledPredictor::Optimization;
using Param = std::tuple<InputType, ModelSpec, int, Optimization>;

std::unique_ptr<caffe::Blob<float>> createInputBlob(
    const ModelSpec& modelSpec) {
  auto blob = std::make_unique<caffe::Blob<float>>();
  blob->Reshape(modelSpec.inputDims);
  FillerParameter param;
  param.set_min(-1000);
  param.set_max(1000);
  UniformFiller<float> filler(param);
  Caffe::set_random_seed(1701);
  filler.Fill(blob.get());
  return blob;
}

class PooledPredictorTest : public ::testing::TestWithParam<Param> {
 protected:
  class TestCallback : public PooledPredictor::Callback {
   public:
    void onJobEnqueued(ssize_t queueSize, uint64_t enqueueDelayMs) override {
      enqueuedJobs_++;
    }

    void onJobDequeued() override {}

    void onJobProcessed(uint64_t processTimeMs) override {
      processedJobs_++;
    }

    std::atomic<uint32_t> enqueuedJobs_{0};
    std::atomic<uint32_t> processedJobs_{0};
  };

  void SetUp() override {
    inputType_ = std::get<0>(GetParam());
    modelSpec_ = std::get<1>(GetParam());
    numThreads_ = std::get<2>(GetParam());
    optimization_ = std::get<3>(GetParam());
  }

  Config getConfig(bool allowInlineScheduling = false) {
    Caffe::set_random_seed(1701);
    SetCaffeModeForTest();

    Config config;
    config.numThreads_ = numThreads_;
    config.mode_ = Caffe::mode();
    config.optimization_ = optimization_;
    config.allowInlineScheduling_ = allowInlineScheduling;

    if (inputType_ == InputType::PATHS) {
      config.protoWeightPaths_.emplace_back(
          modelSpec_.prototxt, modelSpec_.caffemodel);
    } else if (inputType_ == InputType::STRINGS) {
      config.protoWeightStrings_.resize(1);
      folly::readFile(
          modelSpec_.prototxt.c_str(), config.protoWeightStrings_[0].first);
      folly::readFile(
          modelSpec_.caffemodel.c_str(), config.protoWeightStrings_[0].second);
    } else {
      throw std::runtime_error("Unexpected input type");
    }

    return config;
  }

  InputType inputType_;
  ModelSpec modelSpec_;
  int numThreads_;
  Optimization optimization_;
  TestCallback cob_;
};

TEST_P(PooledPredictorTest, Correctness) {
  auto pp = PooledPredictor::makePredictor(getConfig(), &cob_);

  // Create input/output
  auto input_blob = createInputBlob(modelSpec_);
  PooledPredictor::OutputLayers output;
  output[modelSpec_.outputLayer] = std::make_unique<caffe::Blob<float>>();

  // Run forward pass
  pp->forward({input_blob.get()}, &output).get();

  // Check result
  const auto& output_blob = output[modelSpec_.outputLayer];
  for (const auto& v : modelSpec_.outputValues) {
    EXPECT_NEAR(v.score, output_blob->cpu_data()[v.index], v.epsilon);
  }
  EXPECT_EQ(cob_.enqueuedJobs_, 1);
  EXPECT_EQ(cob_.processedJobs_, 1);

  const std::vector<caffe::Blob<float>*>& input_blobs = {input_blob.get()};
  auto future = pp->forward(input_blobs, &output).then([&] {
    EXPECT_EQ(cob_.enqueuedJobs_, 2);
    EXPECT_EQ(cob_.processedJobs_, 2);
    return pp->forward(input_blobs, &output);
  });
  future.get();
  EXPECT_EQ(cob_.enqueuedJobs_, 3);
  EXPECT_EQ(cob_.processedJobs_, 3);

  // Test that the predictor doesn't blow up when given a nonexisting layer.
  output.clear();
  output["undefined"] = std::make_unique<caffe::Blob<float>>();
  pp->forward({input_blob.get()}, &output).get();
  EXPECT_EQ(cob_.enqueuedJobs_, 4);
  EXPECT_EQ(cob_.processedJobs_, 4);
}

TEST_P(PooledPredictorTest, Threading) {
  auto pp = PooledPredictor::makePredictor(getConfig(), &cob_);

  // Run twice as many threads as the pooled predictor uses
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads_; i++) {
    threads.emplace_back([&] {
        // Create input/output
        auto input_blob = createInputBlob(modelSpec_);
        PooledPredictor::OutputLayers output;
        output[modelSpec_.outputLayer] = std::make_unique<caffe::Blob<float>>();

        for (int j = 0; j < 5; j++) {
          // Run forward pass
          pp->forward({input_blob.get()}, &output).get();

          // Check result
          const auto& output_blob = output[modelSpec_.outputLayer];
          for (const auto& kv : modelSpec_.outputValues) {
            auto actual = output_blob->cpu_data()[kv.index];
            EXPECT_NEAR(kv.score, actual, kv.epsilon);
          }
        }
      });
  }

  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(cob_.enqueuedJobs_, 5 * numThreads_);
  EXPECT_EQ(cob_.processedJobs_, 5 * numThreads_);
}

TEST_P(PooledPredictorTest, InlineScheduling) {
  auto pp = PooledPredictor::makePredictor(getConfig(true), &cob_);

  // Create input/output
  auto input_blob = createInputBlob(modelSpec_);
  PooledPredictor::OutputLayers output;
  output[modelSpec_.outputLayer] = std::make_unique<caffe::Blob<float>>();

  // Run forward pass
  auto future = pp->forward({input_blob.get()}, &output).then([&] {
    EXPECT_EQ(cob_.enqueuedJobs_, 1);
    EXPECT_EQ(cob_.processedJobs_, 1);
    return pp->forward({input_blob.get()}, &output);
  });
  future.get();

  // The second job should have bypassed the queue since it was added
  // from the PooledPredictor thread.
  EXPECT_EQ(cob_.enqueuedJobs_, 1);
  EXPECT_EQ(cob_.processedJobs_, 2);

  pp->forward({input_blob.get()}, &output).get();
  EXPECT_EQ(cob_.enqueuedJobs_, 2);
  EXPECT_EQ(cob_.processedJobs_, 3);
}

// Run against all model specs to test for model correctness
INSTANTIATE_TEST_CASE_P(
    Models,
    PooledPredictorTest,
    ::testing::Combine(
        ::testing::Values(InputType::PATHS, InputType::STRINGS),
        ::testing::ValuesIn(path_specs),
        ::testing::Values(1),
        ::testing::Values(Optimization::NONE)));

// Run against single model spec with varying number of threads
// to test for thread correctness
INSTANTIATE_TEST_CASE_P(
    Threads,
    PooledPredictorTest,
    ::testing::Combine(
        ::testing::Values(InputType::PATHS),
        ::testing::Values(path_specs[0]),
        ::testing::Range(1, 20),
        ::testing::Values(Optimization::NONE)));


// Run all models with and without memory optimization to test
// for correctness in both single-threaded and multi-threaded mode.
INSTANTIATE_TEST_CASE_P(
    Memory,
    PooledPredictorTest,
    ::testing::Combine(
        ::testing::Values(InputType::PATHS),
        ::testing::ValuesIn(path_specs),
        ::testing::Values(1, 10),
        ::testing::Values(Optimization::NONE, Optimization::MEMORY)));

/// Test multi-model PooledPredictor

using MultiNetParam = std::tuple<std::vector<ModelSpec>, int>;

class PooledPredictorMultiNetTest
    : public ::testing::TestWithParam<MultiNetParam> {
 protected:
  void SetUp() override {
    modelSpecs_ = std::get<0>(GetParam());
    numThreads_ = std::get<1>(GetParam());
  }

  Config getConfig() {
    Caffe::set_random_seed(1701);
    SetCaffeModeForTest();

    Config config;
    config.numThreads_ = numThreads_;
    config.mode_ = Caffe::mode();
    config.optimization_ = Optimization::MEMORY;
    for (const auto& spec : modelSpecs_) {
      config.protoWeightPaths_.emplace_back(spec.prototxt, spec.caffemodel);
    }

    return config;
  }

  std::vector<ModelSpec> modelSpecs_;
  int numThreads_;
};

TEST_P(PooledPredictorMultiNetTest, Correctness) {
  auto predictors = PooledPredictor::makePredictors(getConfig());

  // Create input/output blobs per model
  std::vector<std::unique_ptr<caffe::Blob<float>>> inputBlobs(
      modelSpecs_.size());
  std::vector<PooledPredictor::OutputLayers> outputs(modelSpecs_.size());
  for (int i = 0; i < modelSpecs_.size(); i++) {
    const auto& spec = modelSpecs_[i];
    inputBlobs[i] = createInputBlob(spec);
    outputs[i][spec.outputLayer] = std::make_unique<caffe::Blob<float>>();
  }

  // Enqueue one forward pass for each model
  std::vector<folly::Future<folly::Unit>> futures;
  for (int i = 0; i < modelSpecs_.size(); i++) {
    futures.push_back(
        predictors[i]->forward({inputBlobs[i].get()}, &outputs[i]));
  }

  // Wait for all nets to complete
  folly::collectAll(futures).get();

  // Check results for all models
  for (int i = 0; i < modelSpecs_.size(); i++) {
    const auto& spec = modelSpecs_[i];
    const auto& output_blob = outputs[i][spec.outputLayer];
    for (const auto& v : spec.outputValues) {
      EXPECT_NEAR(v.score, output_blob->cpu_data()[v.index], v.epsilon);
    }
  }
}

// Run against a set of model specs with varying number of threads
// for correctness across all models/nets within PooledPredictor.
// Vary threads from 1 (serial execution of all nets) to
// model count (complete parallel execution of all nets).
INSTANTIATE_TEST_CASE_P(
    Threads,
    PooledPredictorMultiNetTest,
    ::testing::Combine(
        ::testing::Values(path_specs),
        ::testing::Range(1, (int)path_specs.size())));
}

}
