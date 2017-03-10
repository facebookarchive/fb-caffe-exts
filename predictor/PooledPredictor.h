/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once

#include <caffe/common.hpp>
#include <caffe/util/benchmark.hpp>
#include <folly/MPMCQueue.h>
#include <folly/futures/Promise.h>
#include <mutex>
#include <thread>

#include "Predictor.h"

namespace caffe {
namespace fb {

class BasePooledPredictor {
 public:
  using OutputLayers =
      std::unordered_map<std::string, std::unique_ptr<caffe::Blob<float>>>;

  virtual ~BasePooledPredictor() {}

  virtual folly::Future<folly::Unit> forward(
      const std::vector<caffe::Blob<float>*>& input_blobs,
      OutputLayers* output) = 0;

  virtual folly::Future<folly::Unit> forward(
      std::vector<caffe::Blob<float>*>&& input_blobs,
      OutputLayers* output) = 0;

  virtual const caffe::Net<float>* canonicalNet() const = 0;
};

class PooledPredictor {
 public:
  using Optimization = Predictor::Optimization;
  using OutputLayers = BasePooledPredictor::OutputLayers;

  class Callback {
   public:
    virtual ~Callback() {}

    /**
     * Callback invoked once a feed-forward job is added to the queue.
     * @param queueSize - Estimated size of the queue once the job is enqueue.
     *                    Can be negative - see folly/MPMCQueue.h sizeGuess()
     *                    for details.
     * @param enqueueDelayMs - Number of milliseconds blocked waiting to
     *                         enqueue the job if the queue is full.
     */
    virtual void onJobEnqueued(ssize_t queueSize, uint64_t enqueueDelayMs) = 0;

    /**
     * Callback invoked when a job is picked up from the queue for processing.
     */
    virtual void onJobDequeued() = 0;

    /**
     * Callback invoked after a feed-forward job has been picked up by a
     * thread, processed, and the promise fulfilled.
     * @param processTimeMs - Time elapsed by a single feed-forward job.
     */
    virtual void onJobProcessed(uint64_t processTimeMs) = 0;
  };

  struct Config {
    // Pairs of (prototxt path, weights path)
    std::vector<std::pair<std::string, std::string>> protoWeightPaths_;

    // Pairs of (prototxt string, weights string)
    std::vector<std::pair<std::string, std::string>> protoWeightStrings_;

    caffe::Caffe::Brew mode_{caffe::Caffe::CPU};
    int numThreads_{1};
    bool disableBlasThreading_{true};
    Optimization optimization_{Optimization::NONE};

    // If set, jobs enqueued inline from a PooledPredictor thread, such as
    // those scheduled from the returned future's then() callbacks without
    // a thread-pool executor, will be run immediately without being
    // added to the end of the queue.
    //
    // For requests that serially chain feed-forward jobs, inline scheduling
    // would cut down the total execution time as each subsequent feed-forward
    // job is run immediately without having to wait in the queue.
    bool allowInlineScheduling_{false};
  };

  explicit PooledPredictor(const Config& config, Callback* cob = nullptr);

  ~PooledPredictor();

  /**
   * For each prototxt/weight in the config, creates a Predictor and returns
   * a vector of the Predictors created. All the Predictors share the same
   * underlying PooledPredictor queue and threads.
   */
  static std::vector<std::unique_ptr<BasePooledPredictor>> makePredictors(
      const Config& config,
      Callback* cob = nullptr);

  /**
   * Single-net equivalent of makePredictors(). Helper for common use-cases
   * of PooledPredictor with only one net.
   */
  static std::unique_ptr<BasePooledPredictor> makePredictor(
      const Config& config,
      Callback* cob = nullptr);

  folly::Future<folly::Unit> forward(
      const std::vector<caffe::Blob<float>*>& input_blobs,
      OutputLayers* output,
      uint32_t netId);

  folly::Future<folly::Unit> forward(
      std::vector<caffe::Blob<float>*>&& input_blobs,
      OutputLayers* output,
      uint32_t netId);

  const caffe::Net<float>* canonicalNet(uint32_t netId) const;

  size_t netCount() const {
    return nets_.size();
  }

 private:
  struct Job {
    using Function = std::function<void(caffe::Net<float>*)>;
    using Promise = folly::Promise<folly::Unit>;

   public:
    Job(Function&& f, Promise&& p, int32_t netId)
        : function_(std::move(f)),
          promise_(std::move(p)),
          netId_(netId) {}

    Function function_;
    Promise promise_;
    uint32_t netId_;
  };

  void initNet(
      std::unique_ptr<caffe::NetParameter> param,
      std::unique_ptr<caffe::NetParameter> weights);

  void startPredictorThread();

  void forward(
      const std::vector<caffe::Blob<float>*>& input_blobs,
      OutputLayers* output,
      caffe::Net<float>* predictor);

  folly::Future<folly::Unit> enqueueJob(Job::Function&& fn, uint32_t netId);

  void processJob(std::unique_ptr<Job> job);

  const caffe::Caffe::Brew mode_;
  const int numThreads_;
  Optimization optimization_{Optimization::NONE};

  // In GPU mode
  int gpuDeviceCount_;

  // Variables needed to construct a new net (happens on demand)
  std::vector<std::unique_ptr<caffe::NetParameter>> params_;
  std::vector<std::unique_ptr<caffe::NetParameter>> gpuWeights_;
  std::vector<std::unique_ptr<caffe::Net<float>>> nets_;

  // Default input blob shapes
  std::vector<std::vector<std::vector<int>>> shapes_;

  // One predictor net per model per thread
  folly::ThreadLocal<std::vector<std::unique_ptr<caffe::Net<float>>>>
      predictors_;

  folly::MPMCQueue<std::unique_ptr<Job>> queue_;
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::atomic<int> availableThreads_{0};

  // Helps determine if the current thread is a PooledPredictor thread
  // or not. Used for checking if a job should be scheduled inline.
  folly::ThreadLocal<bool> inPredictorThread_;
  bool allowInlineScheduling_;

  Callback* cob_{nullptr};
};

}
}
