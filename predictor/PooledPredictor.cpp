/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "PooledPredictor.h"

#include "Optimize.h"
#include "PinnedPooledPredictor.h"
#include "Util.h"

#include <caffe/net.hpp>

DEFINE_bool(log_caffe_predictor, false, "Show layer info for debugging");

namespace caffe {
namespace fb {

PooledPredictor::PooledPredictor(const Config& config, Callback* cob)
    : mode_(config.mode_),
      numThreads_(config.numThreads_),
      optimization_(config.optimization_),
      allowInlineScheduling_(config.allowInlineScheduling_),
      cob_(cob) {
  if (config.disableBlasThreading_) {
    disable_blas_threading();
  }

  CHECK(config.protoWeightPaths_.empty() ^ config.protoWeightStrings_.empty())
      << "Specify exactly one of prototxt/weights paths OR strings";
  if (!config.protoWeightPaths_.empty()) {
    for (const auto& it : config.protoWeightPaths_) {
      auto param = loadNetFromFile(it.first);
      auto weights = loadWeightsFromFile(it.second);
      initNet(std::move(param), std::move(weights));
    }
  } else {
    for (const auto& it : config.protoWeightStrings_) {
      auto param = loadNetFromString(it.first);
      auto weights = loadWeightsFromString(it.second);
      initNet(std::move(param), std::move(weights));
    }
  }
  DCHECK_EQ(params_.size(), nets_.size());
  DCHECK(gpuWeights_.empty() || (gpuWeights_.size() == nets_.size()));
  DCHECK_EQ(shapes_.size(), nets_.size());

  // Initialize queue
  queue_ = folly::MPMCQueue<std::unique_ptr<Job>>(numThreads_);

#ifndef CPU_ONLY
  // Initialize GPU count
  if (mode_ == caffe::Caffe::GPU) {
    CUDA_CHECK(cudaGetDeviceCount(&gpuDeviceCount_));
  }
#endif
}

PooledPredictor::~PooledPredictor() {
  auto n = threads_.size();

  // Send nullptr's to signal the threads they can exit
  for (int i = 0; i < n; i++) {
    queue_.blockingWrite(nullptr);
  }

  // Wait for all threads to exit
  for (int i = 0; i < n; i++) {
    threads_[i].join();
  }
}

std::vector<std::unique_ptr<BasePooledPredictor>>
PooledPredictor::makePredictors(const Config& config, Callback* cob) {
  auto pooledPredictor = std::make_shared<PooledPredictor>(config, cob);
  std::vector<std::unique_ptr<BasePooledPredictor>> predictors;
  for (auto id = 0; id < pooledPredictor->netCount(); id++) {
    predictors.push_back(
        std::make_unique<PinnedPooledPredictor>(pooledPredictor, id));
  }
  return predictors;
}

std::unique_ptr<BasePooledPredictor> PooledPredictor::makePredictor(
    const Config& config,
    Callback* cob) {
  auto predictors = makePredictors(config, cob);
  CHECK_EQ(predictors.size(), 1)
      << "Did you mean to use PooledPredictor::makePredictors?";
  return std::move(predictors[0]);
}

void PooledPredictor::initNet(
    std::unique_ptr<caffe::NetParameter> param,
    std::unique_ptr<caffe::NetParameter> weights) {
  // Check that we have some layers - empty strings/files, for
  // example, are forgivingly deserialized.
  CHECK_GT(param->layer().size(), 0);
  CHECK_GT(weights->layer().size(), 0);
  param->mutable_state()->set_phase(caffe::TEST);

  // Initialize the canonical net
  auto net = std::make_unique<caffe::Net<float>>(*param);
  net->CopyTrainedLayersFrom(*weights);

  // Store default input blob shapes
  shapes_.emplace_back();
  for (const auto& blob : net->input_blobs()) {
    shapes_.back().push_back(blob->shape());
  }

  params_.push_back(std::move(param));
  nets_.push_back(std::move(net));
  if (mode_ == caffe::Caffe::GPU) {
    // Stash the weights to be copied to the GPU nets
    gpuWeights_.push_back(std::move(weights));
  }
}

void PooledPredictor::startPredictorThread() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto threadId = threads_.size();

  // Never exceed capacity
  if (threadId >= numThreads_) {
    return;
  }

  // Create thread and add to list of threads
  threads_.push_back(std::thread([&, threadId] () {
        if (mode_ == caffe::Caffe::CPU) {
          caffe::Caffe::set_mode(caffe::Caffe::CPU);
        } else {
          caffe::Caffe::set_mode(caffe::Caffe::GPU);
          caffe::Caffe::SetDevice(threadId % gpuDeviceCount_);
        }

        // Setup the predictor nets
        for (int i = 0; i < nets_.size(); i++) {
          auto predictor = std::make_unique<caffe::Net<float>>(*params_[i]);
          if (mode_ == caffe::Caffe::CPU) {
            predictor->ShareTrainedLayersWith(nets_[i].get());
          } else {
            // We tried adding weight sharing between nets on the same GPU,
            // which resulted in sporadic NaN outputs of cuDNN (R5) layers.
            // Removing weight sharing immediately solved this problem.
            predictor->CopyTrainedLayersFrom(*gpuWeights_[i]);
          }

          if (optimization_ == Optimization::MEMORY) {
            optimizeMemory(predictor.get());
          }
          predictors_->push_back(std::move(predictor));
        }

        for (;;) {
          availableThreads_++;
          std::unique_ptr<Job> job;
          queue_.blockingRead(job);
          availableThreads_--;
          if (job == nullptr) {
            return;
          }

          if (cob_) {
            cob_->onJobDequeued();
          }

          *inPredictorThread_ = true;
          processJob(std::move(job));
          *inPredictorThread_ = false;
        }
      }));
}

folly::Future<folly::Unit> PooledPredictor::forward(
    const std::vector<caffe::Blob<float>*>& input_blobs,
    OutputLayers* output,
    uint32_t netId) {
  auto fn = [=](caffe::Net<float>* predictor) {
    forward(input_blobs, output, predictor);
  };

  return enqueueJob(std::move(fn), netId);
}

folly::Future<folly::Unit> PooledPredictor::forward(
    std::vector<caffe::Blob<float>*>&& input_blobs,
    OutputLayers* output,
    uint32_t netId) {
  auto fn = [ this, in_blobs = std::move(input_blobs), output ](
      caffe::Net<float> * predictor) {
    forward(in_blobs, output, predictor);
  };

  return enqueueJob(std::move(fn), netId);
}

const caffe::Net<float>* PooledPredictor::canonicalNet(uint32_t netId) const {
  CHECK(netId < nets_.size()) << "Invalid netId: " << netId;
  return nets_[netId].get();
}

void PooledPredictor::forward(
    const std::vector<caffe::Blob<float>*>& input_blobs,
    OutputLayers* output,
    caffe::Net<float>* predictor) {
  CHECK(predictor);
  CHECK_EQ(input_blobs.size(), predictor->input_blobs().size());
  for (auto i = 0; i < input_blobs.size(); ++i) {
    auto& blob = input_blobs[i];
    CHECK(blob);
    predictor->input_blobs()[i]->ReshapeLike(*blob);
    // mutable_cpu_data b/c the interface demands it, but logically const.
    predictor->input_blobs()[i]->set_cpu_data(blob->mutable_cpu_data());
  }
  predictor->Reshape();
  predictor->ForwardPrefilled();

  if (FLAGS_log_caffe_predictor && optimization_ == Optimization::NONE) {
    auto blob_names = predictor->blob_names();
    for (auto& bname : blob_names) {
      auto& blob = predictor->blob_by_name(bname);
      LOG(INFO) << bname << " " << blob->shape_string();
    }
  }

  for (auto& it : *output) {
    auto predictor_blob = predictor->blob_by_name(it.first);
    auto target_blob = it.second.get();

    if (predictor_blob == nullptr) {
      LOG(WARNING) << "Requested output blob not found: " << it.first;
      continue;
    }

    target_blob->ReshapeLike(*predictor_blob);

    if (mode_ == caffe::Caffe::CPU) {
      caffe_copy(
          predictor_blob->count(),
          predictor_blob->cpu_data(),
          target_blob->mutable_cpu_data());
    } else {
      caffe_copy(
          predictor_blob->count(),
          predictor_blob->gpu_data(),
          target_blob->mutable_cpu_data());
    }
  }
}

folly::Future<folly::Unit> PooledPredictor::enqueueJob(
    Job::Function&& fn,
    uint32_t netId) {
  folly::Promise<folly::Unit> promise;
  folly::Future<folly::Unit> future = promise.getFuture();
  auto job = std::make_unique<Job>(std::move(fn), std::move(promise), netId);

  if (allowInlineScheduling_ && *inPredictorThread_) {
    // Note: This prevents tail-call optimization, so if lots of subsequent
    // jobs are being chained, disabling inline scheduling would be safer
    // to avoid running out of stack memory.
    processJob(std::move(job));
    return future;
  }

  // Optimistically try to add a predictor thread if none are available
  if (availableThreads_.load() == 0) {
    startPredictorThread();
  }

  CPUTimer timer;
  timer.Start();
  queue_.blockingWrite(std::move(job));
  timer.Stop();
  if (cob_) {
    cob_->onJobEnqueued(queue_.sizeGuess(), timer.MilliSeconds());
  }

  return future;
}

void PooledPredictor::processJob(std::unique_ptr<Job> job) {
  auto netId = job->netId_;
  auto predictor = predictors_->at(netId).get();

  caffe::CPUTimer timer;
  timer.Start();
  job->function_(predictor);
  timer.Stop();
  if (cob_) {
    cob_->onJobProcessed(timer.MilliSeconds());
  }

  job->promise_.setValue();

  // Restore network to original shape.
  //
  // If the network just processed a large input it will hold
  // on to it until the next job comes along. Without precise
  // memory accounting and input-size-based dispatch, this can
  // cause the process to tip over and OOM. Shrinking the
  // network after every forward pass doesn't eliminate the
  // probability of this happening, it just reduces it.
  //
  // Originally, processing an image in scanning mode would run
  // multiple forward passes and have the last one be the
  // smallest input shape, all against the same predictor
  // instance. This effectively means resizing the network to
  // the smallest input size after processing an image. Since all
  // feed-forwards for a single request are no longer pinned to a
  // single predictor (dispatch happens for every call to the pooled
  // predictor), this implied reshape to a smaller shape no
  // longer happens.
  for (auto i = 0; i < predictor->input_blobs().size(); ++i) {
    predictor->input_blobs()[i]->Reshape(shapes_[netId][i]);
  }
  predictor->Reshape();
}

}
}
