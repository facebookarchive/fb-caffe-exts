/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "Optimize.h"

#include <unordered_map>
#include <boost/make_shared.hpp>

#include <glog/stl_logging.h>

#include <folly/Memory.h>
#include <folly/Format.h>

#include "caffe/net.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {
namespace fb {

namespace {

constexpr int64_t kNotDefined = 0;
constexpr int64_t kNotUsed = -1;
constexpr int64_t kAlwaysLive = 10000;
constexpr int64_t kMinimumCountForSharing = 10000;

struct LiveRange {
  int64_t defined{kNotDefined}, used{kNotUsed};
};

template<typename T>
using Analysis = std::unordered_map<SyncedMemory*, T>;
template<typename T>
using OrderedAnalysis = std::vector<std::pair<SyncedMemory*, T>>;
using SyncedMemoryRange = std::pair<SyncedMemory*, LiveRange>;
using Assignment = std::vector<SyncedMemoryRange>;
using Assignments = std::vector<Assignment>;

template <typename T>
T& findOrInsert(OrderedAnalysis<T>* analysis, SyncedMemory* needle) {
  for (auto& kv : *analysis) {
    if (kv.first == needle) {
      return kv.second;
    }
  }
  analysis->push_back({needle, T()});
  return analysis->back().second;
}

OrderedAnalysis<LiveRange> analyze(const caffe::Net<float>& net) {
  // Build up the liveness analysis by walking the SyncedMemory
  // pointers attached the the blobs in the network.
  const auto& bottoms = net.bottom_vecs();
  const auto& tops = net.top_vecs();
  OrderedAnalysis<LiveRange> analysis;
  for (int64_t i = 0; i < bottoms.size(); ++i) {
    for (const auto* bottom : bottoms[i]) {
      auto& range = findOrInsert(&analysis, bottom->data().get());
      if (range.used == kNotUsed) {
        range.used = i;
        continue;
      }
      range.used = std::max(range.used, i);
    }
  }
  for (int64_t i = 0; i < tops.size(); ++i) {
    for (const auto* top : tops[i]) {
      auto& range = findOrInsert(&analysis, top->data().get());
      if (range.defined == kNotDefined) {
        range.defined = i;
        continue;
      }
      range.defined = std::min(range.defined, i);
    }
  }
  for (const auto* input : net.input_blobs()) {
    findOrInsert(&analysis, input->data().get()).defined = -kAlwaysLive;
    findOrInsert(&analysis, input->data().get()).used = kAlwaysLive;
  }
  return analysis;
}

// Is the candidate range compatible with this assignment?
bool isCompatible(const SyncedMemoryRange& candidate,
                  const Assignment& assignment) {
  if (candidate.second.used == -1) {
    return false;
  }
  if (candidate.first->size() <= kMinimumCountForSharing) {
    return false;
  }
  CHECK_GE(assignment.size(), 1);
  return candidate.second.defined > assignment.back().second.used;
};

Analysis<std::vector<std::string>> blobNames(const caffe::Net<float>& net) {
  Analysis<std::vector<std::string>> names;
  const auto& blobs = net.blobs();
  for (auto i = 0; i < blobs.size(); ++i) {
    names[blobs[i]->data().get()].push_back(net.blob_names().at(i));
  }
  return names;
}

// Compute an assignment of blobs to non-overlapping blobs.
Assignments assign(const Net<float>& net, OrderedAnalysis<LiveRange> analysis) {
  const auto& names = blobNames(net);
  std::stable_sort(analysis.begin(),
                   analysis.end(),
                   [](const SyncedMemoryRange& a, const SyncedMemoryRange& b) {
                     return a.second.used < b.second.used;
                   });
  for (const auto& kv : analysis) {
    LOG(INFO) << names.at(kv.first)
              << folly::format(": {}->{}", kv.second.defined, kv.second.used);
  }

  Assignments assignments;
  for (const auto& candidate : analysis) {
    auto assigned = false;
    for (auto& assignment : assignments) {
      if (isCompatible(candidate, assignment)) {
        assignment.push_back(candidate);
        assigned = true;
        break;
      }
    }
    if (assigned) {
      continue;
    }
    assignments.push_back({candidate});
  }
  return assignments;
}

template <typename T>
void logAssignmentMetrics(const OrderedAnalysis<T>& analysis,
                          const Assignments& assignments) {
  size_t beforeTotalSize = 0;
  for (const auto& kv : analysis) {
    beforeTotalSize += kv.first->size();
  }
  size_t afterTotalSize = 0;
  for (const auto& assignment : assignments) {
    size_t assignmentMaxSize = 0;
    for (const auto& kv : assignment) {
      assignmentMaxSize = std::max(assignmentMaxSize, kv.first->size());
    }
    LOG(INFO) << "Assignment max size: " << assignmentMaxSize;
    afterTotalSize += assignmentMaxSize;
  }
  LOG(INFO)
      << folly::format("Before: {}, After: {}, Compression: {:.2f}%",
                       beforeTotalSize,
                       afterTotalSize,
                       100.0 * (1.0 - afterTotalSize * 1.0 / beforeTotalSize));
}

void applyAssignments(caffe::Net<float>* net, const Assignments& assignments) {
  const auto& names = blobNames(*net);
  Analysis<boost::shared_ptr<Blob<float>>> reusedBlobs;
  for (const auto& assignment : assignments) {
    auto reused = boost::make_shared<Blob<float>>(1, 1, 1, 1);
    // Instantiate so blob->data() is valid.
    reused->cpu_data();
    LOG(INFO) << "Assignment: ";
    for (const auto& kv : assignment) {
      LOG(INFO) << "Blob: " << names.at(kv.first);
      reusedBlobs[kv.first] = reused;
    }
  }

  using BV = std::vector<Blob<float>*>;
  using SBV = std::vector<boost::shared_ptr<Blob<float>>>;
  for (auto& blob : const_cast<BV&>(net->input_blobs())) {
    reusedBlobs.at(blob->data().get())->ReshapeLike(*blob);
    blob = reusedBlobs.at(blob->data().get()).get();
  }
  for (auto& blob : const_cast<BV&>(net->output_blobs())) {
    blob = reusedBlobs.at(blob->data().get()).get();
  }
  for (auto& vec : net->top_vecs()) {
    for (auto& blob : const_cast<BV&>(vec)) {
      blob = reusedBlobs.at(blob->data().get()).get();
    }
  }
  for (auto& vec : net->bottom_vecs()) {
    for (auto& blob : const_cast<BV&>(vec)) {
      blob = reusedBlobs.at(blob->data().get()).get();
    }
  }
  for (auto& blob : const_cast<SBV&>(net->blobs())) {
    auto reusedBlob = reusedBlobs.at(blob->data().get());
    blob = reusedBlob;
  }
}
}

void optimizeMemory(caffe::Net<float>* net) {
  net->Reshape();
  // If the net does sharing (e.g. SplitLayer), run a forward pass to
  // get the sharing setup so that it is indentified when we use the
  // SyncedMemory addresses as identifiers for def/use ranges.
  net->ForwardPrefilled();
  const auto& analysis = analyze(*net);
  const auto& assignments = assign(*net, analysis);
  logAssignmentMetrics(analysis, assignments);
  applyAssignments(net, assignments);
}
}
}
