#pragma once
#include "tide/types.h"
#include <memory>

namespace tide::device_online {
struct ContentLimits {
  int64_t queue=1024, arrivals=1024, outputs=1024, trace=4096, stages=4096;
  int64_t workspace_bytes=64*1024*1024;
  bool prefill=true;
};
// Experimental complete forward loop for an explicit existing-module profile:
// sum Aggregate, identity/EMA memory, content linear Read, count/positive
// selection, adopt/clear Next and identity broadcast Full. FP32, no autograd.
// Arbitrary legal positive-delay topology, including feedback. Inputs/initial
// state and exported observables are CPU values; persistent runtime data and
// all decisions between submission and the complete-cut boundary stay on NPU.
// An execution failure poisons this owner; restore a prior cut into a new one.
class ContentFlow {
 public:
  ContentFlow(Graph,Model,const Continuation&,at::Device,ContentLimits={});
  ~ContentFlow();
  ContentFlow(const ContentFlow&)=delete;
  ContentFlow& operator=(const ContentFlow&)=delete;
  Result advance(const std::vector<External>&,Index stop);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace tide::device_online
