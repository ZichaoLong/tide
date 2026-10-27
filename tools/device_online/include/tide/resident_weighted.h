#pragma once
// Public complete weighted inference owner. See docs/resident-weighted-flow.md.
#include "tide/types.h"
namespace tide::device_online {
struct WeightedFlowLimits {
  int64_t queue=64,stages=128,cache=64;
  int64_t tensor_bytes=512*1024*1024,operator_bytes=64*1024*1024;
  int32_t timeout_ms=30000;
  bool prefill=true;
};
class WeightedFlow {
 public:
  WeightedFlow(Graph,Model,const Continuation&,at::Device,WeightedFlowLimits={});
  // Empty placement keeps every node on the coordinator. Otherwise exactly one
  // indexed same-backend device per node; region/queue ownership stays global.
  WeightedFlow(Graph,Model,const Continuation&,at::Device,WeightedFlowLimits,std::vector<at::Device> node_devices);
  ~WeightedFlow();
  WeightedFlow(const WeightedFlow&)=delete;
  WeightedFlow& operator=(const WeightedFlow&)=delete;
  // Seals the complete input interval [current cut, until). All recursive
  // readiness, selection and progress within this interval remain on device.
  Result advance(const std::vector<External>&,Index until);
  Continuation checkpoint()const; // Detached exact raw state, ledger and pending.
  int64_t reserved_bytes()const;
  std::vector<at::Device> node_devices()const;
  void close();
 private:
  friend class WeightedWindowAdjoint;
  friend class WeightedTrainingSession;
  friend struct WeightedWindowSnapshot;
  friend struct WeightedFlowPeerTestAccess;
  struct Impl;std::unique_ptr<Impl> impl_;
};
}
