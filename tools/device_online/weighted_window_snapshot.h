#pragma once
// Saved actual finite-cut operands. See docs/resident-weighted-retained.md.
#include "weighted_flow_internal.h"
namespace tide::device_online {
struct WeightedWindowSnapshot {
  Graph graph;Model model;Continuation boundary;
  at::Device device=at::kCPU;WeightedFlowLimits limits;
  std::vector<at::Device> node_devices;
  int64_t samples=0,start=0,generation=0,bytes=0,bank_bytes=0;
  at::Tensor stages;AtomBatch external,pending;
  std::shared_ptr<WeightedWindowJournal> journal;
  std::vector<WeightedRawLayout> layouts;
  std::vector<std::vector<std::string>> descriptors;
  std::vector<WeightedRawState> final;
  std::vector<at::Tensor> present,history;
  // Live views account all forward-held storage; retained views have no live
  // program pointers and own every tensor needed by a later reverse.
  std::vector<at::Tensor> held;
  std::vector<const DeviceProgram*> programs;
  static std::shared_ptr<WeightedWindowSnapshot> borrow(WeightedFlow&);
  static std::shared_ptr<WeightedWindowSnapshot> retain(WeightedFlow&,int64_t byte_budget);
  void validate()const;
 private:
  struct Version {at::Tensor value;int64_t version;const void* data;bool trainable;};
  std::vector<Version> versions_;
};
}
