#pragma once
// Pinned weighted functions and finite cuts: docs/resident-weighted-training.md.
#include "tide/resident_weighted.h"
#include "tide/resident_training.h"
namespace tide::device_online {
struct WeightedTrainingLimits {
  WeightedFlowLimits forward;
  Index windows=8,retained_bytes=256*1024*1024,backward_bytes=1024*1024*1024;
  Index optimizer_bytes=128*1024*1024,total_bytes=2048LL*1024*1024;
};
// Each present row is an independent mathematical root. A ragged field has a
// separate valid mask over [row,slot]; slots outside it are not logical values.
// Raw/payload values use the model dtype; arithmetic fields and all cotangents
// use FP32. Clocks, counts and coordinates remain separate int64 metadata.
struct WeightedTrainingField {Tensor values,present,valid;};
struct WeightedTrainingWindow {
  ResidentToken token;
  Index start=0,stop=0;
  std::map<std::string,WeightedTrainingField> fields;
  std::map<std::string,Tensor> metadata;
};
struct WeightedCotangent {Tensor values,connected;};
struct WeightedCotangents {
  ResidentToken token;
  // Omitted fields mean None. A supplied pair must match the window field's
  // values/present shapes. Connected ragged padding must have zero cotangent.
  std::map<std::string,WeightedCotangent> fields;
};
struct WeightedStateGradient {
  Index node=0;std::string family;
  Tensor integer,present;
  std::map<std::string,WeightedCotangent> fields;
};
struct WeightedGradients {
  ResidentParameterGradient parameters;
  std::vector<WeightedStateGradient> initial;
  std::vector<ResidentBoundaryGradient> boundaries;
  std::map<std::string,Index> statistics;
};
struct WeightedPeerStep {
  ResidentStep step;
  std::vector<ResidentParameterGradient> parameters; // Complete sum on each device.
  Index transferred_bytes=0,reserved_tensor_bytes=0;
};
// Explicit FP32/FP16 storage, complete weighted first-order training owner.
// Loss/task heads are consumer-owned; all calls require no-grad. Tokens name
// actual retained windows under one parameter generation, never a CPU replay.
class WeightedTrainingSession {
 public:
  WeightedTrainingSession(Graph,Model,const Continuation&,at::Device,
    ResidentOptimizerKind kind=ResidentOptimizerKind::sgd,std::vector<OptimizerGroup> groups={},WeightedTrainingLimits limits={});
  WeightedTrainingSession(Graph,Model,const ResidentTrainingCheckpoint&,at::Device,WeightedTrainingLimits={});
  WeightedTrainingSession(Graph,Model,const Continuation&,at::Device,ResidentOptimizerKind,
    std::vector<OptimizerGroup>,WeightedTrainingLimits,std::vector<at::Device> node_devices);
  WeightedTrainingSession(Graph,Model,const ResidentTrainingCheckpoint&,at::Device,WeightedTrainingLimits,std::vector<at::Device> node_devices);
  ~WeightedTrainingSession();
  WeightedTrainingSession(const WeightedTrainingSession&)=delete;
  WeightedTrainingSession& operator=(const WeightedTrainingSession&)=delete;
  WeightedTrainingWindow advance(const std::vector<External>&,Index stop);
  WeightedGradients backward(const std::vector<WeightedCotangents>&);
  void accumulate(); // Sum detached groups without changing optimizer boundary.
  ResidentStep step();
  // Independent sample replicas, identical parameters/optimizer state. Sum
  // gradients once (never average), agree on refusal, then update every replica.
  static WeightedPeerStep step_replicas(const std::vector<WeightedTrainingSession*>&,
    Index communication_bytes=256*1024*1024,Index timeout_ms=30000);
  void detach();
  ResidentTrainingCheckpoint checkpoint()const; // No outstanding tapes/gradients.
  Result result()const; // Explicit diagnostic decode of the latest actual window.
  Index cut()const;
  Index generation()const;
  ResidentOptimizerKind optimizer_kind()const;
  std::vector<OptimizerGroup> optimizer_groups()const;
  std::vector<at::Device> node_devices()const;
  void close();
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
}
