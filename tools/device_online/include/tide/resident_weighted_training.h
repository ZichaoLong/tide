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
// Values use FP32; clocks, counts and coordinates remain separate int64 metadata.
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
// Explicit single-device FP32, complete weighted first-order training owner.
// Loss/task heads are consumer-owned; all calls require no-grad. Tokens name
// actual retained windows under one parameter generation, never a CPU replay.
class WeightedTrainingSession {
 public:
  WeightedTrainingSession(Graph,Model,const Continuation&,at::Device,
    ResidentOptimizerKind kind=ResidentOptimizerKind::sgd,std::vector<OptimizerGroup> groups={},WeightedTrainingLimits limits={});
  WeightedTrainingSession(Graph,Model,const ResidentTrainingCheckpoint&,at::Device,WeightedTrainingLimits={});
  ~WeightedTrainingSession();
  WeightedTrainingSession(const WeightedTrainingSession&)=delete;
  WeightedTrainingSession& operator=(const WeightedTrainingSession&)=delete;
  WeightedTrainingWindow advance(const std::vector<External>&,Index stop);
  WeightedGradients backward(const std::vector<WeightedCotangents>&);
  void accumulate(); // Sum detached groups without changing optimizer boundary.
  ResidentStep step();
  void detach();
  ResidentTrainingCheckpoint checkpoint()const; // No outstanding tapes/gradients.
  Result result()const; // Explicit diagnostic decode of the latest actual window.
  Index cut()const;
  Index generation()const;
  ResidentOptimizerKind optimizer_kind()const;
  std::vector<OptimizerGroup> optimizer_groups()const;
  void close();
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
}
