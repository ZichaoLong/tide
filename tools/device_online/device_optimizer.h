#pragma once
#include "parameter_vjp.h"
#include "optimizer_storage.h"
#include "tide/optimizer.h"
#include "tide/resident_training.h"

namespace tide::device_online {
enum class DeviceOptimizerKind {sgd,adamw};
struct OptimizerParameterLayout {
  std::vector<ParameterOwner> owners;
  std::vector<int64_t> offsets;
  int64_t elements=1;
};
// Internal FP32 master update for FP32/FP16 payload owners. State and decisions
// remain on device; a finite-check failure commits no parameter or slot. This
// does not itself publish parameters into a forward owner's frozen banks.
class DeviceOptimizer {
 public:
  DeviceOptimizer(const ParameterVjp&,DeviceOptimizerKind,std::vector<OptimizerGroup>,int64_t tensor_budget_bytes);
  // A canonical identity needs no placeholder gradient allocation. Actual
  // gradient values/connection bits are supplied only when a step is appended.
  DeviceOptimizer(const OptimizerParameterLayout&,at::Device,DeviceOptimizerKind,
    std::vector<OptimizerGroup>,int64_t tensor_budget_bytes);
  void append_step(DeviceProgram&,const ParameterVjp&,const at::Tensor& error);
  // Distributed composition: evaluate proposals on all owners, reach a common
  // device error decision, then append every commit. No master/slot changes
  // occur in propose. Gradients/connection bits/old masters remain frozen until
  // commit recomputes the same elementwise update. Caller owns this ordering
  // and failure lifecycle; no full parameter-sized proposal bank is retained.
  void append_propose(DeviceProgram&,const ParameterVjp&,const at::Tensor& error);
  void append_commit(DeviceProgram&,const ParameterVjp&,const at::Tensor& error);
  const at::Tensor& values() const {return values_;}
  const at::Tensor& first() const {return first_;}
  const at::Tensor& second() const {return second_;}
  const at::Tensor& maximum() const {return maximum_;}
  const at::Tensor& steps() const {return steps_;}
  const at::Tensor& corrections() const {return corrections_;}
  const std::vector<OptimizerGroup>& groups() const {return groups_;}
  std::vector<at::Tensor> tensors()const;
  int64_t tensor_bytes()const;
  const OptimizerStoragePlan& storage_plan()const{return storage_;}
  ResidentOptimizerState snapshot() const;
  void restore(const ResidentOptimizerState&); // Validate all CPU fields before device writes.
 private:
  void append_phase(DeviceProgram&,const ParameterVjp&,const at::Tensor&,bool commit);
  OptimizerParameterLayout identity_;
  OptimizerStoragePlan storage_{};
  std::vector<OptimizerGroup> groups_;
  DeviceOptimizerKind kind_;
  int64_t tasks_,count_;
  at::Tensor table_,tiles_,options_,flags_,values_,first_,second_,maximum_,steps_,corrections_;
  at::Tensor next_steps_,next_corrections_,tile_errors_;
};
} // namespace tide::device_online
