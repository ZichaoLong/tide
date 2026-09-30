#pragma once
#include "parameter_vjp.h"
#include "tide/optimizer.h"
#include "tide/resident_training.h"

namespace tide::device_online {
enum class DeviceOptimizerKind {sgd,adamw};
// Internal first-order FP32 owner update. Mutable numerical state and decisions
// remain on device; a finite-check failure commits no parameter or slot. This
// does not itself publish parameters into a forward owner's frozen banks.
class DeviceOptimizer {
 public:
  DeviceOptimizer(const ParameterVjp&,DeviceOptimizerKind,std::vector<OptimizerGroup>,int64_t tensor_budget_bytes);
  void append_step(CannProgram&,const ParameterVjp&,const at::Tensor& error);
  const at::Tensor& values() const {return values_;}
  const at::Tensor& first() const {return first_;}
  const at::Tensor& second() const {return second_;}
  const at::Tensor& maximum() const {return maximum_;}
  const at::Tensor& steps() const {return steps_;}
  const at::Tensor& corrections() const {return corrections_;}
  const std::vector<OptimizerGroup>& groups() const {return groups_;}
  ResidentOptimizerState snapshot() const;
  void restore(const ResidentOptimizerState&); // Validate all CPU fields before device writes.
 private:
  ParameterVjp identity_;
  std::vector<OptimizerGroup> groups_;
  DeviceOptimizerKind kind_;
  int64_t tasks_,count_;
  at::Tensor table_,tiles_,options_,flags_,values_,first_,second_,maximum_,steps_,corrections_;
  at::Tensor next_values_,next_first_,next_second_,next_maximum_,next_steps_,next_corrections_,tile_errors_;
};
} // namespace tide::device_online
