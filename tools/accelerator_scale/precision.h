#pragma once
#include "training.h"

namespace accelerator_scale {
// Consumer-owned mixed precision. Public NamedOptimizer continues to receive
// FP32 owners; it never silently creates half-precision moments.
class TrainingOwners {
 public:
  TrainingOwners(const pdg_scale::Fixture&, const TrainingConfig&);
  void zero_grad();
  void backward(const Tensor& loss);
  void step();
  const std::vector<Tensor>& masters() const { return masters_; }
  const NamedOptimizer& optimizer() const { return *optimizer_; }
  double master_bytes() const;
 private:
  std::vector<Tensor> payload_, masters_;
  ParameterRegistry registry_;
  std::unique_ptr<NamedOptimizer> optimizer_;
  double loss_scale_;
  bool half_;
};
}  // namespace accelerator_scale
