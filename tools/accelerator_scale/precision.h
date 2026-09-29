#pragma once
#include "training.h"
#include "training_threads.h"

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
  CpuThreadCounts backward_threads() const {return backward_counts_;}
  CpuThreadCounts optimizer_threads() const {return optimizer_counts_;}
 private:
  std::vector<Tensor> payload_, masters_;
  ParameterRegistry registry_;
  std::unique_ptr<NamedOptimizer> optimizer_;
  double loss_scale_;
  bool half_;
  int backward_threads_=0, optimizer_threads_=0;
  CpuThreadCounts backward_counts_,optimizer_counts_;
};
}  // namespace accelerator_scale
