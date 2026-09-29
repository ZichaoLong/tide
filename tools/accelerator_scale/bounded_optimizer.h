#pragma once
#include "bounded.h"

namespace accelerator_scale::bounded {
struct OptimizerState { Tensor master, first, second, step; };
// Device predicates, including finite preflight, gate all owner/slot updates.
// Host code never reads the used-owner mask inside a captured training window.
class Optimizer {
 public:
  Optimizer(const std::vector<Tensor>&, std::string kind, double lr=1e-4, double scale=1., Index max_updates=1024);
  Tensor step(const Value& objective, const std::vector<Tensor>& gradients);
  std::vector<OptimizerState> snapshot() const;
  void reset();
 private:
  std::vector<Tensor> payload_, initial_;
  std::vector<OptimizerState> states_;
  std::string kind_;
  double lr_, scale_;
  Index max_updates_;
  std::map<std::string,std::pair<Tensor,Tensor>> corrections_;
};
}  // namespace accelerator_scale::bounded
