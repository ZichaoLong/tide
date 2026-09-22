#pragma once

#include "tide/parameters.h"
#include <map>
#include <string>
#include <vector>

namespace tide {

// The group keeps the ordered owner names and all options needed by the two
// built-in optimizers.  An optimizer normalizes aliases to canonical names at
// construction, then rejects duplicate owners across groups.
struct OptimizerGroup {
  std::vector<std::string> parameters;
  double lr = 1e-3;
  double weight_decay = 0.0;
  double momentum = 0.0;
  double dampening = 0.0;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double eps = 1e-8;
  bool nesterov = false;
  bool amsgrad = false;
  bool maximize = false;
};

struct OptimizerLayout {
  std::string class_name;
  std::vector<std::vector<std::string>> groups;
};

struct OptimizerState {
  Index step = 0;
  Tensor momentum_buffer;
  Tensor exp_avg;
  Tensor exp_avg_sq;
  Tensor max_exp_avg_sq;
};

class NamedOptimizer {
 public:
  virtual ~NamedOptimizer() = default;
  void step();
  void zero_grad(bool set_to_none = true);
  const ParameterRegistry& registry() const { return registry_; }
  const std::vector<OptimizerGroup>& groups() const { return groups_; }
  OptimizerLayout layout() const;
  const std::map<std::string, OptimizerState>& state() const { return state_; }

 protected:
  NamedOptimizer(ParameterRegistry& registry, std::vector<OptimizerGroup> groups,
                 std::string class_name);
  virtual void update(const std::string& name, const Tensor& parameter,
                      const Tensor& gradient, const OptimizerGroup& group) = 0;
  std::map<std::string, OptimizerState> state_;

 private:
  ParameterRegistry& registry_;
  std::vector<OptimizerGroup> groups_;
  std::string class_name_;
};

class SGD final : public NamedOptimizer {
 public:
  explicit SGD(ParameterRegistry& registry,
               std::vector<OptimizerGroup> groups = {});

 protected:
  void update(const std::string& name, const Tensor& parameter,
              const Tensor& gradient, const OptimizerGroup& group) override;
};

class AdamW final : public NamedOptimizer {
 public:
  explicit AdamW(ParameterRegistry& registry,
                 std::vector<OptimizerGroup> groups = {});

 protected:
  void update(const std::string& name, const Tensor& parameter,
              const Tensor& gradient, const OptimizerGroup& group) override;
};

}  // namespace tide
