#pragma once

#include "tide/parameters.h"
#include <map>
#include <string>
#include <vector>

namespace tide {

class Checkpoint;
class GraphCheckpoint;

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

// Explicit half payload policy. Masters and optimizer slots are FP32; scaling
// is static and serialized. No clipping, skipped update or automatic fallback.
struct OptimizerPrecision {
  bool fp32_masters = false;
  double loss_scale = 1.;
};

class NamedOptimizer {
 public:
  virtual ~NamedOptimizer() = default;
  void step();
  void zero_grad(bool set_to_none = true);
  void backward(const Tensor& loss, bool retain_graph = false);
  const ParameterRegistry& registry() const { return registry_; }
  const std::vector<OptimizerGroup>& groups() const { return groups_; }
  OptimizerLayout layout() const;
  const std::map<std::string, OptimizerState>& state() const { return state_; }
  const OptimizerPrecision& precision() const { return precision_; }
  const std::map<std::string, Tensor>& masters() const { return masters_; }

 protected:
  NamedOptimizer(ParameterRegistry& registry, std::vector<OptimizerGroup> groups,
                 std::string class_name, OptimizerPrecision precision = {});
  virtual void update(const std::string& name, const Tensor& parameter,
                      const Tensor& gradient, const OptimizerGroup& group) = 0;
  std::map<std::string, OptimizerState> state_;

 private:
  friend class Checkpoint;
  friend class GraphCheckpoint;
  ParameterRegistry& registry_;
  std::vector<OptimizerGroup> groups_;
  std::string class_name_;
  OptimizerPrecision precision_;
  std::map<std::string, Tensor> masters_;
  bool scaled_backward_ = false;
  void initialize_precision();
  void master_step();
};

class SGD final : public NamedOptimizer {
 public:
  explicit SGD(ParameterRegistry& registry,
               std::vector<OptimizerGroup> groups = {}, OptimizerPrecision precision = {});

 protected:
  void update(const std::string& name, const Tensor& parameter,
              const Tensor& gradient, const OptimizerGroup& group) override;
};

class AdamW final : public NamedOptimizer {
 public:
  explicit AdamW(ParameterRegistry& registry,
                 std::vector<OptimizerGroup> groups = {}, OptimizerPrecision precision = {});

 protected:
  void update(const std::string& name, const Tensor& parameter,
              const Tensor& gradient, const OptimizerGroup& group) override;
};

}  // namespace tide
