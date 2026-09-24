#include "tide/isolated_linear.h"
#include <torch/csrc/autograd/custom_function.h>
#include <stdexcept>

namespace tide {
namespace {
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
class IsolatedLinear final : public torch::autograd::Function<IsolatedLinear> {
 public:
  static variable_list forward(AutogradContext* ctx, at::TensorList rows, Tensor weight) {
    ctx->set_materialize_grads(false);
    auto x = at::stack(rows);
    auto outputs = at::linear(x, weight).unbind(0);
    variable_list frozen;
    for (size_t i = 0; i < rows.size(); ++i)
      if (!rows[i].requires_grad() && !weight.requires_grad()) frozen.push_back(outputs[i]);
    ctx->mark_non_differentiable(frozen);
    // Stack owns a snapshot; like ordinary stack/linear it need not retain the
    // original row values. SavedVariable checks weight mutation/version on VJP.
    ctx->save_for_backward({x, weight});
    return outputs;
  }
  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    if (at::GradMode::is_enabled()) throw std::invalid_argument("isolated linear supports first-order VJP only");
    variable_list result(grads.size()+1), used_grads;
    std::vector<Index> used;
    for (size_t i = 0; i < grads.size(); ++i) if (grads[i].defined()) {
      used.push_back(i); used_grads.push_back(grads[i]);
    }
    if (used.empty()) return result;  // No numerical-zero test: undefined only.
    const auto saved = ctx->get_saved_variables();
    auto dy = at::stack(used_grads);
    bool need_rows = false;
    for (auto i : used) need_rows |= ctx->needs_input_grad(i);
    if (need_rows) {
      auto dx = at::matmul(dy, saved[1]);
      for (size_t j = 0; j < used.size(); ++j)
        if (ctx->needs_input_grad(used[j])) result[used[j]] = dx[j];
    }
    if (ctx->needs_input_grad(grads.size())) {
      auto ids = at::tensor(used, at::TensorOptions().dtype(at::kLong).device(saved[0].device()));
      result.back() = at::matmul(dy.t(), saved[0].index_select(0, ids));
    }
    return result;
  }
};
}  // namespace
std::vector<Tensor> isolated_linear(const std::vector<Tensor>& rows, const Tensor& weight) {
  if (rows.empty()) return {};
  if (!weight.defined() || weight.dim() != 2 || !weight.device().is_cpu()
      || (weight.scalar_type() != at::kFloat && weight.scalar_type() != at::kDouble))
    throw std::invalid_argument("isolated linear requires CPU FP32/FP64 weight matrix");
  for (const auto& row : rows)
    if (!row.defined() || row.dim() != 1 || row.size(0) != weight.size(1)
        || row.device() != weight.device() || row.scalar_type() != weight.scalar_type())
      throw std::invalid_argument("isolated linear row metadata mismatch");
  // Explicit TensorList is essential: custom Function does not traverse vectors.
  return IsolatedLinear::apply(at::TensorList(rows), weight);
}
}  // namespace tide
