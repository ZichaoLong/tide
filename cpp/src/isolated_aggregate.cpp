#include "tide/isolated_aggregate.h"
#include <torch/csrc/autograd/custom_function.h>
#include <stdexcept>

namespace tide {
namespace {
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
class IsolatedAggregate final : public torch::autograd::Function<IsolatedAggregate> {
 public:
  static variable_list forward(AutogradContext* ctx, at::TensorList atoms,
      at::TensorList scales, Tensor coefficients, Index sources, bool mean) {
    ctx->set_materialize_grads(false);
    const Index rows = atoms.size()/sources;
    auto x = at::stack(atoms).reshape({rows, sources, -1});
    auto s = at::stack(scales).reshape({rows, sources});
    auto values = x*s.unsqueeze(-1);
    if (mean) values = values/static_cast<double>(sources);
    else if (coefficients.numel()) values = values*coefficients.reshape({coefficients.dim() == 2 ? rows : 1, sources, 1});
    // Match the scalar program's source-order addition, not a different reduction.
    auto total = values.select(1, 0);
    for (Index j = 1; j < sources; ++j) total = total+values.select(1, j);
    variable_list outputs, frozen;
    for (Index i = 0; i < rows; ++i) {
      auto summary = total[i]; outputs.push_back(summary);
      bool active = false;
      for (Index j = 0; j < sources; ++j) {
        const auto k = i*sources+j;
        auto value = values[i][j]; outputs.push_back(value);
        const bool needs = atoms[k].requires_grad() || scales[k].requires_grad() || coefficients.requires_grad();
        active |= needs;
        if (!needs) frozen.push_back(value);
      }
      if (!active) frozen.push_back(summary);
    }
    ctx->mark_non_differentiable(frozen);
    ctx->saved_data["sources"] = sources;
    ctx->saved_data["mean"] = mean;
    // Packing owns snapshots, like stack followed by elementwise multiplication.
    ctx->save_for_backward({x, s, coefficients});
    return outputs;
  }
  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    if (at::GradMode::is_enabled()) throw std::invalid_argument("isolated Aggregate supports first-order VJP only");
    const auto sources = ctx->saved_data["sources"].toInt();
    const bool mean = ctx->saved_data["mean"].toBool();
    const Index rows = grads.size()/(sources+1), count = rows*sources;
    variable_list result(2*count+3);
    const auto saved = ctx->get_saved_variables();
    const auto& x = saved[0]; const auto& s = saved[1]; const auto& coe = saved[2];
    Tensor dc;
    for (Index j = 0; j < sources; ++j) {
      std::vector<Index> used;
      variable_list bars;
      bool need_x = false, need_s = false;
      for (Index i = 0; i < rows; ++i) {
        const auto& total = grads[i*(sources+1)];
        const auto& part = grads[i*(sources+1)+1+j];
        if (!total.defined() && !part.defined()) continue;
        used.push_back(i);
        bars.push_back(total.defined() && part.defined() ? total+part : total.defined() ? total : part);
        need_x |= ctx->needs_input_grad(i*sources+j);
        need_s |= ctx->needs_input_grad(count+i*sources+j);
      }
      if (used.empty()) continue; // Numerical zero is still a connected cotangent.
      auto ids = at::tensor(used, x.options().dtype(at::kLong));
      auto dy = at::stack(bars);
      auto xs = x.select(1,j).index_select(0,ids);
      auto ss = s.select(1,j).index_select(0,ids);
      auto dv = dy;
      if (mean) dv = dy/static_cast<double>(sources);
      else if (coe.numel()) dv = dy*(coe.dim() == 2 ? coe.index_select(0,ids).select(1,j).unsqueeze(-1) : coe[j]);
      if (need_x) {
        auto dx = dv*ss.unsqueeze(-1);
        for (size_t k = 0; k < used.size(); ++k)
          if (ctx->needs_input_grad(used[k]*sources+j)) result[used[k]*sources+j] = dx[k];
      }
      if (need_s) {
        auto ds = (dv*xs).sum(-1);
        for (size_t k = 0; k < used.size(); ++k)
          if (ctx->needs_input_grad(count+used[k]*sources+j)) result[count+used[k]*sources+j] = ds[k];
      }
      if (coe.numel() && ctx->needs_input_grad(2*count)) {
        if (!dc.defined()) dc = at::zeros_like(coe);
        auto terms = dy*(xs*ss.unsqueeze(-1));
        if (coe.dim() == 2) dc.select(1,j).index_copy_(0,ids,terms.sum(-1));
        else dc.select(0,j).copy_(terms.sum());
      }
    }
    result[2*count] = dc;
    return result;
  }
};
} // namespace
std::vector<Tensor> isolated_aggregate(const std::vector<Tensor>& atoms,
    const std::vector<Tensor>& scales, const Tensor& coefficients, Index sources, bool mean) {
  if (sources < 1 || atoms.empty() || atoms.size()%sources || atoms.size() != scales.size())
    throw std::invalid_argument("isolated Aggregate source layout mismatch");
  const auto& first = atoms.front();
  if (!first.defined() || first.dim() != 1 || !first.device().is_cpu()
      || (first.scalar_type() != at::kFloat && first.scalar_type() != at::kDouble))
    throw std::invalid_argument("isolated Aggregate requires CPU FP32/FP64 vectors");
  for (size_t i = 0; i < atoms.size(); ++i)
    if (!atoms[i].defined() || atoms[i].sizes() != first.sizes() || atoms[i].options().dtype() != first.options().dtype()
        || atoms[i].device() != first.device() || !scales[i].defined() || scales[i].dim() != 0
        || scales[i].scalar_type() != first.scalar_type() || scales[i].device() != first.device())
      throw std::invalid_argument("isolated Aggregate input metadata mismatch");
  const bool shape = coefficients.defined() && ((coefficients.dim() == 1 && (!coefficients.numel() || coefficients.numel() == sources))
      || (coefficients.dim() == 2 && coefficients.size(0) == static_cast<Index>(atoms.size())/sources && coefficients.size(1) == sources));
  if (!shape
      || coefficients.scalar_type() != first.scalar_type() || coefficients.device() != first.device()
      || (mean && coefficients.numel())) throw std::invalid_argument("isolated Aggregate coefficients mismatch");
  return IsolatedAggregate::apply(at::TensorList(atoms), at::TensorList(scales), coefficients, sources, mean);
}
} // namespace tide
