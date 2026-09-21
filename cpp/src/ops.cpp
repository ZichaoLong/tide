#include "tide/ops.h"
#include <torch/csrc/autograd/custom_function.h>
#include <stdexcept>

namespace tide {
namespace {
class HST : public torch::autograd::Function<HST> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor h, Tensor g, Tensor p, double zeta) {
    ctx->save_for_backward({g - h});
    ctx->saved_data["zeta"] = zeta;
    return g.clone();
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                torch::autograd::variable_list grad) {
    const auto delta = ctx->get_saved_variables().at(0);
    return {at::zeros_like(grad[0]), grad[0],
            (grad[0] * delta).sum(-1) * ctx->saved_data["zeta"].toDouble(), Tensor()};
  }
};
}  // namespace
Tensor aggregate(const Model& m, const std::vector<Atom>& atoms) {
  Tensor h;
  for (const auto& a : atoms) {
    auto value = a.value * (a.kind == 0 ? m.input_scale.at(a.source) : m.agg_scale.at(a.source));
    h = h.defined() ? h + value : value;
  }
  return h;
}
Tensor emit(const Tensor& h, const Tensor& g, const Tensor& p, const std::string& mode, double zeta) {
  if (mode == "hard") return g;
  if (mode == "softp") return h + p.unsqueeze(-1) * (g - h);
  if (mode == "hst") return HST::apply(h, g, p, zeta);
  throw std::invalid_argument("invalid emit mode");
}
Tensor full(const NodeWeights& w, const Tensor& comparison, const Tensor& h,
            const Tensor& p, const Options& options) {
  auto g = h + at::tanh(at::matmul(comparison, w.weight) + w.bias);
  return emit(h, g, p, options.mode, options.zeta);
}
}  // namespace tide
