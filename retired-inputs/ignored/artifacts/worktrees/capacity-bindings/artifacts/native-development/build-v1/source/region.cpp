#include "region.h"
#include "runtime.h"
#include <torch/autograd.h>
#include <cmath>

namespace tide_native {
namespace {
// Nonempty rows retain the right derivative of an explicitly zero prior.
// The chosen VJP of an entirely empty row is zero, as in Tide's math path.
class WeightedSoftmax : public torch::autograd::Function<WeightedSoftmax> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                            at::Tensor logits, at::Tensor weights) {
    auto positive = weights > 0;
    auto weighted = at::where(positive,
        logits + at::where(positive, weights, at::ones_like(weights)).log(), -INFINITY);
    auto nonempty = positive.any(-1, true);
    auto safe = at::where(nonempty, weighted, at::zeros_like(weighted));
    auto log_z = safe.logsumexp(-1, true);
    auto p = at::where(nonempty, (safe - log_z).exp(), at::zeros_like(safe));
    ctx->save_for_backward({logits, p, log_z, nonempty});
    return p;
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad) {
    TORCH_CHECK(!at::GradMode::is_enabled(), "regional weighted softmax supports first-order VJPs only");
    auto saved = ctx->get_saved_variables();
    auto centered = grad[0] - (saved[1] * grad[0]).sum(-1, true);
    auto factor = at::where(saved[3], saved[0] - saved[2], -INFINITY).exp();
    return {saved[1] * centered, factor * centered};
  }
};
at::Tensor norm(const at::Tensor& x) {
  auto work = x.to(at::kFloat);
  return (work * (work.square().mean(-1, true) + 1e-5).rsqrt()).to(x.scalar_type());
}
}  // namespace

Region::Region(const Tensors& source, const at::Device& device) {
  const std::vector<std::string> names = {"q_weight", "k_weight", "v_weight", "o_weight",
      "first_weight", "first_bias", "second_weight", "second_bias", "readout"};
  auto z = source.at("z");
  TORCH_CHECK(z.dim() == 5, "z shape must be [tasks,nodes,batch,positions,width]");
  width = z.size(-1);
  TORCH_CHECK(width >= 32 && width % 32 == 0, "description width must be a positive multiple of 32");
  heads = width / 32;
  for (const auto& name : names) {
    auto tensor = source.at(name);
    const auto expected = name == "readout" || name == "second_bias"
        ? std::vector<int64_t>{width} : name == "first_bias"
        ? std::vector<int64_t>{2 * width} : name == "first_weight"
        ? std::vector<int64_t>{2 * width, width} : name == "second_weight"
        ? std::vector<int64_t>{width, 2 * width} : std::vector<int64_t>{width, width};
    TORCH_CHECK(tensor.sizes().vec() == expected, "parameter shape mismatch: ", name);
    values[name] = register_parameter(name, tensor.to(device).clone().set_requires_grad(true));
  }
}
at::Tensor Region::forward(const at::Tensor& z, const at::Tensor& score,
                          const at::Tensor& w, bool bf16) {
  AutocastGuard precision(z.device().type(), bf16);
  auto x = z.permute({1, 2, 0, 3});
  auto project = [&](const std::string& name) {
    auto result = at::linear(x, values.at(name));
    TORCH_CHECK(result.scalar_type() == (bf16 ? at::kBFloat16 : at::kFloat),
                "projection did not execute the selected arithmetic policy");
    return result.reshape({x.size(0), x.size(1), x.size(2), heads, 32}).transpose(-3, -2);
  };
  auto q = project("q_weight"), k = project("k_weight"), v = project("v_weight");
  at::Tensor read;
  {
    AutocastGuard stable(z.device().type(), false);
    auto logits = q.to(at::kFloat).matmul(k.to(at::kFloat).transpose(-2, -1)) / std::sqrt(32.);
    auto weights = w.permute({1, 2, 0}).unsqueeze(-2).unsqueeze(-2).expand_as(logits);
    auto p = WeightedSoftmax::apply(logits, weights);
    read = p.matmul(v.to(at::kFloat)).to(q.scalar_type());
  }
  auto delta = at::linear(read.transpose(-3, -2).flatten(-2), values.at("o_weight"));
  auto hidden = at::silu(at::linear(norm(x + delta), values.at("first_weight"), values.at("first_bias")));
  AutocastGuard stable(z.device().type(), false);
  auto correction = delta.to(at::kFloat).matmul(values.at("readout"));
  correction = correction + hidden.to(at::kFloat).matmul(
      values.at("second_weight").t().matmul(values.at("readout")));
  correction = correction + values.at("second_bias").matmul(values.at("readout"));
  return score + at::where(w > 0, correction.permute({2, 0, 1}), at::zeros_like(score));
}
}  // namespace tide_native
