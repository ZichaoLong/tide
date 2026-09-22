#include "tide/dense.h"
#include <algorithm>
#include <stdexcept>

namespace tide {
Tensor DenseLinear::run(const Tensor& input, const Tensor& weight, const Tensor& bias) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (input.dim() != 2 || weight.dim() != 2 || input.size(1) != weight.size(1)
      || !input.device().is_cpu() || weight.device() != input.device()
      || (input.scalar_type() != at::kFloat && input.scalar_type() != at::kDouble)
      || weight.scalar_type() != input.scalar_type())
    throw std::invalid_argument("dense projection requires compatible CPU FP32/FP64 matrices");
  if (bias.defined() && (bias.dim() != 1 || bias.size(0) != weight.size(0)
      || bias.device() != input.device() || bias.scalar_type() != input.scalar_type()))
    throw std::invalid_argument("dense projection bias metadata mismatch");
  const auto count = std::min(workers_, weight.size(0));
  if (count < 2) return at::linear(input, weight, bias);
  std::vector<Tensor> blocks(count);
  std::vector<std::function<void()>> jobs;
  for (Index i = 0; i < count; ++i) jobs.push_back([&, i] {
    const auto begin = weight.size(0)*i/count, end = weight.size(0)*(i+1)/count;
    blocks[i] = at::linear(input, weight.slice(0, begin, end),
                          bias.defined() ? bias.slice(0, begin, end) : Tensor());
  });
  pool_.run(std::move(jobs));
  return at::cat(blocks, 1);
}
} // namespace tide
