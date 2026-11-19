#pragma once
#include <torch/nn/module.h>
#include <map>
#include <string>

namespace tide_native {
using Tensors = std::map<std::string, at::Tensor>;
struct Region : torch::nn::Module {
  Region(const Tensors& source, const at::Device& device);
  at::Tensor forward(const at::Tensor& z, const at::Tensor& score,
                     const at::Tensor& w, bool bf16);
  std::map<std::string, at::Tensor> values;
  int64_t width, heads;
};
}  // namespace tide_native
