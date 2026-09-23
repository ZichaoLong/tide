#include "tide/operator_profile.h"
#include "fiber_pool.h"
#include "tide/fiber_attention.h"
#include <stdexcept>

namespace tide {
namespace {
const std::map<std::string, std::string> profiles{
  {"lh-fiber-attention-sum-repeat-v1", "sum"}, {"lh-fiber-attention-mean-repeat-v1", "mean"},
  {"lh-fiber-attention-linear-repeat-v1", "linear"},
  {"lh-fiber-attention-active-softmax-repeat-v1", "active-softmax"},
  {"lh-fiber-attention-all-softmax-repeat-v1", "all-softmax"}};
}
bool is_fiber_attention_profile(const std::string& name) { return profiles.count(name); }
std::string fiber_pool_kind(const std::string& name) {
  auto found = profiles.find(name);
  if (found == profiles.end()) throw std::invalid_argument("unknown fiber attention profile");
  return found->second;
}
bool fiber_pool_learned(const std::string& kind) {
  return kind == "linear" || kind == "active-softmax" || kind == "all-softmax";
}
void validate_fiber_pool(const NodeWeights& w, const std::string& kind, Index slots) {
  if (!fiber_pool_learned(kind)) return;
  auto it = w.extra.find("fiber_pool");
  if (slots < 0 || it == w.extra.end() || !it->second.defined() || it->second.sizes() != at::IntArrayRef({slots})
      || it->second.scalar_type() != w.bias.scalar_type() || it->second.device() != w.bias.device()
      || !at::isfinite(it->second).all().item<bool>())
    throw std::invalid_argument("invalid fiber pooling parameter/domain");
}
Tensor fiber_pool_rows(const NodeWeights& w, const std::string& kind, const std::vector<Index>& slots,
                       const Tensor& rows) {
  op_profile::Scope profile(op_profile::Pooling);
  if (kind == "sum") return rows.sum(0);
  if (kind == "mean") return rows.mean(0);
  if (!fiber_pool_learned(kind)) throw std::invalid_argument("unknown fiber pooling kind");
  const auto& weight = w.extra.at("fiber_pool");
  auto index = at::tensor(slots, rows.options().dtype(at::kLong));
  auto coefficient = kind == "all-softmax" ? at::softmax(weight, 0).index_select(0, index) : weight.index_select(0, index);
  if (kind == "active-softmax") coefficient = at::softmax(coefficient, 0);
  return at::matmul(coefficient, rows);
}
}  // namespace tide
