#include "tide/lh_full.h"
#include <stdexcept>

namespace tide {
namespace {
enum class Act { Relu, Silu, Identity };
enum class Norm { Identity, Rms, Layer };
const std::map<std::string, std::pair<Act, Norm>> profiles{
  {"lh-relu-identity-v1", {Act::Relu, Norm::Identity}}, {"lh-relu-rms-v1", {Act::Relu, Norm::Rms}},
  {"lh-relu-layer-v1", {Act::Relu, Norm::Layer}}, {"lh-silu-identity-v1", {Act::Silu, Norm::Identity}},
  {"lh-silu-rms-v1", {Act::Silu, Norm::Rms}}, {"lh-silu-layer-v1", {Act::Silu, Norm::Layer}},
  {"lh-identity-identity-v1", {Act::Identity, Norm::Identity}},
  {"lh-identity-rms-v1", {Act::Identity, Norm::Rms}}, {"lh-identity-layer-v1", {Act::Identity, Norm::Layer}}};
}  // namespace
bool is_lh_full(const std::string& name) { return profiles.count(name); }
void validate_lh_full(const NodeWeights& w) {
  const auto norm = profiles.at(w.full_kind).second;
  std::vector<std::string> names;
  if (norm != Norm::Identity) names.push_back("lh_norm_weight");
  if (norm == Norm::Layer) names.push_back("lh_norm_bias");
  for (const auto& name : names) {
    auto it = w.extra.find(name);
    if (it == w.extra.end() || !it->second.defined() || it->second.sizes() != w.bias.sizes()
        || it->second.scalar_type() != w.bias.scalar_type() || it->second.device() != w.bias.device()
        || !at::isfinite(it->second).all().item<bool>())
      throw std::invalid_argument("invalid LH Full normalization parameter");
  }
}
Tensor lh_full_fresh(const NodeWeights& w, const Tensor& comparison) {
  const auto [act, norm] = profiles.at(w.full_kind);
  auto value = act == Act::Relu ? at::relu(comparison) : act == Act::Silu ? at::silu(comparison) : comparison;
  const auto d = w.bias.numel();
  if (norm == Norm::Rms) return at::rms_norm(value, {d}, w.extra.at("lh_norm_weight"), 1e-7);
  if (norm == Norm::Layer) return at::layer_norm(value, {d}, w.extra.at("lh_norm_weight"), w.extra.at("lh_norm_bias"), 1e-5);
  return value;
}
}  // namespace tide
