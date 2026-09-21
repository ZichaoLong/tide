#include "tide/full.h"
#include "tide/ops.h"
#include <stdexcept>

namespace tide {
std::vector<FullResult> FullKernel::batch(const NodeWeights& w, const std::vector<FullInput>& inputs,
                                        Index slots, const Options& options) const {
  std::vector<FullResult> results;
  for (const auto& input : inputs) results.push_back(step(w, input, slots, options));
  return results;
}
namespace {
class ProjectionEmit final : public FullKernel {
  std::string kind_;
  Index period_;
  std::vector<Index> phases_;
  bool identity_;
  bool present(Index slot, Index time) const {
    const auto phase = phases_.empty() ? -1 : phases_[slot];
    return phase == -1 || (phase >= 0 && time % period_ == phase);
  }
 public:
  explicit ProjectionEmit(const Node& n) : kind_(n.emission), period_(n.emit_period), phases_(n.emit_phases), identity_(n.identity) {
    if (kind_ != "broadcast" && kind_ != "slot_affine") throw std::invalid_argument("unknown emission program");
  }
  bool joint_batch() const override { return true; }
  FullResult step(const NodeWeights& w, const FullInput& input, Index slots, const Options& options) const override {
    const auto& h = input.content.value; const auto& p = input.control;
    auto fresh = full_fresh(w, input.comparison->value, h, identity_);
    FullResult result{identity_ ? h : emit(h, fresh, p, options.mode, options.zeta), {}};
    for (Index slot = 0; slot < slots; ++slot) {
      if (!present(slot, input.time)) continue;
      auto value = result.value;
      if (kind_ == "slot_affine") {
        const auto& weight = w.extra.at("emit_w_" + std::to_string(slot));
        auto held = at::matmul(h, weight), active = at::matmul(fresh, weight) + w.extra.at("emit_b_" + std::to_string(slot));
        value = emit(held, active, p, options.mode, options.zeta);
      }
      result.emitted.push_back({slot, value});
    }
    return result;
  }
  std::vector<FullResult> batch(const NodeWeights& w, const std::vector<FullInput>& inputs,
                                Index slots, const Options& options) const override {
    std::vector<Tensor> comparisons, contents, controls;
    for (const auto& input : inputs) {
      comparisons.push_back(input.comparison->value); contents.push_back(input.content.value); controls.push_back(input.control);
    }
    auto h = at::stack(contents), p = at::stack(controls);
    auto fresh = full_fresh(w, at::stack(comparisons), h, identity_);
    auto values = identity_ ? h : emit(h, fresh, p, options.mode, options.zeta);
    std::vector<FullResult> results;
    for (size_t i = 0; i < inputs.size(); ++i) results.push_back({values[i], {}});
    for (Index slot = 0; slot < slots; ++slot) {
      std::vector<Index> rows;
      for (size_t i = 0; i < inputs.size(); ++i) if (present(slot, inputs[i].time)) rows.push_back(i);
      if (rows.empty()) continue;
      if (kind_ == "broadcast") {
        for (auto i : rows) results[i].emitted.push_back({slot, results[i].value});
      } else {
        auto index = at::tensor(rows, at::TensorOptions().dtype(at::kLong).device(h.device()));
        const auto& weight = w.extra.at("emit_w_" + std::to_string(slot));
        auto projected = emit(at::matmul(h.index_select(0, index), weight),
                              at::matmul(fresh.index_select(0, index), weight) + w.extra.at("emit_b_" + std::to_string(slot)),
                              p.index_select(0, index), options.mode, options.zeta);
        for (size_t j = 0; j < rows.size(); ++j) results[rows[j]].emitted.push_back({slot, projected[j]});
      }
    }
    return results;
  }
  void validate_weights(const NodeWeights& w, Index slots) const override {
    const auto d = w.bias.numel();
    auto check = [&](const std::string& name, at::IntArrayRef shape) {
      auto it = w.extra.find(name);
      if (it == w.extra.end() || it->second.sizes() != shape)
        throw std::invalid_argument("Full program parameter slot domain mismatch");
    };
    if (w.full_kind == "swiglu") {
      check("ffn_gate", {d, 2*d}); check("ffn_up", {d, 2*d}); check("ffn_down", {2*d, d});
    } else if (w.full_kind != "tanh" && w.full_kind != "identity") throw std::invalid_argument("unknown Full profile");
    if (kind_ == "slot_affine") for (Index slot = 0; slot < slots; ++slot) {
      check("emit_w_" + std::to_string(slot), {d, d}); check("emit_b_" + std::to_string(slot), {d});
    }
  }
};
}  // namespace
std::shared_ptr<const FullKernel> make_full_kernel(const Node& n) { return std::make_shared<ProjectionEmit>(n); }
}  // namespace tide
