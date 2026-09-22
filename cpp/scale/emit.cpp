#include "tide/operator_work.h"
#include "scale.h"
#include "tide/ops.h"
#include "tide/lh_full.h"
#include <stdexcept>

namespace pdg_scale {
namespace {
// A benchmark instance of the public FullKernel interface. One dense Linear
// per node row; phase aliases route slices without multiplying parameters.
class RowEmit final : public tide::FullKernel {
  std::vector<Index> logical_, phases_;
  Index period_, targets_;
 public:
  RowEmit(std::vector<Index> logical, std::vector<Index> phases, Index period, Index targets)
      : logical_(std::move(logical)), phases_(std::move(phases)), period_(period), targets_(targets) {}
  bool joint_batch() const override { return true; }
  std::vector<tide::FullResult> batch(const tide::NodeWeights& w, const std::vector<tide::FullInput>& inputs,
                                    Index slots, const tide::Options& options) const override {
    if (options.mode != "hard") throw std::invalid_argument("scale row Emit implements hard mode only");
    if (slots != static_cast<Index>(logical_.size())) throw std::invalid_argument("row Emit slot count");
    std::vector<at::Tensor> rows;
    for (const auto& input : inputs) rows.push_back(input.comparison->value);
    auto fresh = tide::lh_full_fresh(w, at::stack(rows));
    auto projected = targets_ ? at::linear(fresh, w.extra.at("row_emit_weight")) : at::Tensor();
    const auto width = w.bias.numel();
    if (tide::work::enabled()) {
      Index pending = 0;
      for (const auto& input : inputs) pending += input.time%period_ == period_-2;
      tide::work::emit(inputs.size(), width, targets_, pending);
    }
    std::vector<tide::FullResult> result;
    for (size_t i = 0; i < inputs.size(); ++i) {
      tide::FullResult r{fresh[i], {}};
      for (Index slot = 0; slot < slots; ++slot) if (inputs[i].time%period_ == phases_[slot]) {
        const auto logical = logical_[slot];
        r.emitted.push_back({slot, logical < 0 ? fresh[i] : projected[i].slice(0, logical*width, (logical+1)*width)});
      }
      result.push_back(std::move(r));
    }
    return result;
  }
  tide::FullResult step(const tide::NodeWeights& w, const tide::FullInput& input,
                        Index slots, const tide::Options& options) const override {
    return batch(w, {input}, slots, options).at(0);
  }
  void validate_weights(const tide::NodeWeights& w, Index slots) const override {
    tide::validate_lh_full(w);
    if (slots != static_cast<Index>(logical_.size()) || logical_.size() != phases_.size())
      throw std::invalid_argument("row Emit domain mismatch");
    if (targets_ && w.extra.at("row_emit_weight").sizes() != at::IntArrayRef({targets_*w.bias.numel(), w.bias.numel()}))
      throw std::invalid_argument("row Emit weight shape mismatch");
    for (size_t i = 0; i < logical_.size(); ++i)
      if (logical_[i] < -1 || logical_[i] >= targets_ || phases_[i] < 0 || phases_[i] >= period_)
        throw std::invalid_argument("row Emit mapping invalid");
  }
};
}
std::shared_ptr<const tide::FullKernel> row_emit(std::vector<Index> logical, std::vector<Index> phases,
                                              Index period, Index targets) {
  return std::make_shared<RowEmit>(std::move(logical), std::move(phases), period, targets);
}
} // namespace pdg_scale
