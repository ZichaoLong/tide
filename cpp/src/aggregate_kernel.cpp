#include "tide/aggregate.h"
#include <algorithm>
#include <stdexcept>

namespace tide {
std::vector<AggregateResult> AggregateKernel::batch(const NodeWeights& w, const std::vector<AggregateInput>& requests) const {
  std::vector<AggregateResult> result;
  for (const auto& request : requests) result.push_back(step(w, request));
  return result;
}
namespace {
class SourceAggregate final : public AggregateKernel {
  std::string kind_;

  Tensor coefficients(const NodeWeights& w, const AggregateInput& request) const {
    if (kind_ == "sum" || kind_ == "mean") return {};
    std::vector<Tensor> values;
    const auto prefix = kind_ == "weighted_mean" ? "agg_mass_" : "agg_logit_";
    if (kind_ == "all_softmax") {
      for (Index slot = 0; slot < request.slots; ++slot) values.push_back(w.extra.at(prefix + std::to_string(slot)));
    } else {
      for (const auto& source : request.sources) values.push_back(w.extra.at(prefix + std::to_string(source.slot)));
    }
    auto coefficients = at::stack(values);
    if (kind_ == "weighted_mean") {
      coefficients = at::softplus(coefficients);
      auto denominator = coefficients.sum();
      if (!(denominator.item<double>() > 0)) throw std::invalid_argument("Aggregate weighted mean has zero mass");
      return coefficients / denominator;
    }
    coefficients = at::softmax(coefficients, 0);
    if (kind_ != "all_softmax") return coefficients;
    values.clear();
    for (const auto& source : request.sources) values.push_back(coefficients[source.slot]);
    return at::stack(values);
  }
  AggregateResult combine(const NodeWeights& w, const AggregateInput& request, const std::vector<Tensor>& values) const {
    auto coe = coefficients(w, request);
    AggregateResult result;
    for (size_t i = 0; i < values.size(); ++i) {
      Tensor value = kind_ == "sum" ? values[i] : kind_ == "mean" ? values[i] / static_cast<double>(values.size()) : values[i] * coe[i];
      result.value = result.value.defined() ? result.value + value : value;
      result.contributions.push_back({request.sources[i].slot, value});
    }
    std::sort(result.contributions.begin(), result.contributions.end(), [](const auto& a, const auto& b) { return a.slot < b.slot; });
    return result;
  }
 public:
  explicit SourceAggregate(std::string kind) : kind_(std::move(kind)) {
    if (kind_ != "sum" && kind_ != "mean" && kind_ != "weighted_mean" && kind_ != "active_softmax" && kind_ != "all_softmax")
      throw std::invalid_argument("unknown Aggregate profile");
  }
  AggregateResult step(const NodeWeights& w, const AggregateInput& request) const override {
    std::vector<Tensor> values;
    for (const auto& source : request.sources) values.push_back(source.atom->value * source.scale);
    return combine(w, request, values);
  }
  std::vector<AggregateResult> batch(const NodeWeights& w, const std::vector<AggregateInput>& requests) const override {
    std::map<std::vector<Index>, std::vector<size_t>> groups;
    for (size_t i = 0; i < requests.size(); ++i) {
      std::vector<Index> signature;
      for (const auto& source : requests[i].sources) signature.push_back(source.slot);
      groups[signature].push_back(i);
    }
    std::vector<AggregateResult> result(requests.size());
    for (const auto& [signature, rows] : groups) {
      const auto& request = requests[rows[0]];
      std::vector<Tensor> values;
      for (size_t j = 0; j < signature.size(); ++j) {
        std::vector<Tensor> atoms, scales;
        for (auto i : rows) { atoms.push_back(requests[i].sources[j].atom->value); scales.push_back(requests[i].sources[j].scale); }
        values.push_back(at::stack(atoms) * at::stack(scales).unsqueeze(-1));
      }
      auto packed = combine(w, request, values);
      for (size_t row = 0; row < rows.size(); ++row) {
        auto& item = result[rows[row]]; item.value = packed.value[row];
        for (const auto& source : packed.contributions) item.contributions.push_back({source.slot, source.value[row]});
      }
    }
    return result;
  }
  bool joint_batch() const override { return true; }
  void validate_weights(const NodeWeights& w, Index slots) const override {
    std::string prefix = kind_ == "weighted_mean" ? "agg_mass_" :
                         kind_ == "active_softmax" || kind_ == "all_softmax" ? "agg_logit_" : "";
    if (prefix.empty()) return;
    Index count = 0;
    for (const auto& [name, value] : w.extra) if (name.compare(0, prefix.size(), prefix) == 0) ++count;
    if (count != slots) throw std::invalid_argument("Aggregate parameter slot domain mismatch");
    for (Index slot = 0; slot < slots; ++slot) {
      auto it = w.extra.find(prefix + std::to_string(slot));
      if (it == w.extra.end() || it->second.dim() != 0) throw std::invalid_argument("Aggregate parameter slot domain mismatch");
    }
  }
};
}  // namespace
std::shared_ptr<const AggregateKernel> make_aggregate_kernel(const Node& node) {
  return std::make_shared<SourceAggregate>(node.aggregation);
}
}  // namespace tide
