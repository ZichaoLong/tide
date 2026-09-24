#include "tide/operator_work.h"
#include "tide/aggregate.h"
#include "tide/isolated_aggregate.h"
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

  Tensor coefficients(const NodeWeights& w, const AggregateInput& request, Index rows = 0) const {
    if (kind_ == "sum" || kind_ == "mean") return {};
    std::vector<Tensor> values;
    const auto prefix = kind_ == "weighted_mean" ? "agg_mass_" : "agg_logit_";
    if (kind_ == "all_softmax") {
      for (Index slot = 0; slot < request.slots; ++slot) values.push_back(w.extra.at(prefix + std::to_string(slot)));
    } else {
      for (const auto& source : request.sources) values.push_back(w.extra.at(prefix + std::to_string(source.slot)));
    }
    auto coefficients = at::stack(values);
    // Softmax keeps an event axis: apply each row's Jacobian before accumulating
    // shared-owner gradients. A shared vector instead aggregates cotangents first
    // and can change near-zero float gradients enough to alter AdamW updates.
    if (rows) coefficients = coefficients.unsqueeze(0).expand({rows, coefficients.numel()});
    if (kind_ == "weighted_mean") {
      coefficients = at::softplus(coefficients);
      auto denominator = coefficients.sum();
      if (!(denominator.item<double>() > 0)) throw std::invalid_argument("Aggregate weighted mean has zero mass");
      return coefficients / denominator;
    }
    coefficients = at::softmax(coefficients, rows ? 1 : 0);
    if (kind_ != "all_softmax") return coefficients;
    values.clear();
    for (const auto& source : request.sources) values.push_back(rows ? coefficients.select(1, source.slot) : coefficients[source.slot]);
    return at::stack(values, rows ? 1 : 0);
  }
  AggregateResult combine(const NodeWeights& w, const AggregateInput& request, const std::vector<Tensor>& values) const {
    if (work::enabled()) {
      work::add(work::AggregateScaleElements, values.size()*values.at(0).numel());
      work::add(work::AggregateAddElements, (values.size()-1)*values.at(0).numel());
    }
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
    for (const auto& source : request.sources) values.push_back(source.atom.value * source.scale);
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
        for (auto i : rows) { atoms.push_back(requests[i].sources[j].atom.value); scales.push_back(requests[i].sources[j].scale); }
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
  bool batched_autograd() const override { return true; }
  std::vector<AggregateResult> batch_grad(const NodeWeights& w, const std::vector<AggregateInput>& requests) const override {
    std::map<std::vector<Index>, std::vector<size_t>> groups;
    for (size_t i = 0; i < requests.size(); ++i) {
      std::vector<Index> signature;
      for (const auto& source : requests[i].sources) signature.push_back(source.slot);
      groups[signature].push_back(i);
    }
    std::vector<AggregateResult> result(requests.size());
    for (const auto& [signature, rows] : groups) {
      Tensor coe;
      if (kind_ == "weighted_mean") {
        // Keep normalization VJPs per event: combining their cotangents before
        // division changes cancellation near zero, which AdamW can amplify.
        std::vector<Tensor> coefficients_by_event;
        for (auto i : rows) coefficients_by_event.push_back(coefficients(w, requests[i]));
        coe = at::stack(coefficients_by_event);
      } else coe = coefficients(w, requests[rows[0]], rows.size());
      std::vector<Tensor> atoms, scales;
      for (auto i : rows) for (const auto& source : requests[i].sources) {
        atoms.push_back(source.atom.value); scales.push_back(source.scale);
      }
      if (!coe.defined()) coe = at::empty({0}, atoms[0].options());
      auto values = isolated_aggregate(atoms, scales, coe, signature.size(), kind_ == "mean");
      if (work::enabled()) {
        work::add(work::AggregateScaleElements, atoms.size()*atoms[0].numel());
        work::add(work::AggregateAddElements, (atoms.size()-rows.size())*atoms[0].numel());
      }
      for (size_t i = 0; i < rows.size(); ++i) {
        auto& item = result[rows[i]];
        item.value = values[i*(signature.size()+1)];
        for (size_t j = 0; j < signature.size(); ++j)
          item.contributions.push_back({signature[j], values[i*(signature.size()+1)+1+j]});
        std::sort(item.contributions.begin(), item.contributions.end(), [](const auto& a, const auto& b) { return a.slot < b.slot; });
      }
    }
    return result;
  }
  bool joint_sources() const override { return true; }
  AggregateBatch source_batch(const NodeWeights& w, const std::vector<AggregateInput>& requests) const override {
    AggregateBatch result;
    if (requests.empty()) return result;
    auto sources = std::make_shared<SourceBatch>();
    std::vector<Tensor> atoms, scales;
    std::map<std::vector<Index>, std::vector<Index>> groups;
    for (size_t i = 0; i < requests.size(); ++i) {
      std::vector<Index> signature;
      for (const auto& source : requests[i].sources) {
        signature.push_back(source.slot); sources->slots.push_back(source.slot);
        atoms.push_back(source.atom.value); scales.push_back(source.scale);
      }
      sources->offsets.push_back(atoms.size()); groups[signature].push_back(i);
    }
    sources->values = at::stack(atoms)*at::stack(scales).unsqueeze(-1);
    result.sources = sources; result.events.resize(requests.size());
    const auto width = sources->values.size(1);
    // Fresh private numeric storage; evaluate_aggregate invokes this under no-grad.
    result.contents = at::empty({static_cast<Index>(requests.size()), width}, sources->values.options());
    for (const auto& [signature, rows] : groups) {
      std::vector<Index> indices;
      for (auto i : rows)
        for (auto j = sources->offsets[i]; j < sources->offsets[i+1]; ++j) indices.push_back(j);
      auto group = sources->values.index_select(0, at::tensor(indices, sources->values.options().dtype(at::kLong)))
                     .reshape({static_cast<Index>(rows.size()), static_cast<Index>(signature.size()), width});
      std::vector<Tensor> values;
      for (size_t j = 0; j < signature.size(); ++j) values.push_back(group.select(1, j));
      auto combined = combine(w, requests[rows[0]], values);
      result.contents.index_copy_(0, at::tensor(rows, sources->values.options().dtype(at::kLong)), combined.value);
      for (size_t row = 0; row < rows.size(); ++row)
        for (const auto& source : combined.contributions)
          result.events[rows[row]].contributions.push_back({source.slot, source.value[row]});
    }
    for (size_t i = 0; i < requests.size(); ++i) result.events[i].value = result.contents[i];
    return result;
  }
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
