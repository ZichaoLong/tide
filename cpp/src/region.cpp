#include "tide/region.h"
#include "tide/counters.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tide {
Index RegionLayout::slot(Index node) const {
  auto it = std::lower_bound(members.begin(), members.end(), node);
  if (it == members.end() || *it != node) throw std::invalid_argument("node is outside region membership");
  return it - members.begin();
}
RegionLayout region_layout(const Graph& g, Index region) {
  auto begin = g.region_index.offsets[region], end = g.region_index.offsets[region+1];
  return {g.regions[region], c10::ArrayRef<Index>(g.region_index.edges).slice(begin, end-begin)};
}
namespace {
class CountSelector final : public RegionKernel {
 public:
  explicit CountSelector(std::string profile) : profile_(std::move(profile)) {}
  History initial(const RegionWeights&, const RegionLayout&, const Tensor& ref) const override {
    History h; h.node_maps["selected"] = {};
    if (tensor_history()) h.tensors["memory"] = at::zeros({}, ref.options());
    return h;
  }
  Selection step(const RegionWeights& w, const RegionInput& r) const override {
    const auto& counts = r.history.node_maps.at("selected");
    std::vector<Tensor> scores;
    std::vector<std::tuple<Index, double, Index>> ranking;
    for (const auto& candidate : r.candidates) {
      const auto node = candidate.node;
      auto score = candidate.descriptor;
      if (tensor_history()) score = score + r.history.tensors.at("memory")*w.extra.at("bias")[r.layout.slot(node)];
      auto value = score.item<double>();
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite region selector score");
      auto count = counts.find(node);
      if (profile_ != "positive-v1" || value > 0)
        ranking.emplace_back(r.layout.spec.count_priority && count != counts.end() ? count->second : 0, -value, node);
      scores.push_back(score);
    }
    std::sort(ranking.begin(), ranking.end());
    Selection result; result.history = r.history; result.history.last_time = r.time;
    for (Index i = 0; i < std::min<Index>(r.layout.spec.budget, ranking.size()); ++i) {
      auto node = std::get<2>(ranking[i]); result.active.insert(node);
      auto& count = result.history.node_maps.at("selected")[node]; count = increment(count);
    }
    auto controls = at::softmax(at::stack(scores), 0).to(r.payload_options);
    for (size_t i = 0; i < scores.size(); ++i) result.controls.emplace(r.candidates[i].node, controls[i]);
    if (tensor_history()) {
      std::vector<Tensor> descriptors;
      for (const auto& candidate : r.candidates) descriptors.push_back(candidate.descriptor);
      result.history.tensors["memory"] = w.extra.at("alpha")*r.history.tensors.at("memory") + at::stack(descriptors).sum().to(r.payload_options);
    }
    return result;
  }
  void validate_weights(const RegionWeights& w, const RegionLayout& layout) const override {
    if (!tensor_history()) {
      if (!w.extra.empty()) throw std::invalid_argument("unexpected count-selector parameters");
      return;
    }
    if (w.extra.size() != 2 || !w.extra.count("alpha") || !w.extra.count("bias")
        || w.extra.at("alpha").dim() != 0 || w.extra.at("bias").sizes() != at::IntArrayRef{static_cast<Index>(layout.members.size())})
      throw std::invalid_argument("region parameter membership layout mismatch");
  }
  void validate_history(const History& h, const RegionLayout&) const override {
    if (!h.scalars.empty() || h.node_maps.size() != 1 || !h.node_maps.count("selected"))
      throw std::invalid_argument("invalid selected-count history layout");
    for (const auto& [node, count] : h.node_maps.at("selected"))
      if (count < 0) throw std::invalid_argument("invalid selected-count history value");
    if (tensor_history()) {
      if (h.tensors.size() != 1 || !h.tensors.count("memory") || h.tensors.at("memory").dim() != 0)
        throw std::invalid_argument("invalid tensor-selector history layout");
    } else if (!h.tensors.empty()) throw std::invalid_argument("unexpected count-selector history tensors");
  }
 private:
  bool tensor_history() const { return profile_ == "tensor-history-v1"; }
  std::string profile_;
};
}  // namespace
std::shared_ptr<const RegionKernel> make_lh_selector();
std::shared_ptr<const RegionKernel> make_region_kernel(const Region& spec) {
  if (spec.selector == "lh-count-affect-v1") return make_lh_selector();
  if (spec.selector == "count-v1" || spec.selector == "positive-v1" || spec.selector == "tensor-history-v1")
    return std::make_shared<CountSelector>(spec.selector);
  throw std::invalid_argument("unknown region selector profile: " + spec.selector);
}
}  // namespace tide
