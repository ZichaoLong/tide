#include "tide/region.h"
#include "tide/counters.h"
#include <algorithm>
#include <stdexcept>

namespace tide {
namespace {
class LHSelector final : public RegionKernel {
 public:
  History initial(const RegionWeights&, const RegionLayout&, const Tensor&) const override {
    return {-1, {}, {{"selected", {}}, {"affected", {}}}, {}};
  }
  Selection step(const RegionWeights&, const RegionInput& r) const override {
    const auto& selected = r.history.node_maps.at("selected");
    const auto& affected = r.history.node_maps.at("affected");
    auto count = [](const auto& counts, Index node) { auto it = counts.find(node); return it == counts.end() ? 0 : it->second; };
    std::vector<std::tuple<Index, Index, double, Index>> order;
    std::vector<Tensor> descriptors;
    for (const auto& c : r.candidates) {
      order.emplace_back(count(selected, c.node), -count(affected, c.node), -c.descriptor.item<double>(), c.node);
      descriptors.push_back(c.descriptor);
    }
    std::sort(order.begin(), order.end());
    Selection out; out.history = r.history; out.history.last_time = r.time;
    for (Index i = 0; i < std::min<Index>(r.layout.spec.budget, order.size()); ++i) out.active.insert(std::get<3>(order[i]));
    auto controls = at::softmax(at::stack(descriptors), 0).to(r.payload_options);
    for (size_t i = 0; i < r.candidates.size(); ++i) {
      auto node = r.candidates[i].node; out.controls[node] = controls[i];
      out.history.node_maps["affected"][node] = increment(count(affected, node));
      if (out.active.count(node)) out.history.node_maps["selected"][node] = increment(count(selected, node));
    }
    return out;
  }
  void validate_weights(const RegionWeights& w, const RegionLayout& layout) const override {
    if (!w.extra.empty() || !layout.spec.count_priority) throw std::invalid_argument("LH selector requires count priority and no parameters");
  }
  void validate_history(const History& h, const RegionLayout&) const override {
    if (!h.scalars.empty() || !h.tensors.empty() || h.node_maps.size() != 2
        || !h.node_maps.count("selected") || !h.node_maps.count("affected"))
      throw std::invalid_argument("invalid LH selector history layout");
    for (const auto& [name, counts] : h.node_maps) for (const auto& [node, count] : counts)
      if (count < 0) throw std::invalid_argument("invalid LH selector history value");
  }
};
}  // namespace
std::shared_ptr<const RegionKernel> make_lh_selector() { return std::make_shared<LHSelector>(); }
}  // namespace tide
