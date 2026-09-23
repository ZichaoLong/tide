#include "fiber_rows.h"
#include "tide/operator_work.h"
#include <algorithm>
#include <stdexcept>

namespace tide {
FiberRows fiber_rows(const ContentViews& views) {
  FiberRows result;
  if (views.empty()) throw std::invalid_argument("fiber attention requires events");
  const auto shared = views.front().source_batch;
  bool common = bool(shared);
  for (const auto& view : views) common = common && view.source_batch == shared;
  std::vector<Tensor> rows;
  std::vector<Index> indices;
  Index scaled = 0;
  for (const auto& view : views) {
    if (view.sources.empty()) throw std::invalid_argument("fiber attention requires complete source rows");
    std::vector<size_t> order;
    for (size_t i = 0; i < view.sources.size(); ++i) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return view.sources[a].slot < view.sources[b].slot; });
    for (auto i : order) {
      const auto& source = view.sources[i];
      result.slots.push_back(source.slot);
      if (common) indices.push_back(shared->offsets.at(view.source_row)+i);
      else {
        rows.push_back(source.atom.value*source.scale);
        scaled += source.atom.value.numel();
      }
    }
    result.offsets.push_back(result.slots.size());
  }
  if (common) {
    bool identity = indices.size() == static_cast<size_t>(shared->values.size(0));
    for (size_t i = 0; identity && i < indices.size(); ++i) identity = indices[i] == static_cast<Index>(i);
    result.values = identity ? shared->values : shared->values.index_select(0,
      at::tensor(indices, shared->values.options().dtype(at::kLong)));
    if (work::enabled()) work::add(work::FiberReusedElements, result.values.numel());
  } else {
    result.values = at::stack(rows);
    if (work::enabled()) work::add(work::FiberScaleElements, scaled);
  }
  return result;
}
} // namespace tide
