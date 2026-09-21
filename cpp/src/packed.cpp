#include "tide/kernel.h"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tide {
void PackedSequence::validate() const {
  auto fail = [] { throw std::invalid_argument("invalid packed sequence metadata"); };
  if (!contents.defined() || contents.dim() != 2 || owners.empty() || offsets.size() != owners.size() + 1
      || offsets.front() != 0 || offsets.back() != contents.size(0)
      || times.size() != static_cast<size_t>(contents.size(0)) || fibers.size() != times.size()) fail();
  std::set<Owner> seen;
  for (size_t i = 0; i < owners.size(); ++i) {
    if (owners[i].first < 0 || owners[i].second < 0 || owners[i].second != owners.front().second
        || !seen.insert(owners[i]).second || offsets[i] < 0 || offsets[i] >= offsets[i + 1]
        || offsets[i + 1] > contents.size(0)) fail();
    for (Index j = offsets[i]; j < offsets[i + 1]; ++j)
      if (!fibers[j] || times[j] < 0 || (j > offsets[i] && times[j] <= times[j - 1])) fail();
  }
}
PackedStates StateKernel::packed_sequence(const NodeWeights& w, const std::vector<State>& old,
                                          const PackedSequence& batch) const {
  batch.validate();
  if (old.size() != batch.owners.size()) throw std::invalid_argument("packed initial-state count mismatch");
  PackedStates result;
  for (size_t i = 0; i < old.size(); ++i) {
    const auto a = batch.offsets[i], b = batch.offsets[i + 1];
    auto states = sequence(w, old[i], batch.contents.slice(0, a, b),
                           {batch.times.begin() + a, batch.times.begin() + b},
                           {batch.fibers.begin() + a, batch.fibers.begin() + b});
    result.states.insert(result.states.end(), states.begin(), states.end());
    ++result.calls; result.max_batch = 1; result.max_length = std::max(result.max_length, b - a);
  }
  return result;
}
}  // namespace tide
