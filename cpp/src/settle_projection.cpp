#include "tide/settle.h"
#include <algorithm>
#include <stdexcept>

namespace tide {
Result SettleGraph::project(const Result& encoded) const {
  const auto& initial = encoded.continuation;
  if (initial.identity != encoded_.identity || initial.batch_size < 1)
    throw std::invalid_argument("SettleGraph projection identity/batch mismatch");
  if (initial.cut < 0 || initial.cut % stride_ || !initial.pending.empty())
    throw std::invalid_argument("SettleGraph projection requires a complete position cut");
  const Index n = body_.nodes.size(), r = body_.regions.size(), edges = body_.edges.size();
  Result result;
  result.outputs = encoded.outputs; result.stats = encoded.stats;
  auto& q = result.continuation;
  q.identity = body_.identity; q.batch_size = initial.batch_size; q.cut = initial.cut;
  for (const auto& [owner, value] : initial.states) if (owner.second < n) q.states.emplace(owner, value);
  for (const auto& [owner, value] : initial.history) if (owner.second < r) q.history.emplace(owner, value);
  // Use the actual occurrence ledger, never cut/token time to invent arrivals.
  for (const auto& [owner, last] : initial.ledger) {
    if (owner.first < 0 || owner.first >= q.batch_size || owner.second != 0 ||
        last.first < 0 || last.second < 0 || last.second >= q.cut ||
        last.second % stride_ || last.second / stride_ != last.first)
      throw std::invalid_argument("invalid SettleGraph boundary occurrence ledger");
    for (Index p = 0; p < static_cast<Index>(body_.inputs.size()); ++p)
      q.ledger[{owner.first, p}] = {last.first, last.second + rank(body_.inputs[p])};
  }
  for (const auto& event : encoded.trace) {
    if (event.node >= n) continue;
    auto restored = event;
    for (auto& atom : restored.fiber) {
      if (atom.kind != 1 || atom.source < 0)
        throw std::invalid_argument("invalid SettleGraph encoded body fiber");
      if (atom.source < edges) continue;
      const auto port = atom.source - edges;
      if (port >= static_cast<Index>(body_.inputs.size()) || body_.inputs[port] != atom.node ||
          atom.position < 0 || atom.position % stride_ || atom.time - atom.position != rank(atom.node))
        throw std::invalid_argument("invalid SettleGraph body input adapter edge/clock");
      atom.kind = 0; atom.source = port; atom.position /= stride_;
    }
    std::sort(restored.fiber.begin(), restored.fiber.end(), [](const auto& a, const auto& b) { return a.key() < b.key(); });
    result.trace.push_back(std::move(restored));
  }
  for (const auto& atom : encoded.messages) if (atom.source < edges) result.messages.push_back(atom);
  return result;
}
}  // namespace tide
