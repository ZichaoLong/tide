#include "tide/token_window.h"
#include "tide/counters.h"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace tide {
std::vector<External> token_inputs(const std::vector<Output>& outputs, const Continuation& q,
                                  Index layers, Index stop, Index body_cut, Index port) {
  if (layers < 1 || q.batch_size < 1 || q.cut < 0 || stop < q.cut || port < 0
      || stop > std::numeric_limits<Index>::max()/layers || body_cut != stop*layers)
    throw std::invalid_argument("invalid or unsealed token window");
  std::vector<External> result;
  std::set<Owner> seen;
  std::set<Index> tokens;
  for (const auto& o : outputs) {
    if (o.port < 0) throw std::invalid_argument("invalid body output port");
    if (o.port != port) continue;
    if (o.batch < 0 || o.batch >= q.batch_size || o.time < q.cut*layers || o.time >= body_cut
        || !seen.emplace(o.batch, o.time).second)
      throw std::invalid_argument("invalid or duplicate body output coordinate");
    const auto token = o.time/layers, phase = o.time%layers;
    tokens.insert(token); result.push_back({o.batch, phase, 0, token, o.value});
  }
  if (tokens.size() != static_cast<size_t>(stop-q.cut))
    throw std::invalid_argument("original Pronounce requires a nonempty global token window");
  std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
    return std::tie(a.time, a.batch, a.port) < std::tie(b.time, b.batch, b.port);
  });
  std::map<Owner, Index> positions;
  for (auto& x : result) {
    const Owner owner{x.batch, x.port};
    auto found = positions.find(owner);
    if (found == positions.end()) {
      const auto last = q.ledger.find(owner);
      if (last != q.ledger.end() && (last->second.first < 0 || last->second.second < 0 || last->second.second >= q.cut))
        throw std::invalid_argument("invalid token input ledger");
      found = positions.emplace(owner, last == q.ledger.end() ? -1 : last->second.first).first;
    }
    x.position = found->second == -1 ? 0 : increment(found->second);
    found->second = x.position;
  }
  return result;
}
}  // namespace tide
