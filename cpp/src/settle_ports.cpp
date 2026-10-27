// Core-5 Settle §2.4/§10: occurrence schedules and identity embedding.
// No reference recursion is used for decoding or native planning.
#include "tide/settle.h"
#include <limits>
#include <stdexcept>

namespace tide {
std::vector<External> SettleGraph::port_external(
    const Tensor& values, const std::vector<Index>& positions,
    Index port, Index batch, Index start) const {
  const auto limit = std::numeric_limits<Index>::max();
  if (binding_ != "ports") throw std::invalid_argument("port schedule requires ports binding");
  if (!values.defined() || values.dim() != 2 || values.size(0) != static_cast<Index>(positions.size()) ||
      port < 0 || port >= static_cast<Index>(body_.inputs.size()) || batch < 0 ||
      start < 0 || values.size(0) > limit-start)
    throw std::invalid_argument("invalid SettleGraph port schedule shape/owner/occurrence");
  const auto r = rank(body_.inputs[port]);
  std::vector<External> result;
  Index previous = -1;
  for (Index k = 0; k < static_cast<Index>(positions.size()); ++k) {
    const auto t = positions[k];
    if (t < 0 || t <= previous || t > (limit-r)/stride_)
      throw std::invalid_argument("port schedule requires increasing representable positions");
    result.push_back({batch, port, start+k, stride_*t+r, values[k]});
    previous = t;
  }
  return result;
}

void SettleGraph::validate_alignment(const Continuation& q, const std::vector<External>& external) const {
  auto aligned = [&](Index time, Index node) {
    if (node < 0 || node >= static_cast<Index>(body_.nodes.size()) ||
        time < 0 || time % stride_ != rank(node))
      throw std::invalid_argument("record violates SettleGraph time code");
  };
  auto input_node = [&](Index port) {
    if (port < 0 || port >= static_cast<Index>(body_.inputs.size()))
      throw std::invalid_argument("invalid SettleGraph input port");
    return body_.inputs[port];
  };
  for (const auto& row : external) aligned(row.time, input_node(row.port));
  for (const auto& [owner, last] : q.ledger) aligned(last.second, input_node(owner.second));
  for (const auto& a : q.pending) {
    if (a.source < 0 || a.source >= static_cast<Index>(body_.edges.size()))
      throw std::invalid_argument("invalid SettleGraph pending edge");
    aligned(a.time, a.node); aligned(a.position, body_.edges[a.source].source);
  }
  for (const auto& [owner, state] : q.states)
    if (state.last_time != -1) aligned(state.last_time, owner.second);
  for (const auto& [owner, history] : q.history) {
    if (owner.second < 0 || owner.second >= static_cast<Index>(ranks_.size()))
      throw std::invalid_argument("invalid SettleGraph history owner");
    if (history.last_time != -1 && (history.last_time < 0 || history.last_time % stride_ != ranks_[owner.second]))
      throw std::invalid_argument("history violates SettleGraph time code");
  }
}

Result SettleExecutor::run_ports(const Continuation& initial, const std::vector<External>& external,
                                Index stop, Index sealed_until) {
  if (spec_.binding() != "ports") throw std::invalid_argument("run_ports requires ports binding");
  spec_.validate_alignment(initial, external);
  if (greedy_) return greedy_->run(initial, external, stop, sealed_until);
  return frontier_ ? frontier_->run(initial, external, stop, sealed_until)
                   : streaming_->run(initial, external, stop, sealed_until);
}
}  // namespace tide
