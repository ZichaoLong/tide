#include "tide/cursor.h"
#include "tide/ops.h"
#include <algorithm>
#include <stdexcept>

namespace tide {
namespace {
void clone_tensors(Continuation& q) {
  for (auto& [owner, state] : q.states) {
    state.value = state.value.clone();
    for (auto& [name, value] : state.slots) value = value.clone();
  }
  for (auto& atom : q.pending) atom.value = atom.value.clone();
}
}  // namespace
void export_pending(Continuation& q, const EventQueue& queue) {
  q.pending.clear();
  for (const auto& [time, atoms] : queue) q.pending.insert(q.pending.end(), atoms.begin(), atoms.end());
  std::sort(q.pending.begin(), q.pending.end(), [](const auto& a, const auto& b) { return a.key() < b.key(); });
}
StreamingCursor::StreamingCursor(Streaming& engine, Continuation initial)
    : engine_(engine), state_(std::move(initial)) {
  validate_window(engine_.graph_, engine_.model_, state_, {}, state_.cut, state_.cut);
  clone_tensors(state_);
  for (const auto& atom : state_.pending) queue_[atom.time].push_back(atom);
  state_.pending.clear();
}
void StreamingCursor::healthy() const {
  if (failed_) throw std::runtime_error("streaming cursor failed during execution; restore an earlier snapshot in a new cursor");
}
AdvanceResult StreamingCursor::advance(const std::vector<External>& external, Index stop, Index seal) {
  std::lock_guard<std::mutex> cursor_lock(mutex_);
  healthy();
  // Invalid inputs never consume state/ledger and may be corrected and retried.
  auto input = validate_external(engine_.graph_, engine_.model_, state_, external, stop, seal);
  std::lock_guard<std::mutex> engine_lock(engine_.run_mutex_);
  try {
    for (const auto& atom : input.atoms) queue_[atom.time].push_back(atom);
    for (const auto& [owner, last] : input.ledger_updates) state_.ledger[owner] = last;
    auto result = engine_.execute(state_, queue_, stop);
    result.stats["external_records"] = input.atoms.size();
    result.stats["cached_states"] = state_.states.size();
    result.stats["queued_times"] = queue_.size();
    return {state_.cut, std::move(result.trace), std::move(result.outputs), std::move(result.messages), std::move(result.stats)};
  } catch (...) {
    // Avoid an O(all state) rollback copy on every successful sparse advance.
    failed_ = true;
    throw;
  }
}
Continuation StreamingCursor::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_); healthy();
  auto q = state_; export_pending(q, queue_);
  clone_tensors(q);
  return q;
}
void StreamingCursor::detach() {
  std::lock_guard<std::mutex> lock(mutex_); healthy();
  for (auto& [owner, state] : state_.states) {
    state.value = state.value.detach();
    for (auto& [name, value] : state.slots) value = value.detach();
  }
  for (auto& [time, atoms] : queue_) for (auto& atom : atoms) atom.value = atom.value.detach();
}
Index StreamingCursor::cut() const { std::lock_guard<std::mutex> lock(mutex_); healthy(); return state_.cut; }
bool StreamingCursor::failed() const { std::lock_guard<std::mutex> lock(mutex_); return failed_; }
}  // namespace tide
