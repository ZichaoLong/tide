#pragma once
#include "tide/stream.h"

namespace tide {
struct AdvanceResult {
  Index cut;
  std::vector<Event> trace;
  std::vector<Output> outputs;
  std::vector<Atom> messages;
  std::map<std::string, Index> stats;
};
// Owns state/queues, borrows the immutable program and worker pool. The engine
// must outlive this cursor. Imports/exports clone state tensors. Advance result
// tensors and supplied inputs are immutable until their consumers have finished.
class StreamingCursor {
 public:
  StreamingCursor(Streaming&, Continuation);
  AdvanceResult advance(const std::vector<External>&, Index stop, Index seal);
  Continuation snapshot() const;  // Explicit state/pending/graph-identity copy and differentiable tensor clones.
  void detach();                 // Explicit O(all state + pending) truncation.
  Index cut() const;
  bool failed() const;
 private:
  void healthy() const;
  Streaming& engine_;
  Continuation state_;
  EventQueue queue_;
  bool failed_ = false;
  mutable std::mutex mutex_;
};
void export_pending(Continuation&, const EventQueue&);
}  // namespace tide
