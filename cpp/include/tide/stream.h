#pragma once
#include "tide/pool.h"
#include "tide/types.h"
#include <mutex>

namespace tide {
using EventQueue = std::map<Index, std::vector<Atom>>;
class StreamingCursor;
class Streaming {
 public:
  Streaming(Graph graph, Model model, Options options);
  Result run(const Continuation&, const std::vector<External>&, Index stop, Index seal);
  const Graph& graph() const { return graph_; }
 private:
  friend class StreamingCursor;
  Result execute(Continuation&, EventQueue&, Index stop);
  Graph graph_;
  Model model_;
  Options options_;
  NodePool pool_;
  std::mutex run_mutex_;  // One shared pool: calls on this engine are serialized.
};
}  // namespace tide
