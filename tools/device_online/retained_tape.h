#pragma once
#include "graph_vjp.h"
#include <memory>

namespace tide::device_online {
// Own all values needed by a later backward, including parameter versions.
// CPU work copies only static graph metadata; dynamic tape copies stay on NPU.
struct RetainedTape {
  std::shared_ptr<const Graph> graph;
  ReverseTape tape;
  int64_t tensor_bytes=0;
};
RetainedTape retain_reverse_tape(const ReverseTape&,int64_t tensor_budget_bytes);
// Add a later window's boundary adjoints to the earlier window's own roots.
// The actual pending-message match and every connection decision are device work.
GraphCotangents append_window_bridge(CannProgram&,const ReverseTape& earlier,
    const GraphCotangents& local,const ReverseTape& later,const GraphVjp& later_gradient,
    const at::Tensor& error,int64_t tensor_budget_bytes);
} // namespace tide::device_online
