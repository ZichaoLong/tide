#pragma once
#include "tide/frontier.h"
#include "tide/greedy.h"
#include "tide/stream.h"

namespace tide {
// Core-5 Settle §2/§10: ports is the identity TimedDAG embedding.
// broadcast_sum retains the explicit §6.4 convenience binding.
// Mapping and independent tests: docs/settle-embedding.md.
class SettleGraph {
 public:
  SettleGraph(Graph body, std::vector<Index> region_ranks,
              std::string binding = "broadcast_sum", Index time_stride = 0);
  const Graph& graph() const { return body_; }
  const Graph& encoded_graph() const { return encoded_; }
  const std::vector<Index>& ranks() const { return ranks_; }
  Index rank(Index node) const;
  Index output_rank() const { return output_rank_; }
  Index stride() const { return stride_; }
  const std::string& binding() const { return binding_; }
  Model embed_model(const Model&) const;
  Continuation embed_initial(const Continuation&) const;
  std::vector<External> external(const Tensor& values, Index start_position = 0,
                                 bool encoded = true) const;
  std::vector<External> port_external(const Tensor&, const std::vector<Index>& positions,
                                      Index port, Index batch = 0, Index start_occurrence = 0) const;
  void validate_alignment(const Continuation&, const std::vector<External>&) const;
  // ports: identity at every cut. broadcast_sum: complete-position observation
  // only; keep the encoded continuation for subsequent windows.
  Result project(const Result&) const;
 private:
  Graph body_, encoded_;
  std::vector<Index> ranks_;
  Index stride_, output_rank_;
  std::string binding_;
};

// Standalone C++ entry: owns the native encoding and selected native scheduler.
// Both input and returned continuation are in the encoded graph namespace.
class SettleExecutor {
 public:
  SettleExecutor(SettleGraph, Model, Options = {}, std::string algorithm = "frontier");
  Result run(const Continuation& encoded_initial, const Tensor& values);
  Result run_ports(const Continuation&, const std::vector<External>&, Index stop, Index sealed_until);
 private:
  SettleGraph spec_;
  std::unique_ptr<Frontier> frontier_;
  std::unique_ptr<Greedy> greedy_;
  std::unique_ptr<Streaming> streaming_;
};
}  // namespace tide
