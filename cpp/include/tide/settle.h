#pragma once
#include "tide/frontier.h"
#include "tide/stream.h"

namespace tide {
// Rank-aligned SettleGraph, with one broadcast input occurrence per sample and
// position and a sum readout boundary. Body identities/slots and Tensor owners
// survive encoding; boundary nodes/scales have no trainable parameters.
class SettleGraph {
 public:
  SettleGraph(Graph body, std::vector<Index> region_ranks);
  const Graph& graph() const { return body_; }
  const Graph& encoded_graph() const { return encoded_; }
  const std::vector<Index>& ranks() const { return ranks_; }
  Index rank(Index node) const;
  Index output_rank() const { return stride_ - 1; }
  Index stride() const { return stride_; }
  Model embed_model(const Model&) const;
  Continuation embed_initial(const Continuation&) const;
  std::vector<External> external(const Tensor& values, Index start_position = 0,
                                 bool encoded = true) const;
  // Complete-position observation only. Keep the encoded continuation to run
  // subsequent windows; a projected value is not a general inverse encoding.
  Result project(const Result&) const;
 private:
  Graph body_, encoded_;
  std::vector<Index> ranks_;
  Index stride_;
};

// Standalone C++ entry: owns the native encoding and selected native scheduler.
// Both input and returned continuation are in the encoded graph namespace.
class SettleExecutor {
 public:
  SettleExecutor(SettleGraph, Model, Options = {}, std::string algorithm = "frontier");
  Result run(const Continuation& encoded_initial, const Tensor& values);
 private:
  SettleGraph spec_;
  std::unique_ptr<Frontier> frontier_;
  std::unique_ptr<Streaming> streaming_;
};
}  // namespace tide
