#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace tide {
using Tensor = at::Tensor;
using Index = int64_t;
using Owner = std::pair<Index, Index>;
struct Edge { Index source, target, delay; };
struct Node {
  Index region;
  bool clear = false, identity = false;
  std::string memory = "ema", full = "tanh";
  Index query_heads = 1, kv_heads = 1, window = 0;
};
struct Region { Index budget; bool observe_all = true, count_priority = true; };
struct Adjacency { std::vector<Index> offsets, edges; };
struct Graph {
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::vector<Region> regions;
  std::vector<Index> inputs, outputs;
  Adjacency csr, csc, output_index;
  std::string identity;
  void compile();
  std::vector<Index> topological_order() const;
};
struct State {
  Tensor value;
  Index last_time = -1, observations = 0;
  std::map<std::string, Tensor> slots;
};
struct External { Index batch, port, position, time; Tensor value; };
struct Atom {
  Index batch, node, time, kind, source, position;
  Tensor value;
  auto key() const { return std::tie(batch, node, time, kind, source, position); }
};
struct Continuation {
  std::string identity;
  Index batch_size = 1, cut = 0;
  std::map<Owner, State> states;
  std::map<Owner, std::map<Index, Index>> history;
  std::vector<Atom> pending;
  std::map<Owner, Owner> ledger;
};
class StateKernel;
struct NodeWeights {
  Tensor decay, weight, bias, read;
  std::map<std::string, Tensor> extra;
  std::shared_ptr<const StateKernel> kernel;
  std::string full_kind = "tanh";
};
struct Model {
  std::vector<NodeWeights> nodes;
  std::vector<Tensor> input_scale, agg_scale, edge_scale, output_scale;
  Index width() const { return nodes.at(0).bias.numel(); }
};
struct Event {
  Index batch, node, time;
  std::vector<Atom> fiber;
  Tensor content, proposal, descriptor, control, comparison, next, full;
  State old, proposed_state, comparison_state, next_state;
  bool active = false;
  std::map<Index, Index> history;
};
struct Output { Index batch, time, port; Tensor value; };
struct Result {
  Continuation continuation;
  std::vector<Event> trace;
  std::vector<Output> outputs;
  std::vector<Atom> messages;
  std::map<std::string, Index> stats;
};
struct Options {
  Index workers = 1;
  bool packed = false, trace = true;
  std::string mode = "hard";
  double zeta = 1.0;
  bool prefill = true;
  Index max_events = 1000000;
};
}  // namespace tide
