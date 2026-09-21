#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
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
  std::string emission = "broadcast";
  Index emit_period = 1;
  std::vector<Index> emit_phases;
  std::string aggregation = "sum", readout = "linear-v1", next_state = "adopt-v1";
};
struct Region {
  Index budget;
  bool observe_all = true, count_priority = true;
  std::string read_mode = "proposal", selector = "count-v1";
};
struct Adjacency { std::vector<Index> offsets, edges; };
struct PortLayout {
  std::vector<Index> edge_source, edge_target, input, output;
};
struct PortBinding { Index kind, id; };  // 0: boundary port, 1: edge
struct PortIndex { std::vector<Index> offsets; std::vector<PortBinding> bindings; };
struct InputOrigin { Index edge, port, stride; };
struct Graph {
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::vector<Region> regions;
  std::vector<Index> inputs, outputs;
  Adjacency csr, csc, output_index, region_index;  // Region rows contain canonical node IDs.
  std::optional<PortLayout> layout;
  PortIndex incoming_ports, outgoing_ports;
  std::vector<InputOrigin> origins;
  std::vector<Index> origin_index;  // Physical edge -> origin record, or -1.
  std::string identity;
  void compile();
  std::vector<Index> topological_order() const;
};
struct State {
  Tensor value;
  Index last_time = -1, observations = 0;
  std::map<std::string, Tensor> slots;
};
struct History {
  Index last_time = -1;
  std::map<std::string, Index> scalars;
  std::map<std::string, std::map<Index, Index>> node_maps;
  std::map<std::string, Tensor> tensors;
};
struct External { Index batch, port, position, time; Tensor value; };
struct Atom {
  Index batch, node, time, kind, source, position;
  Tensor value;
  auto key() const { return std::tie(batch, node, time, kind, source, position); }
};
struct SourceInput { Index slot; Atom atom; Tensor scale; };
struct SlotValue { Index slot; Tensor value; };
// Metadata views live only for a synchronous kernel call, never in persistent
// State. Tensor handles may be retained by autograd; storage is read-only.
struct ContentView {
  Tensor value;
  c10::ArrayRef<SourceInput> sources;
  c10::ArrayRef<SlotValue> contributions;
  ContentView with_value(Tensor v) const { return {std::move(v), sources, contributions}; }
};
struct Continuation {
  std::string identity;
  Index batch_size = 1, cut = 0;
  std::map<Owner, State> states;
  std::map<Owner, History> history;
  std::vector<Atom> pending;
  std::map<Owner, Owner> ledger;
};
class StateKernel;
class FullKernel;
class AggregateKernel;
class ReadKernel;
class NextKernel;
class RegionKernel;
struct RegionWeights {
  std::map<std::string, Tensor> extra;
  std::shared_ptr<const RegionKernel> kernel;
};
struct NodeWeights {
  Tensor decay, weight, bias, read;
  std::map<std::string, Tensor> extra;
  std::shared_ptr<const StateKernel> kernel;
  std::string full_kind = "tanh";
  std::shared_ptr<const FullKernel> full_kernel;
  std::shared_ptr<const AggregateKernel> aggregate_kernel;
  std::shared_ptr<const ReadKernel> read_kernel;
  std::shared_ptr<const NextKernel> next_kernel;
};
struct Model {
  std::vector<NodeWeights> nodes;
  std::vector<RegionWeights> regions;
  std::vector<Tensor> input_scale, agg_scale, edge_scale, output_scale;
  Index width() const { return nodes.at(0).bias.numel(); }
};
struct Event {
  Index batch, node, time;
  std::vector<Atom> fiber;
  Tensor content, proposal, descriptor, control, comparison, next, full;
  State old, proposed_state, comparison_state, next_state;
  bool active = false;
  std::vector<SlotValue> emitted;
  std::vector<SlotValue> contributions;
  std::vector<SourceInput> sources;
  ContentView local_content() const { return {content, sources, contributions}; }
  History history;
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
