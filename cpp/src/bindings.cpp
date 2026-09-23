// Python is only an adapter; core library sources contain no Python headers.
#include "tide/stream.h"
#include "tide/dense.h"
#include "tide/ops.h"
#include "tide/frontier.h"
#include "tide/specialized.h"
#include "tide/cursor.h"
#include "tide/lazy_add.h"
#include "tide/fiber_attention.h"
#include "tide/token_window.h"
#include "tide/parameters.h"
#include "tide/optimizer.h"
#include "tide/checkpoint.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace tide;
#define FIELD(T, name) .def_readwrite(#name, &T::name)
PYBIND11_MODULE(_tide_native, m) {
  py::class_<StateClock>(m, "StateClock").def(py::init<>()).def(py::init<Index, Index, Index>())
    FIELD(StateClock, period) FIELD(StateClock, first) FIELD(StateClock, count)
    .def("validate", &StateClock::validate).def("to_local", &StateClock::to_local)
    .def("to_global", &StateClock::to_global).def("cut", &StateClock::cut);
  py::class_<Edge>(m, "Edge").def(py::init<Index, Index, Index>())
    FIELD(Edge, source) FIELD(Edge, target) FIELD(Edge, delay);
  py::class_<Node>(m, "Node").def(py::init<Index, bool, bool, std::string, std::string, Index, Index, Index, std::string, Index, std::vector<Index>, std::string, std::string, std::string>())
    FIELD(Node, region) FIELD(Node, clear) FIELD(Node, identity) FIELD(Node, memory) FIELD(Node, full)
    FIELD(Node, query_heads) FIELD(Node, kv_heads) FIELD(Node, window)
    FIELD(Node, emission) FIELD(Node, emit_period) FIELD(Node, emit_phases) FIELD(Node, aggregation) FIELD(Node, readout) FIELD(Node, next_state)
    FIELD(Node, state_clock);
  py::class_<Region>(m, "Region").def(py::init<Index, bool, bool, std::string, std::string>())
    .def(py::init<Index, bool, bool, std::string>())
    FIELD(Region, budget) FIELD(Region, observe_all) FIELD(Region, count_priority) FIELD(Region, read_mode) FIELD(Region, selector);
  py::class_<Adjacency>(m, "Adjacency") FIELD(Adjacency, offsets) FIELD(Adjacency, edges);
  py::class_<PortLayout>(m, "PortLayout").def(py::init<>())
    FIELD(PortLayout, edge_source) FIELD(PortLayout, edge_target) FIELD(PortLayout, input) FIELD(PortLayout, output);
  py::class_<PortBinding>(m, "PortBinding").def_readonly("kind", &PortBinding::kind).def_readonly("id", &PortBinding::id);
  py::class_<PortIndex>(m, "PortIndex").def_readonly("offsets", &PortIndex::offsets).def_readonly("bindings", &PortIndex::bindings);
  py::class_<InputOrigin>(m, "InputOrigin").def(py::init<Index, Index, Index>())
    FIELD(InputOrigin, edge) FIELD(InputOrigin, port) FIELD(InputOrigin, stride);
  py::class_<SourceDomain>(m, "SourceDomain").def(py::init<>())
    FIELD(SourceDomain, edge_target) FIELD(SourceDomain, input);
  py::class_<Graph>(m, "Graph").def(py::init<>()).def("compile", &Graph::compile)
    FIELD(Graph, nodes) FIELD(Graph, edges) FIELD(Graph, regions) FIELD(Graph, inputs) FIELD(Graph, outputs)
    FIELD(Graph, layout)
    FIELD(Graph, origins)
    FIELD(Graph, source_domain)
    .def_readonly("source_counts", &Graph::source_counts)
    .def_readonly("incoming_ports", &Graph::incoming_ports).def_readonly("outgoing_ports", &Graph::outgoing_ports)
    .def_readonly("csr", &Graph::csr).def_readonly("csc", &Graph::csc).def_readonly("identity", &Graph::identity);
  py::class_<State>(m, "State").def(py::init<Tensor, Index, Index, std::map<std::string, Tensor>>())
    FIELD(State, value) FIELD(State, last_time) FIELD(State, observations) FIELD(State, slots);
  py::class_<History>(m, "History").def(py::init<Index, std::map<std::string, Index>, std::map<std::string, std::map<Index, Index>>, std::map<std::string, Tensor>>())
    FIELD(History, last_time) FIELD(History, scalars) FIELD(History, node_maps) FIELD(History, tensors);
  py::class_<External>(m, "External").def(py::init<Index, Index, Index, Index, Tensor>())
    FIELD(External, batch) FIELD(External, port) FIELD(External, position) FIELD(External, time) FIELD(External, value);
  py::class_<Atom>(m, "Atom").def(py::init<Index, Index, Index, Index, Index, Index, Tensor>())
    FIELD(Atom, batch) FIELD(Atom, node) FIELD(Atom, time) FIELD(Atom, kind)
    FIELD(Atom, source) FIELD(Atom, position) FIELD(Atom, value);
  py::class_<Continuation>(m, "Continuation").def(py::init<>())
    FIELD(Continuation, identity) FIELD(Continuation, batch_size) FIELD(Continuation, cut)
    FIELD(Continuation, states) FIELD(Continuation, history) FIELD(Continuation, pending) FIELD(Continuation, ledger);
  py::class_<NodeWeights>(m, "NodeWeights").def(py::init<Tensor, Tensor, Tensor, Tensor>())
    FIELD(NodeWeights, decay) FIELD(NodeWeights, weight) FIELD(NodeWeights, bias) FIELD(NodeWeights, read) FIELD(NodeWeights, extra);
  py::class_<RegionWeights>(m, "RegionWeights").def(py::init<>()) FIELD(RegionWeights, extra);
  m.def("decode_add_repeat", &decode_add_repeat, py::arg("weights"), py::arg("state"), py::arg("cut"), py::arg("clock") = py::none());
  m.def("decode_fiber_bias", &decode_fiber_bias, py::arg("weights"), py::arg("state"), py::arg("cut"), py::arg("clock") = py::none());
  py::class_<Model>(m, "Model").def(py::init<>())
    FIELD(Model, nodes) FIELD(Model, regions) FIELD(Model, input_scale) FIELD(Model, agg_scale) FIELD(Model, edge_scale) FIELD(Model, output_scale)
    .def("parameters", &Model::parameters, py::arg("trainable_only") = true);
  m.def("configure_fiber_attention", &configure_fiber_attention, py::arg("graph"), py::arg("model"),
        py::arg("packing"), py::arg("pooling") = "event", py::arg("cache") = "cloned", py::arg("layout") = "event");
  py::class_<ParameterOwner>(m, "ParameterOwner")
    .def_readonly("canonical", &ParameterOwner::canonical)
    .def_readonly("aliases", &ParameterOwner::aliases)
    .def_readonly("value", &ParameterOwner::value);
  py::class_<ParameterRegistry>(m, "ParameterRegistry")
    .def(py::init<>())
    .def("add", &ParameterRegistry::add, py::arg("name"), py::arg("value"))
    .def("add_model", &ParameterRegistry::add_model, py::arg("model"), py::arg("prefix") = "",
         py::arg("trainable_only") = true)
    .def("contains", &ParameterRegistry::contains)
    .def("value", &ParameterRegistry::value)
    .def("canonical_name", &ParameterRegistry::canonical_name)
    .def("names", &ParameterRegistry::names)
    .def("owners", &ParameterRegistry::owners)
    .def("alias_partitions", &ParameterRegistry::alias_partitions);
  py::class_<OptimizerGroup>(m, "OptimizerGroup")
    .def(py::init<>())
    FIELD(OptimizerGroup, parameters) FIELD(OptimizerGroup, lr) FIELD(OptimizerGroup, weight_decay)
    FIELD(OptimizerGroup, momentum) FIELD(OptimizerGroup, dampening) FIELD(OptimizerGroup, beta1)
    FIELD(OptimizerGroup, beta2) FIELD(OptimizerGroup, eps) FIELD(OptimizerGroup, nesterov)
    FIELD(OptimizerGroup, amsgrad) FIELD(OptimizerGroup, maximize);
  py::class_<OptimizerLayout>(m, "OptimizerLayout")
    .def_readonly("class_name", &OptimizerLayout::class_name)
    .def_readonly("groups", &OptimizerLayout::groups);
  py::class_<OptimizerState>(m, "OptimizerState")
    .def_readonly("step", &OptimizerState::step)
    .def_readonly("momentum_buffer", &OptimizerState::momentum_buffer)
    .def_readonly("exp_avg", &OptimizerState::exp_avg)
    .def_readonly("exp_avg_sq", &OptimizerState::exp_avg_sq)
    .def_readonly("max_exp_avg_sq", &OptimizerState::max_exp_avg_sq);
  py::class_<NamedOptimizer>(m, "NamedOptimizer")
    .def("step", &NamedOptimizer::step)
    .def("zero_grad", &NamedOptimizer::zero_grad, py::arg("set_to_none") = true)
    .def("layout", &NamedOptimizer::layout)
    .def("groups", &NamedOptimizer::groups)
    .def("state", &NamedOptimizer::state);
  py::class_<SGD, NamedOptimizer>(m, "SGD")
    .def(py::init<ParameterRegistry&, std::vector<OptimizerGroup>>(), py::arg("registry"), py::arg("groups") = std::vector<OptimizerGroup>{},
         py::keep_alive<1, 2>())
    .def("step", &SGD::step)
    .def("zero_grad", &SGD::zero_grad, py::arg("set_to_none") = true)
    .def("layout", &SGD::layout)
    .def("groups", &SGD::groups)
    .def("state", &SGD::state);
  py::class_<AdamW, NamedOptimizer>(m, "AdamW")
    .def(py::init<ParameterRegistry&, std::vector<OptimizerGroup>>(), py::arg("registry"), py::arg("groups") = std::vector<OptimizerGroup>{},
         py::keep_alive<1, 2>())
    .def("step", &AdamW::step)
    .def("zero_grad", &AdamW::zero_grad, py::arg("set_to_none") = true)
    .def("layout", &AdamW::layout)
    .def("groups", &AdamW::groups)
    .def("state", &AdamW::state);
  m.def("save_checkpoint", [](const std::string& path, const ParameterRegistry& registry,
                               const NamedOptimizer* optimizer, const std::string& identity) {
    Checkpoint::save(path, registry, optimizer, identity);
  }, py::arg("path"), py::arg("registry"), py::arg("optimizer") = nullptr,
     py::arg("identity") = "");
  m.def("load_checkpoint", [](const std::string& path, ParameterRegistry& registry,
                               NamedOptimizer* optimizer, const std::string& expected_identity) {
    Checkpoint::load(path, registry, optimizer, expected_identity);
  }, py::arg("path"), py::arg("registry"), py::arg("optimizer") = nullptr,
     py::arg("expected_identity") = "");
  py::class_<Event>(m, "Event")
    FIELD(Event, batch) FIELD(Event, node) FIELD(Event, time) FIELD(Event, fiber) FIELD(Event, content)
    FIELD(Event, proposal) FIELD(Event, descriptor) FIELD(Event, control) FIELD(Event, comparison)
    FIELD(Event, next) FIELD(Event, full) FIELD(Event, active) FIELD(Event, history)
    .def_property_readonly("emitted", [](const Event& e) {
      std::map<Index, Tensor> result;
      for (const auto& value : e.emitted) result.emplace(value.slot, value.value);
      return result;
    })
    .def_property_readonly("contributions", [](const Event& e) {
      std::map<Index, Tensor> result;
      for (const auto& value : e.contributions) result.emplace(value.slot, value.value);
      return result;
    })
    .def_property_readonly("proposal_slots", [](const Event& e) { return e.proposed_state.slots; })
    .def_property_readonly("comparison_slots", [](const Event& e) { return e.comparison_state.slots; })
    .def_property_readonly("next_slots", [](const Event& e) { return e.next_state.slots; });
  py::class_<Output>(m, "Output").def(py::init<Index, Index, Index, Tensor>())
    FIELD(Output, batch) FIELD(Output, time) FIELD(Output, port) FIELD(Output, value);
  m.def("token_inputs", &token_inputs, py::call_guard<py::gil_scoped_release>());
  py::class_<Result>(m, "Result") FIELD(Result, continuation) FIELD(Result, trace)
    FIELD(Result, outputs) FIELD(Result, messages) FIELD(Result, stats);
  py::class_<Options>(m, "Options").def(py::init<>())
    FIELD(Options, workers) FIELD(Options, packed) FIELD(Options, trace) FIELD(Options, mode) FIELD(Options, zeta)
    FIELD(Options, prefill) FIELD(Options, max_events)
    FIELD(Options, profile) FIELD(Options, parallel_regions) FIELD(Options, compact_events) FIELD(Options, defer_state_release);
  py::class_<DenseLinear>(m, "DenseLinear").def(py::init<Index>())
    .def("run", [](DenseLinear& head, const Tensor& x, const Tensor& weight, const std::optional<Tensor>& bias) {
      return head.run(x, weight, bias.value_or(Tensor()));
    }, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::call_guard<py::gil_scoped_release>());
  py::class_<Streaming>(m, "Streaming").def(py::init<Graph, Model, Options>())
    .def("run", &Streaming::run, py::call_guard<py::gil_scoped_release>());
  py::class_<AdvanceResult>(m, "AdvanceResult") FIELD(AdvanceResult, cut) FIELD(AdvanceResult, trace)
    FIELD(AdvanceResult, outputs) FIELD(AdvanceResult, messages) FIELD(AdvanceResult, stats);
  py::class_<StreamingCursor>(m, "StreamingCursor")
    .def(py::init<Streaming&, Continuation>(), py::keep_alive<1, 2>())
    .def("advance", &StreamingCursor::advance, py::call_guard<py::gil_scoped_release>())
    .def("snapshot", &StreamingCursor::snapshot, py::call_guard<py::gil_scoped_release>())
    .def("detach", &StreamingCursor::detach, py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("cut", &StreamingCursor::cut)
    .def_property_readonly("failed", &StreamingCursor::failed);
  py::class_<Frontier>(m, "Frontier").def(py::init<Graph, Model, Options>())
    .def("run", &Frontier::run, py::call_guard<py::gil_scoped_release>());
  py::class_<Specialized>(m, "Specialized").def(py::init<Graph, Model, Options, std::string>())
    .def("run", &Specialized::run, py::call_guard<py::gil_scoped_release>());
  m.def("emit", &emit);
}
#undef FIELD
