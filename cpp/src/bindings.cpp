// Python is only an adapter; core library sources contain no Python headers.
#include "tide/stream.h"
#include "tide/ops.h"
#include "tide/frontier.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace tide;
#define FIELD(T, name) .def_readwrite(#name, &T::name)
PYBIND11_MODULE(_tide_native, m) {
  py::class_<Edge>(m, "Edge").def(py::init<Index, Index, Index>())
    FIELD(Edge, source) FIELD(Edge, target) FIELD(Edge, delay);
  py::class_<Node>(m, "Node").def(py::init<Index, bool>()) FIELD(Node, region) FIELD(Node, clear);
  py::class_<Region>(m, "Region").def(py::init<Index, bool, bool>())
    FIELD(Region, budget) FIELD(Region, observe_all) FIELD(Region, count_priority);
  py::class_<Adjacency>(m, "Adjacency") FIELD(Adjacency, offsets) FIELD(Adjacency, edges);
  py::class_<Graph>(m, "Graph").def(py::init<>()).def("compile", &Graph::compile)
    FIELD(Graph, nodes) FIELD(Graph, edges) FIELD(Graph, regions) FIELD(Graph, inputs) FIELD(Graph, outputs)
    .def_readonly("csr", &Graph::csr).def_readonly("csc", &Graph::csc).def_readonly("identity", &Graph::identity);
  py::class_<State>(m, "State").def(py::init<Tensor, Index, Index>())
    FIELD(State, value) FIELD(State, last_time) FIELD(State, observations);
  py::class_<External>(m, "External").def(py::init<Index, Index, Index, Index, Tensor>())
    FIELD(External, batch) FIELD(External, port) FIELD(External, position) FIELD(External, time) FIELD(External, value);
  py::class_<Atom>(m, "Atom").def(py::init<Index, Index, Index, Index, Index, Index, Tensor>())
    FIELD(Atom, batch) FIELD(Atom, node) FIELD(Atom, time) FIELD(Atom, kind)
    FIELD(Atom, source) FIELD(Atom, position) FIELD(Atom, value);
  py::class_<Continuation>(m, "Continuation").def(py::init<>())
    FIELD(Continuation, identity) FIELD(Continuation, batch_size) FIELD(Continuation, cut)
    FIELD(Continuation, states) FIELD(Continuation, history) FIELD(Continuation, pending) FIELD(Continuation, ledger);
  py::class_<NodeWeights>(m, "NodeWeights").def(py::init<Tensor, Tensor, Tensor, Tensor>())
    FIELD(NodeWeights, decay) FIELD(NodeWeights, weight) FIELD(NodeWeights, bias) FIELD(NodeWeights, read);
  py::class_<Model>(m, "Model").def(py::init<>())
    FIELD(Model, nodes) FIELD(Model, input_scale) FIELD(Model, agg_scale) FIELD(Model, edge_scale) FIELD(Model, output_scale);
  py::class_<Event>(m, "Event")
    FIELD(Event, batch) FIELD(Event, node) FIELD(Event, time) FIELD(Event, fiber) FIELD(Event, content)
    FIELD(Event, proposal) FIELD(Event, descriptor) FIELD(Event, control) FIELD(Event, comparison)
    FIELD(Event, next) FIELD(Event, full) FIELD(Event, active) FIELD(Event, history);
  py::class_<Output>(m, "Output") FIELD(Output, batch) FIELD(Output, time) FIELD(Output, port) FIELD(Output, value);
  py::class_<Result>(m, "Result") FIELD(Result, continuation) FIELD(Result, trace)
    FIELD(Result, outputs) FIELD(Result, messages) FIELD(Result, stats);
  py::class_<Options>(m, "Options").def(py::init<>())
    FIELD(Options, workers) FIELD(Options, packed) FIELD(Options, trace) FIELD(Options, mode) FIELD(Options, zeta)
    FIELD(Options, prefill) FIELD(Options, max_events);
  py::class_<Streaming>(m, "Streaming").def(py::init<Graph, Model, Options>())
    .def("run", &Streaming::run, py::call_guard<py::gil_scoped_release>());
  py::class_<Frontier>(m, "Frontier").def(py::init<Graph, Model, Options>())
    .def("run", &Frontier::run, py::call_guard<py::gil_scoped_release>());
  m.def("emit", &emit);
}
#undef FIELD
