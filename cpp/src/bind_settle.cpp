// Client/test adapter only. Construction, encoding and projection live in core.
#include "tide/settle.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

namespace py = pybind11;
void bind_settle(py::module_& module) {
  using namespace tide;
  py::class_<SettleGraph>(module, "SettleGraph")
    .def(py::init<Graph, std::vector<Index>, std::string, Index>(), py::arg("graph"),
         py::arg("ranks"), py::arg("binding") = "broadcast_sum", py::arg("time_stride") = 0)
    .def_property_readonly("graph", [](const SettleGraph& s) { return s.graph(); })
    .def_property_readonly("encoded_graph", [](const SettleGraph& s) { return s.encoded_graph(); })
    .def_property_readonly("ranks", &SettleGraph::ranks)
    .def_property_readonly("stride", &SettleGraph::stride)
    .def_property_readonly("binding", &SettleGraph::binding)
    .def_property_readonly("output_rank", &SettleGraph::output_rank)
    .def("rank", &SettleGraph::rank)
    .def("embed_model", &SettleGraph::embed_model)
    .def("embed_initial", &SettleGraph::embed_initial)
    .def("external", &SettleGraph::external, py::arg("values"), py::arg("start_position") = 0,
         py::arg("encoded") = true)
    .def("port_external", &SettleGraph::port_external, py::arg("values"), py::arg("positions"),
         py::arg("port"), py::arg("batch") = 0, py::arg("start_occurrence") = 0)
    .def("project", &SettleGraph::project);
  py::class_<SettleExecutor>(module, "SettleExecutor")
    .def(py::init<SettleGraph, Model, Options, std::string>(), py::arg("spec"), py::arg("model"),
         py::arg("options"), py::arg("algorithm") = "frontier")
    .def("run", &SettleExecutor::run, py::call_guard<py::gil_scoped_release>())
    .def("run_ports", &SettleExecutor::run_ports, py::call_guard<py::gil_scoped_release>());
}
