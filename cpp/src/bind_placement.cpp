#include "tide/placement.h"
#include "tide/ownership.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

namespace py=pybind11;
void bind_placement(py::module_& m) {
  using namespace tide;
  py::class_<ExecutionPlacement>(m,"ExecutionPlacement").def(py::init<>())
    .def_readwrite("preset",&ExecutionPlacement::preset)
    .def_readwrite("read",&ExecutionPlacement::read)
    .def_readwrite("control",&ExecutionPlacement::control)
    .def_readwrite("selection",&ExecutionPlacement::selection)
    .def_readwrite("events",&ExecutionPlacement::events)
    .def_readwrite("scoring_dtype",&ExecutionPlacement::scoring_dtype);
  m.def("resolve_placement",[](const ExecutionPlacement& p,at::Device payload){return resolve_placement(p,payload).record();},
        py::arg("placement"),py::arg("payload_device"));
  m.def("place_model",&place_model,py::arg("graph"),py::arg("model"),py::arg("placement"));
  m.def("place_payloads",&place_payloads,py::arg("graph"),py::arg("model"),py::arg("node_devices"));
}
