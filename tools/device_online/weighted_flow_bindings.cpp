#include "tide/resident_weighted.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_weighted_training(py::module_&);
PYBIND11_MODULE(_tide_weighted_resident,module){
  using namespace tide;using namespace tide::device_online;
#define FIELD(name) .def_readwrite(#name,&WeightedFlowLimits::name)
  py::class_<WeightedFlowLimits>(module,"WeightedLimits").def(py::init<>())
    FIELD(queue) FIELD(stages) FIELD(cache) FIELD(tensor_bytes) FIELD(operator_bytes) FIELD(timeout_ms) FIELD(prefill);
  py::class_<WeightedFlow>(module,"WeightedFlow")
    .def(py::init<Graph,Model,const Continuation&,at::Device,WeightedFlowLimits>(),py::call_guard<py::gil_scoped_release>())
    .def("advance",&WeightedFlow::advance,py::call_guard<py::gil_scoped_release>())
    .def("checkpoint",&WeightedFlow::checkpoint,py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("reserved_bytes",&WeightedFlow::reserved_bytes)
    .def("close",&WeightedFlow::close,py::call_guard<py::gil_scoped_release>());
#undef FIELD
  bind_weighted_training(module);
}
