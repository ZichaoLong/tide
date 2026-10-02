#include "tide/resident.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

namespace py=pybind11;
using namespace tide;
#define FIELD(T, name) .def_readwrite(#name,&T::name)
void bind_resident_training(py::module_&);
PYBIND11_MODULE(_tide_resident,m) {
  // Graph/Model/Continuation/Result belong to the matching _tide_native module.
  // The loader validates that core before importing this optional backend.
  py::enum_<ResidentChunkPolicy>(m,"ChunkPolicy")
    .value("conservative",ResidentChunkPolicy::conservative).value("aggressive",ResidentChunkPolicy::aggressive);
  py::class_<ResidentPlacement>(m,"Placement").def(py::init<>())
    FIELD(ResidentPlacement,devices) FIELD(ResidentPlacement,policy)
    FIELD(ResidentPlacement,full_owners) FIELD(ResidentPlacement,state_owners);
  m.attr("TrainingPlacement")=m.attr("Placement"); // Preserve existing clients.
  py::class_<ResidentContinuation>(m,"DeviceContinuation")
    .def_property_readonly("cut",&ResidentContinuation::cut)
    .def_property_readonly("batch_size",&ResidentContinuation::batch_size)
    .def_property_readonly("tensor_bytes",&ResidentContinuation::tensor_bytes);
  py::class_<ResidentLimits>(m,"Limits").def(py::init<>())
    FIELD(ResidentLimits,queue) FIELD(ResidentLimits,arrivals) FIELD(ResidentLimits,outputs)
    FIELD(ResidentLimits,trace) FIELD(ResidentLimits,stages) FIELD(ResidentLimits,workspace_bytes)
    FIELD(ResidentLimits,chunk_policy) FIELD(ResidentLimits,full_chunk_rows) FIELD(ResidentLimits,emission_chunk_rows)
    FIELD(ResidentLimits,aggregate_chunk_rows) FIELD(ResidentLimits,attention_chunk_rows)
    FIELD(ResidentLimits,attention_key_rows) FIELD(ResidentLimits,kv_rows) FIELD(ResidentLimits,kv_trace_rows)
    FIELD(ResidentLimits,max_repeat_ticks) FIELD(ResidentLimits,prefill) FIELD(ResidentLimits,diagnostics)
    FIELD(ResidentLimits,vectorized_aggregate) FIELD(ResidentLimits,vectorized_state) FIELD(ResidentLimits,vectorized_read)
    FIELD(ResidentLimits,mode) FIELD(ResidentLimits,zeta);
  py::class_<ResidentWindow>(m,"Window")
    .def_readonly("coordinates",&ResidentWindow::coordinates).def_readonly("values",&ResidentWindow::values)
    .def_readonly("valid",&ResidentWindow::valid).def_readonly("output_stats",&ResidentWindow::output_stats)
    .def_readonly("pending_stats",&ResidentWindow::pending_stats).def_readonly("stages",&ResidentWindow::stages)
    .def_readonly("events",&ResidentWindow::events).def_readonly("full_chunks",&ResidentWindow::full_chunks)
    .def_readonly("emission_chunks",&ResidentWindow::emission_chunks);
  py::class_<ResidentSession>(m,"Session")
    .def(py::init<Graph,Model,const Continuation&,at::Device,ResidentLimits>(),py::call_guard<py::gil_scoped_release>())
    .def(py::init<Graph,Model,const Continuation&,at::Device,ResidentLimits,ResidentPlacement>(),py::call_guard<py::gil_scoped_release>())
    .def("advance",&ResidentSession::advance,py::call_guard<py::gil_scoped_release>())
    .def("snapshot",&ResidentSession::snapshot,py::call_guard<py::gil_scoped_release>())
    .def("snapshot_device",py::overload_cast<Index,bool>(&ResidentSession::snapshot_device,py::const_),
         py::arg("max_bytes"),py::arg("compact")=false,py::call_guard<py::gil_scoped_release>())
    .def("restore_device",&ResidentSession::restore_device,py::call_guard<py::gil_scoped_release>())
    .def("result",&ResidentSession::result,py::call_guard<py::gil_scoped_release>())
    .def("close",&ResidentSession::close,py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("cut",&ResidentSession::cut)
    .def_property_readonly("placement",&ResidentSession::placement);
  bind_resident_training(m);
}
#undef FIELD
