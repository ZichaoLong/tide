#include "tide/resident_blocks.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace tide;using namespace tide::device_online;
#define FIELD(T,name) .def_readwrite(#name,&T::name)
#define VIEW(T,name) .def_readonly(#name,&T::name)
void bind_block_resident(py::module_& m){
  py::class_<BlockResidentLimits>(m,"BlockResidentLimits").def(py::init<>())
    FIELD(BlockResidentLimits,cache) FIELD(BlockResidentLimits,forward_bytes) FIELD(BlockResidentLimits,backward_bytes)
    FIELD(BlockResidentLimits,optimizer_bytes) FIELD(BlockResidentLimits,total_bytes) FIELD(BlockResidentLimits,workspace_bytes)
    FIELD(BlockResidentLimits,windows) FIELD(BlockResidentLimits,timeout_ms);
  py::class_<BlockField>(m,"BlockField") VIEW(BlockField,values) VIEW(BlockField,present) VIEW(BlockField,valid);
  py::class_<BlockRoot>(m,"BlockRoot").def(py::init<>()) FIELD(BlockRoot,values) FIELD(BlockRoot,connected);
  py::class_<BlockWindow>(m,"BlockWindow") VIEW(BlockWindow,token) VIEW(BlockWindow,fields) VIEW(BlockWindow,metadata);
  py::class_<BlockRoots>(m,"BlockRoots").def(py::init<>()) FIELD(BlockRoots,token) FIELD(BlockRoots,fields);
  py::class_<BlockGradients>(m,"BlockGradients") VIEW(BlockGradients,parameters) VIEW(BlockGradients,inputs)
    VIEW(BlockGradients,initial) VIEW(BlockGradients,bound_initial);
  py::class_<BlockRawRecord>(m,"BlockRawRecord").def(py::init<>()) FIELD(BlockRawRecord,floating) FIELD(BlockRawRecord,integer);
  py::class_<BlockCheckpoint>(m,"BlockCheckpoint").def(py::init<>())
    FIELD(BlockCheckpoint,schema) FIELD(BlockCheckpoint,generation) FIELD(BlockCheckpoint,next_token) FIELD(BlockCheckpoint,identity)
    FIELD(BlockCheckpoint,aliases) FIELD(BlockCheckpoint,parameters) FIELD(BlockCheckpoint,optimizer) FIELD(BlockCheckpoint,groups)
    FIELD(BlockCheckpoint,slots) FIELD(BlockCheckpoint,state) FIELD(BlockCheckpoint,bound_initial) FIELD(BlockCheckpoint,last_positions)
    FIELD(BlockCheckpoint,node_present) FIELD(BlockCheckpoint,region_times) FIELD(BlockCheckpoint,input_counts) FIELD(BlockCheckpoint,input_times);
  py::class_<BlockResidentSession>(m,"BlockResidentSession")
    .def(py::init<SettleGraph,Model,const Continuation&,at::Device,ResidentOptimizerKind,std::vector<OptimizerGroup>,BlockResidentLimits>(),py::call_guard<py::gil_scoped_release>())
    .def(py::init<SettleGraph,Model,const Continuation&,const BlockCheckpoint&,at::Device,BlockResidentLimits>(),py::call_guard<py::gil_scoped_release>())
    .def("advance",&BlockResidentSession::advance,py::call_guard<py::gil_scoped_release>())
    .def("backward",&BlockResidentSession::backward,py::call_guard<py::gil_scoped_release>())
    .def("step",&BlockResidentSession::step,py::call_guard<py::gil_scoped_release>())
    .def("detach",&BlockResidentSession::detach,py::call_guard<py::gil_scoped_release>())
    .def("checkpoint",&BlockResidentSession::checkpoint,py::call_guard<py::gil_scoped_release>())
    .def("continuation",&BlockResidentSession::continuation,py::call_guard<py::gil_scoped_release>())
    .def("optimizer_kind",&BlockResidentSession::optimizer_kind).def("optimizer_groups",&BlockResidentSession::optimizer_groups)
    .def("resource_usage",&BlockResidentSession::resource_usage)
    .def_property_readonly("cut",&BlockResidentSession::cut).def_property_readonly("generation",&BlockResidentSession::generation)
    .def("close",&BlockResidentSession::close,py::call_guard<py::gil_scoped_release>());
  m.def("save_block_checkpoint",&save_block_checkpoint,py::arg("checkpoint"),py::arg("path"),py::arg("max_bytes")=int64_t{1}<<32,py::call_guard<py::gil_scoped_release>());
  m.def("load_block_checkpoint",&load_block_checkpoint,py::arg("path"),py::arg("max_bytes")=int64_t{1}<<32,py::call_guard<py::gil_scoped_release>());
}
#undef FIELD
#undef VIEW
