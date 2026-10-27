#include "tide/resident_weighted_training.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace tide;using namespace tide::device_online;
void bind_training_types(py::module_&);
#define FIELD(T,name) .def_readwrite(#name,&T::name)
#define VIEW(T,name) .def_readonly(#name,&T::name)
void bind_weighted_training(py::module_& m){
  bind_training_types(m);
  py::class_<WeightedTrainingLimits>(m,"WeightedTrainingLimits").def(py::init<>())
    FIELD(WeightedTrainingLimits,forward) FIELD(WeightedTrainingLimits,windows) FIELD(WeightedTrainingLimits,retained_bytes)
    FIELD(WeightedTrainingLimits,backward_bytes) FIELD(WeightedTrainingLimits,optimizer_bytes) FIELD(WeightedTrainingLimits,total_bytes);
  py::class_<WeightedTrainingField>(m,"WeightedTrainingField")
    VIEW(WeightedTrainingField,values) VIEW(WeightedTrainingField,present) VIEW(WeightedTrainingField,valid);
  py::class_<WeightedTrainingWindow>(m,"WeightedTrainingWindow")
    VIEW(WeightedTrainingWindow,token) VIEW(WeightedTrainingWindow,start) VIEW(WeightedTrainingWindow,stop)
    VIEW(WeightedTrainingWindow,fields) VIEW(WeightedTrainingWindow,metadata);
  py::class_<WeightedCotangent>(m,"WeightedCotangent").def(py::init<>()) FIELD(WeightedCotangent,values) FIELD(WeightedCotangent,connected);
  py::class_<WeightedCotangents>(m,"WeightedCotangents").def(py::init<>()) FIELD(WeightedCotangents,token) FIELD(WeightedCotangents,fields);
  py::class_<WeightedStateGradient>(m,"WeightedStateGradient")
    VIEW(WeightedStateGradient,node) VIEW(WeightedStateGradient,family) VIEW(WeightedStateGradient,integer)
    VIEW(WeightedStateGradient,present) VIEW(WeightedStateGradient,fields);
  py::class_<WeightedGradients>(m,"WeightedGradients") VIEW(WeightedGradients,parameters) VIEW(WeightedGradients,initial)
    VIEW(WeightedGradients,boundaries) VIEW(WeightedGradients,statistics);
  py::class_<WeightedPeerStep>(m,"WeightedPeerStep") VIEW(WeightedPeerStep,step) VIEW(WeightedPeerStep,parameters)
    VIEW(WeightedPeerStep,transferred_bytes) VIEW(WeightedPeerStep,reserved_tensor_bytes);
  py::class_<WeightedTrainingSession>(m,"WeightedTrainingSession")
    .def(py::init<Graph,Model,const Continuation&,at::Device,ResidentOptimizerKind,std::vector<OptimizerGroup>,WeightedTrainingLimits>(),py::call_guard<py::gil_scoped_release>())
    .def(py::init<Graph,Model,const ResidentTrainingCheckpoint&,at::Device,WeightedTrainingLimits>(),py::call_guard<py::gil_scoped_release>())
    .def("advance",&WeightedTrainingSession::advance,py::call_guard<py::gil_scoped_release>())
    .def("backward",&WeightedTrainingSession::backward,py::call_guard<py::gil_scoped_release>())
    .def("accumulate",&WeightedTrainingSession::accumulate,py::call_guard<py::gil_scoped_release>())
    .def("step",&WeightedTrainingSession::step,py::call_guard<py::gil_scoped_release>())
    .def_static("step_replicas",&WeightedTrainingSession::step_replicas,py::arg("sessions"),
      py::arg("communication_bytes")=256*1024*1024,py::arg("timeout_ms")=30000,py::call_guard<py::gil_scoped_release>())
    .def("detach",&WeightedTrainingSession::detach,py::call_guard<py::gil_scoped_release>())
    .def("checkpoint",&WeightedTrainingSession::checkpoint,py::call_guard<py::gil_scoped_release>())
    .def("optimizer_kind",&WeightedTrainingSession::optimizer_kind)
    .def("optimizer_groups",&WeightedTrainingSession::optimizer_groups)
    .def("result",&WeightedTrainingSession::result,py::call_guard<py::gil_scoped_release>())
    .def("close",&WeightedTrainingSession::close,py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("cut",&WeightedTrainingSession::cut).def_property_readonly("generation",&WeightedTrainingSession::generation);
}
#undef FIELD
#undef VIEW
