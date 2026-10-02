#include "tide/resident_training.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

namespace py=pybind11;
using namespace tide;
#define FIELD(T, name) .def_readwrite(#name,&T::name)
#define VIEW(T, name) .def_readonly(#name,&T::name)
void bind_resident_training(py::module_& m) {
  py::enum_<ResidentOptimizerKind>(m,"OptimizerKind")
    .value("sgd",ResidentOptimizerKind::sgd).value("adamw",ResidentOptimizerKind::adamw);
  py::class_<ResidentTrainingLimits>(m,"TrainingLimits").def(py::init<>())
    FIELD(ResidentTrainingLimits,forward) FIELD(ResidentTrainingLimits,windows) FIELD(ResidentTrainingLimits,retained_bytes)
    FIELD(ResidentTrainingLimits,backward_bytes) FIELD(ResidentTrainingLimits,optimizer_bytes)
    FIELD(ResidentTrainingLimits,program_workspace_bytes) FIELD(ResidentTrainingLimits,reverse_chunk_rows)
    FIELD(ResidentTrainingLimits,placement);
  py::class_<ResidentToken>(m,"TrainingToken")
    VIEW(ResidentToken,session) VIEW(ResidentToken,index) VIEW(ResidentToken,generation);
  py::class_<ResidentCacheWindow>(m,"CacheWindow")
    VIEW(ResidentCacheWindow,nodes) VIEW(ResidentCacheWindow,key) VIEW(ResidentCacheWindow,value)
    VIEW(ResidentCacheWindow,lengths) VIEW(ResidentCacheWindow,present) VIEW(ResidentCacheWindow,log_bias);
  py::class_<ResidentCacheCotangents>(m,"CacheCotangents").def(py::init<>())
    FIELD(ResidentCacheCotangents,key) FIELD(ResidentCacheCotangents,value)
    FIELD(ResidentCacheCotangents,key_connected) FIELD(ResidentCacheCotangents,value_connected)
    FIELD(ResidentCacheCotangents,log_bias) FIELD(ResidentCacheCotangents,log_bias_connected);
  py::class_<ResidentCacheGradient>(m,"CacheGradient")
    VIEW(ResidentCacheGradient,nodes) VIEW(ResidentCacheGradient,key) VIEW(ResidentCacheGradient,value)
    VIEW(ResidentCacheGradient,lengths) VIEW(ResidentCacheGradient,key_connected) VIEW(ResidentCacheGradient,value_connected)
    VIEW(ResidentCacheGradient,log_bias) VIEW(ResidentCacheGradient,log_bias_connected);
  py::class_<ResidentStateWindow>(m,"StateWindow")
    VIEW(ResidentStateWindow,nodes) VIEW(ResidentStateWindow,values) VIEW(ResidentStateWindow,present) VIEW(ResidentStateWindow,cache);
  py::class_<ResidentStateCotangents>(m,"StateCotangents").def(py::init<>())
    FIELD(ResidentStateCotangents,final) FIELD(ResidentStateCotangents,final_connected) FIELD(ResidentStateCotangents,cache);
  py::class_<ResidentStateGradient>(m,"StateGradient")
    VIEW(ResidentStateGradient,nodes) VIEW(ResidentStateGradient,initial)
    VIEW(ResidentStateGradient,initial_connected) VIEW(ResidentStateGradient,cache);
  py::class_<ResidentParameterGradient>(m,"ParameterGradient")
    VIEW(ResidentParameterGradient,names) VIEW(ResidentParameterGradient,aliases) VIEW(ResidentParameterGradient,offsets)
    VIEW(ResidentParameterGradient,values) VIEW(ResidentParameterGradient,connected);
  py::class_<ResidentTrainingWindow>(m,"TrainingWindow")
    VIEW(ResidentTrainingWindow,token) VIEW(ResidentTrainingWindow,start) VIEW(ResidentTrainingWindow,stop)
    VIEW(ResidentTrainingWindow,outputs) VIEW(ResidentTrainingWindow,pending_coordinates)
    VIEW(ResidentTrainingWindow,pending_values) VIEW(ResidentTrainingWindow,pending_valid)
    VIEW(ResidentTrainingWindow,state_values) VIEW(ResidentTrainingWindow,state_present) VIEW(ResidentTrainingWindow,cache)
    VIEW(ResidentTrainingWindow,states);
  py::class_<ResidentCotangents>(m,"Cotangents").def(py::init<>())
    FIELD(ResidentCotangents,token) FIELD(ResidentCotangents,outputs) FIELD(ResidentCotangents,outputs_connected)
    FIELD(ResidentCotangents,pending) FIELD(ResidentCotangents,pending_connected)
    FIELD(ResidentCotangents,final) FIELD(ResidentCotangents,final_connected) FIELD(ResidentCotangents,cache)
    FIELD(ResidentCotangents,states);
  py::class_<ResidentBoundaryGradient>(m,"BoundaryGradient")
    VIEW(ResidentBoundaryGradient,token) VIEW(ResidentBoundaryGradient,coordinates) VIEW(ResidentBoundaryGradient,values)
    VIEW(ResidentBoundaryGradient,valid) VIEW(ResidentBoundaryGradient,connected);
  py::class_<ResidentGradients>(m,"Gradients")
    VIEW(ResidentGradients,names) VIEW(ResidentGradients,aliases) VIEW(ResidentGradients,offsets)
    VIEW(ResidentGradients,values) VIEW(ResidentGradients,connected) VIEW(ResidentGradients,initial)
    VIEW(ResidentGradients,initial_connected) VIEW(ResidentGradients,boundaries) VIEW(ResidentGradients,initial_cache)
    VIEW(ResidentGradients,parameter_shards) VIEW(ResidentGradients,initial_shards) VIEW(ResidentGradients,statistics);
  py::class_<ResidentOptimizerState>(m,"DeviceOptimizerState").def(py::init<>())
    FIELD(ResidentOptimizerState,values) FIELD(ResidentOptimizerState,first) FIELD(ResidentOptimizerState,second)
    FIELD(ResidentOptimizerState,maximum) FIELD(ResidentOptimizerState,steps) FIELD(ResidentOptimizerState,corrections);
  py::class_<ResidentTrainingCheckpoint>(m,"TrainingCheckpoint").def(py::init<>())
    FIELD(ResidentTrainingCheckpoint,schema) FIELD(ResidentTrainingCheckpoint,generation) FIELD(ResidentTrainingCheckpoint,next_token)
    FIELD(ResidentTrainingCheckpoint,continuation) FIELD(ResidentTrainingCheckpoint,parameters) FIELD(ResidentTrainingCheckpoint,aliases)
    FIELD(ResidentTrainingCheckpoint,trainable) FIELD(ResidentTrainingCheckpoint,optimizer) FIELD(ResidentTrainingCheckpoint,groups)
    FIELD(ResidentTrainingCheckpoint,offsets) FIELD(ResidentTrainingCheckpoint,state)
    FIELD(ResidentTrainingCheckpoint,mode) FIELD(ResidentTrainingCheckpoint,zeta);
  py::class_<ResidentStep>(m,"TrainingStep")
    VIEW(ResidentStep,applied) VIEW(ResidentStep,refusal_code) VIEW(ResidentStep,generation);
  py::class_<ResidentTrainingSession>(m,"TrainingSession")
    .def(py::init<Graph,Model,const Continuation&,at::Device,ResidentOptimizerKind,std::vector<OptimizerGroup>,ResidentTrainingLimits>(),py::call_guard<py::gil_scoped_release>())
    .def(py::init<Graph,Model,const ResidentTrainingCheckpoint&,at::Device,ResidentTrainingLimits>(),py::call_guard<py::gil_scoped_release>())
    .def("advance",&ResidentTrainingSession::advance,py::call_guard<py::gil_scoped_release>())
    .def("backward",&ResidentTrainingSession::backward,py::call_guard<py::gil_scoped_release>())
    .def("accumulate",&ResidentTrainingSession::accumulate,py::arg("max_bytes")=128*1024*1024,py::call_guard<py::gil_scoped_release>())
    .def("step",&ResidentTrainingSession::step,py::call_guard<py::gil_scoped_release>())
    .def("detach",&ResidentTrainingSession::detach,py::call_guard<py::gil_scoped_release>())
    .def("checkpoint",&ResidentTrainingSession::checkpoint,py::call_guard<py::gil_scoped_release>())
    .def("snapshot_device",py::overload_cast<Index,bool,const std::map<Index,Index>&>(&ResidentTrainingSession::snapshot_device,py::const_),
         py::arg("max_bytes"),py::arg("compact")=false,py::arg("device_budgets")=std::map<Index,Index>{},py::call_guard<py::gil_scoped_release>())
    .def("restore_device",&ResidentTrainingSession::restore_device,py::call_guard<py::gil_scoped_release>())
    .def("result",&ResidentTrainingSession::result,py::call_guard<py::gil_scoped_release>())
    .def("close",&ResidentTrainingSession::close,py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("cut",&ResidentTrainingSession::cut)
    .def_property_readonly("generation",&ResidentTrainingSession::generation)
    .def_property_readonly("accumulated_batches",&ResidentTrainingSession::accumulated_batches)
    .def_property_readonly("placement",&ResidentTrainingSession::placement)
    .def_property_readonly("retained_windows",&ResidentTrainingSession::retained_windows);
}
#undef FIELD
#undef VIEW
