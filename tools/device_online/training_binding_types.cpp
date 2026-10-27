// Shared C++ value types may be loaded by either public Python adapter first.
#include "tide/resident_training.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace tide;
namespace {
template<class T> bool alias_registered(py::module_& m,const char* name){
  const auto* registered=py::detail::get_type_info(typeid(T),false);
  if(!registered)return false;
  // py::type::of<T>() rejects enum casters in newer pybind11. The registered
  // Python class is shared by both adapters, including their optimizer enum.
  m.attr(name)=py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(registered->type));
  return true;
}
}
#define FIELD(T, name) .def_readwrite(#name,&T::name)
#define VIEW(T, name) .def_readonly(#name,&T::name)
void bind_training_types(py::module_& m) {
  if(!alias_registered<ResidentOptimizerKind>(m,"OptimizerKind")) {
  py::enum_<ResidentOptimizerKind>(m,"OptimizerKind")
    .value("sgd",ResidentOptimizerKind::sgd).value("adamw",ResidentOptimizerKind::adamw);
  }
  if(!alias_registered<ResidentToken>(m,"TrainingToken")) {
  py::class_<ResidentToken>(m,"TrainingToken")
    VIEW(ResidentToken,session) VIEW(ResidentToken,index) VIEW(ResidentToken,generation);
  }
  if(!alias_registered<ResidentParameterGradient>(m,"ParameterGradient")) {
  py::class_<ResidentParameterGradient>(m,"ParameterGradient")
    VIEW(ResidentParameterGradient,names) VIEW(ResidentParameterGradient,aliases) VIEW(ResidentParameterGradient,offsets)
    VIEW(ResidentParameterGradient,values) VIEW(ResidentParameterGradient,connected);
  }
  if(!alias_registered<ResidentBoundaryGradient>(m,"BoundaryGradient")) {
  py::class_<ResidentBoundaryGradient>(m,"BoundaryGradient")
    VIEW(ResidentBoundaryGradient,token) VIEW(ResidentBoundaryGradient,coordinates) VIEW(ResidentBoundaryGradient,values)
    VIEW(ResidentBoundaryGradient,valid) VIEW(ResidentBoundaryGradient,connected);
  }
  if(!alias_registered<ResidentOptimizerState>(m,"DeviceOptimizerState")) {
  py::class_<ResidentOptimizerState>(m,"DeviceOptimizerState").def(py::init<>())
    FIELD(ResidentOptimizerState,values) FIELD(ResidentOptimizerState,first) FIELD(ResidentOptimizerState,second)
    FIELD(ResidentOptimizerState,maximum) FIELD(ResidentOptimizerState,steps) FIELD(ResidentOptimizerState,corrections);
  }
  if(!alias_registered<ResidentTrainingCheckpoint>(m,"TrainingCheckpoint")) {
  py::class_<ResidentTrainingCheckpoint>(m,"TrainingCheckpoint").def(py::init<>())
    FIELD(ResidentTrainingCheckpoint,schema) FIELD(ResidentTrainingCheckpoint,generation) FIELD(ResidentTrainingCheckpoint,next_token)
    FIELD(ResidentTrainingCheckpoint,continuation) FIELD(ResidentTrainingCheckpoint,parameters) FIELD(ResidentTrainingCheckpoint,aliases)
    FIELD(ResidentTrainingCheckpoint,trainable) FIELD(ResidentTrainingCheckpoint,optimizer) FIELD(ResidentTrainingCheckpoint,groups)
    FIELD(ResidentTrainingCheckpoint,offsets) FIELD(ResidentTrainingCheckpoint,state)
    FIELD(ResidentTrainingCheckpoint,mode) FIELD(ResidentTrainingCheckpoint,zeta);
  }
  if(!alias_registered<ResidentStep>(m,"TrainingStep")) {
  py::class_<ResidentStep>(m,"TrainingStep")
    VIEW(ResidentStep,applied) VIEW(ResidentStep,refusal_code) VIEW(ResidentStep,generation);
  }
}
#undef FIELD
#undef VIEW
