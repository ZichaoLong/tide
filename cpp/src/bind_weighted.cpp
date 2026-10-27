#include "tide/weighted.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

namespace py=pybind11;
void bind_weighted(py::module_& module){
  using namespace tide;using namespace tide::weighted;
#define FIELD(type,name) .def_readwrite(#name,&type::name)
  py::class_<KVConfig>(module,"WeightedKVConfig").def(py::init<>())
    FIELD(KVConfig,width) FIELD(KVConfig,summary_width) FIELD(KVConfig,adapter_rank)
    FIELD(KVConfig,heads) FIELD(KVConfig,ffn_width) FIELD(KVConfig,window) FIELD(KVConfig,memory_read)
    FIELD(KVConfig,beta_short) FIELD(KVConfig,epsilon) FIELD(KVConfig,alpha_att) FIELD(KVConfig,alpha_ff)
    FIELD(KVConfig,position_origin) FIELD(KVConfig,position_scale) FIELD(KVConfig,full_frequencies)
    FIELD(KVConfig,read_frequencies) FIELD(KVConfig,distance_slopes).def("validate",&KVConfig::validate);
  py::class_<RegionalConfig>(module,"WeightedRegionalConfig").def(py::init<>())
    FIELD(RegionalConfig,hidden_width) FIELD(RegionalConfig,region_control_width) FIELD(RegionalConfig,k)
    FIELD(RegionalConfig,gamma_max) FIELD(RegionalConfig,allocation_temperature)
    FIELD(RegionalConfig,total_fraction) FIELD(RegionalConfig,region_control_mode).def("validate",&RegionalConfig::validate);
  module.def("configure_weighted_node",[](NodeWeights& w,const KVConfig& c,const RegionalConfig& r,
                                         const std::vector<Index>& edges,const std::vector<Index>& inputs,double alpha){
    configure_node(w,c,r,SourceClasses{edges,inputs},alpha);
  });
  module.def("configure_weighted_region",&configure_region);
  module.def("weighted_allocate_quad",[](const Tensor& t,const Tensor& mu,const Tensor& lo,const Tensor& hi,const Tensor& tau){
    return allocate_quad(t,mu,lo,hi,tau).controls;
  });
#undef FIELD
}
