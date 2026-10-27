#include "tide/weighted.h"
#include "tide/weighted_sending.h"
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
  py::class_<RegionalConfig>(module,"WeightedRegionalConfig").def(py::init<>()).def(py::init<std::string>())
    FIELD(RegionalConfig,hidden_width) FIELD(RegionalConfig,region_control_width) FIELD(RegionalConfig,k)
    FIELD(RegionalConfig,gamma_max) FIELD(RegionalConfig,allocation_temperature)
    FIELD(RegionalConfig,total_fraction) FIELD(RegionalConfig,region_control_mode)
    FIELD(RegionalConfig,regularizer) FIELD(RegionalConfig,ent_alpha) FIELD(RegionalConfig,selection_mode)
    FIELD(RegionalConfig,region_evidence) FIELD(RegionalConfig,send_gain) FIELD(RegionalConfig,fixed_gain)
    FIELD(RegionalConfig,gain_logit) FIELD(RegionalConfig,eta) FIELD(RegionalConfig,allocation_base).def("validate",&RegionalConfig::validate);
  py::class_<RecurrentConfig>(module,"WeightedRecurrentConfig").def(py::init<>())
    FIELD(RecurrentConfig,memory_family) FIELD(RecurrentConfig,memory_read)
    FIELD(RecurrentConfig,width) FIELD(RecurrentConfig,summary_width) FIELD(RecurrentConfig,key_width)
    FIELD(RecurrentConfig,value_width) FIELD(RecurrentConfig,ffn_width) FIELD(RecurrentConfig,heads)
    FIELD(RecurrentConfig,groups) FIELD(RecurrentConfig,matrix_rates) FIELD(RecurrentConfig,conv_size)
    FIELD(RecurrentConfig,conv_stride) FIELD(RecurrentConfig,lambda_short) FIELD(RecurrentConfig,epsilon)
    FIELD(RecurrentConfig,alpha_mem) FIELD(RecurrentConfig,alpha_ff) FIELD(RecurrentConfig,qk_epsilon)
    FIELD(RecurrentConfig,output_epsilon).def("validate",&RecurrentConfig::validate);
  module.def("configure_weighted_node",[](NodeWeights& w,const KVConfig& c,const RegionalConfig& r,
                                         const std::vector<Index>& edges,const std::vector<Index>& inputs,double alpha){
    configure_node(w,c,r,SourceClasses{edges,inputs},alpha);
  });
  module.def("configure_weighted_recurrent_node",[](NodeWeights& w,const RecurrentConfig& c,const RegionalConfig& r,
                                         const std::vector<Index>& edges,const std::vector<Index>& inputs,double alpha){
    configure_node(w,c,r,SourceClasses{edges,inputs},alpha);
  });
  module.def("configure_weighted_region",&configure_region);
  module.def("weighted_allocate_quad",[](const Tensor& t,const Tensor& mu,const Tensor& lo,const Tensor& hi,const Tensor& tau){
    return allocate_quad(t,mu,lo,hi,tau).controls;
  });
  module.def("weighted_allocate",[](const Tensor& t,const Tensor& mu,const Tensor& lo,const Tensor& hi,const Tensor& tau,
                                    const std::string& regularizer,std::optional<double> alpha){
    return allocate(t,mu,lo,hi,tau,regularizer,alpha).controls;
  });
  module.def("weighted_attention_shares",&attention_shares);
  module.def("weighted_attention_send",&attention_send);
  module.def("weighted_proportional",&proportional);
  module.def("weighted_total_zero",&total_zero);
  module.def("weighted_hard_selection",&hard_selection);
#undef FIELD
}
