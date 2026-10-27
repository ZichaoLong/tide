#include "weighted_flow_parameters.h"
#include <ATen/core/grad_mode.h>
#include <limits>
namespace tide::device_online {
WeightedBanks::WeightedBanks(const std::map<std::string,at::Tensor>& p,std::string prefix,at::Device d)
  :parameters_(p),prefix_(std::move(prefix)),device_(d){}
at::Tensor WeightedBanks::bind(const std::string& role,const std::string& name,const std::vector<int64_t>& shape,int64_t offset){
  if(at::GradMode::is_enabled()||roles_.count(role))throw std::invalid_argument("duplicate/grad-enabled weighted parameter binding");
  const auto& x=parameters_.at(name);int64_t count=1;
  for(auto n:shape){if(n<1||count>std::numeric_limits<int64_t>::max()/n)throw std::invalid_argument("invalid parameter bank shape");count*=n;}
  if(!x.device().is_cpu()||x.scalar_type()!=at::kFloat||offset<0||offset>x.numel()||count>x.numel()-offset)
    throw std::invalid_argument("weighted bank requires explicit CPU FP32 parameter and exact slice");
  auto value=x.detach().reshape({-1}).narrow(0,offset,count).reshape(shape).to(device_).clone();
  roles_.emplace(role,WeightedBankRole{prefix_+name,value,x.sizes().vec(),offset});return value;
}
at::Tensor WeightedBanks::direct(const std::string& role,const std::string& name){
  auto shape=parameters_.at(name).sizes().vec();shape.insert(shape.begin(),1);return bind(role,name,shape);
}
WeightedKVParameters bind_weighted_kv_update(WeightedBanks& b){
  return {b.direct("upd.base","kernel.base"),b.direct("upd.left","kernel.adapter_u"),
    b.direct("upd.right","kernel.adapter_v"),b.direct("upd.summary","kernel.summary")};
}
WeightedKVQueryParameters bind_weighted_kv_query(WeightedBanks& b){
  return {b.direct("mem.query","kernel.read_q"),b.direct("mem.key","kernel.read_k"),b.direct("mem.value","kernel.read_v")};
}
WeightedKVFullParameters bind_weighted_kv_full(WeightedBanks& b){
  return {b.direct("full.query","full_program.content.query"),{b.direct("full.out","full_program.content.out"),
    b.direct("full.gate","full_program.content.gate"),b.direct("full.up","full_program.content.up"),b.direct("full.down","full_program.content.down")}};
}
WeightedRecurrentParameters bind_weighted_recurrent(WeightedBanks& b,const weighted::RecurrentConfig& c){
  WeightedRecurrentParameters result;
  for(const auto& [role,shape]:weighted_recurrent_parameter_shapes(1,c)){
    std::string name=role;int64_t offset=0;
    if(name=="conv.weight"||name=="conv.bias")name="kernel."+name;
    else {
      bool bias=name.size()>=5&&name.substr(name.size()-5)==".bias";
      name=name.substr(0,name.size()-(bias?5:7)); // .bias or .matrix, explicitly declared roles.
      if(name=="read.out.0"||name=="read.out.1"){
        offset=(name.back()-'0')*c.summary_width*c.value_width;name="kernel.read_out";
      }else if(name=="full.out.0"||name=="full.out.1"){
        offset=(name.back()-'0')*c.width*c.value_width;name="full_program.content.out";
      }else if(name.rfind("full.",0)==0)name="full_program.content."+name.substr(5);
      else if(name=="read.query")name="kernel.read_query";
      else if(name=="read.out")name="kernel.read_out";
      else name="kernel."+name;
      if(bias)name+="_bias";
    }
    result[role]=b.bind("recurrent."+role,name,shape,offset);
  }
  return result;
}
WeightedRegionParameters bind_weighted_region(WeightedBanks& b,const weighted::RegionalConfig& c,int64_t rows){
  WeightedRegionParameters p;
  auto head=[&](const std::string& role,const std::string& prefix,int64_t input,int64_t hidden,int64_t output,bool scalar){
    return WeightedHeadParameters{b.bind(role+".matrix",prefix+(scalar?".matrix":".in_weight"),{1,hidden,input}),
      b.bind(role+".hidden_bias",prefix+".hidden_bias",{1,hidden}),
      b.bind(role+".projection",prefix+(scalar?".vector":".out_weight"),{1,output,hidden}),
      b.bind(role+".bias",prefix+(scalar?".bias":".out_bias"),{1,output})};
  };
  if(c.region_control_mode=="context")p.context=head("region.context","head",*c.region_control_width+2,c.hidden_width,1,true);
  if(c.region_score=="att")p.query=head("region.query","query",*c.selection_width,*c.selection_width,*c.selection_width,false);
  const auto d=b.device();
  p.context_owners=at::zeros({rows},at::TensorOptions().dtype(at::kLong).device(d));p.query_owners=at::zeros_like(p.context_owners);return p;
}
}
