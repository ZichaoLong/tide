#include "weighted_flow_internal.h"
#include <set>
namespace tide::device_online {
WeightedFlowSource bind_weighted_flow_source(DeviceProgram& p,WeightedBanks& banks,const Graph& g,int64_t node,
    const weighted::NodeBinding& binding,const at::Tensor& error,int64_t budget){
  WeightedFlowSource out;const auto d=g.nodes.size()?std::visit([](const auto& c){return c.width;},binding.memory):0;
  auto held=[&](){std::vector<at::Tensor> xs{out.classes,out.values.a,out.values.b,out.values.embedding};
    for(const auto& [_,role]:banks.roles())xs.push_back(role.bank);return xs;};
  const auto before=p.retained_tensor_bytes(held());
  auto remaining=[&](){const auto used=p.retained_tensor_bytes(held())-before;
    if(used>budget)throw std::invalid_argument("weighted source tensor budget exceeded");return budget-used;};
  const long double table_bytes=40.L*std::max<size_t>(1,g.inputs.size()+g.edges.size());
  if(table_bytes>budget)throw std::invalid_argument("weighted source classes exceed tensor budget before allocation");
  std::vector<int64_t> table(5*std::max<size_t>(1,g.inputs.size()+g.edges.size()),-1);std::map<int64_t,int64_t> labels;
  auto add=[&](int64_t physical,int64_t label,int64_t slot,int64_t kind,int64_t id,int64_t stride){
    if(!labels.count(label))labels[label]=labels.size();const int64_t data[]{labels.at(label),slot,kind,id,stride};
    for(int j=0;j<5;++j)table[5*physical+j]=data[j];
  };
  for(size_t i=0;i<g.inputs.size();++i)if(g.inputs[i]==node)add(i,binding.sources.inputs[i],g.source_domain->input[i],0,i,1);
  for(size_t i=0;i<g.edges.size();++i)if(g.edges[i].target==node){
    if(!g.origins.empty()&&g.origin_index[i]>=0){const auto& origin=g.origins[g.origin_index[i]];
      add(g.inputs.size()+i,binding.sources.inputs[origin.port],g.source_domain->edge_target[i],0,origin.port,origin.stride);
    }else add(g.inputs.size()+i,binding.sources.edges[i],g.source_domain->edge_target[i],1,i,1);
  }
  // Node planner needs only source domains, independent of whether alpha=0.
  out.classes=at::tensor(table,at::kLong).reshape({-1,5}).to(banks.device());
  if(binding.aggregate_alpha==0)return out;
  const auto& params=banks.parameters();int64_t hidden=1;
  for(const auto& [label,_]:labels)hidden=std::max(hidden,params.at("aggregate_program.transforms."+std::to_string(label)+".a").size(0));
  auto opts=at::TensorOptions().dtype(at::kFloat).device(banks.device());const auto count=std::max<int64_t>(1,labels.size());
  if(4.L*count*(2.L*hidden*d+hidden)>remaining())throw std::invalid_argument("weighted source padding budget exceeded");
  out.values={at::zeros({count,hidden,d},opts),at::zeros({count,d,hidden},opts),at::zeros({count,hidden},opts)};
  out.adaptations.resize(count);
  for(const auto& [label,index]:labels){
    const auto prefix="aggregate_program.transforms."+std::to_string(label),role="source."+std::to_string(index);
    banks.limit_bytes(remaining());
    WeightedSourceParameters base{banks.direct(role+".a",prefix+".a"),banks.direct(role+".b",prefix+".b"),banks.direct(role+".embedding",prefix+".embedding")};
    auto effective=base;
    if(params.count(prefix+".a_left")){
      WeightedSourceFactors factors{banks.direct(role+".a_left",prefix+".a_left"),banks.direct(role+".a_right",prefix+".a_right"),
        banks.direct(role+".b_left",prefix+".b_left"),banks.direct(role+".b_right",prefix+".b_right")};
      out.adaptations[index]=append_weighted_source_adaptation(p,base,factors,error,remaining());effective=out.adaptations[index]->effective;
    }
    const auto h=base.a.size(1);
    weighted_copy(p,out.values.a.narrow(0,index,1).narrow(1,0,h),effective.a);
    weighted_copy(p,out.values.b.narrow(0,index,1).narrow(2,0,h),effective.b);
    weighted_copy(p,out.values.embedding.narrow(0,index,1).narrow(1,0,h),effective.embedding);
  }
  return out;
}
}
