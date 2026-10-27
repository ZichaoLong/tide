#include "weighted_flow_vjp.h"
#include "device_launch_tide_weighted_row_accumulate.h"
namespace tide::device_online::weighted_reverse {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(const at::Tensor& x,const at::Tensor& reference,at::ScalarType dtype,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=reference.device()||x.scalar_type()!=dtype||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid weighted stage adjoint shape/device/dtype");
}
}
WeightedRowAdjoint zeros(DeviceProgram& p,const at::Tensor& shape){
  WeightedRowAdjoint g{at::zeros(shape.sizes(),shape.options().dtype(at::kFloat)),at::zeros({shape.size(0)},shape.options().dtype(at::kBool))};
  p.zero(g.value);p.zero(g.connected);return g;
}
WeightedRawAdjoint zeros(DeviceProgram& p,const WeightedRawLayout& l,const at::Tensor& shape){
  WeightedRawAdjoint g{at::zeros(shape.sizes(),shape.options().dtype(at::kFloat)),at::zeros({shape.size(0),int64_t(l.fields().size())},shape.options().dtype(at::kBool))};
  p.zero(g.value);p.zero(g.connected);return g;
}
void add(DeviceProgram& p,const WeightedRowAdjoint& target,const WeightedRowAdjoint& source,
    const at::Tensor& owners,const at::Tensor& error){
  if(!source.value.defined()&&!source.connected.defined())return;
  const auto rows=target.value.size(0),width=target.value.numel()/rows;
  check(target.value,error,at::kFloat,target.value.sizes());check(source.value,error,at::kFloat,target.value.sizes());
  check(target.connected,error,at::kBool,{rows});check(source.connected,error,at::kBool,{rows});check(owners,error,at::kLong,{rows});
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_row_accumulate)(1,stream,
    ptr(owners),ptr(source.value),ptr(source.connected),ptr(target.value),ptr(target.connected),ptr(error),rows,width),
    "weighted stage independent-root accumulation");},{owners,source.value,source.connected,target.value,target.connected,error});
}
void add(DeviceProgram& p,const WeightedRawLayout& l,const WeightedRawAdjoint& target,
    const WeightedRawAdjoint& source,const at::Tensor& error){
  if(!source.value.defined()&&!source.connected.defined())return;
  const auto rows=target.value.size(0);check(source.value,error,at::kFloat,{rows,l.floating_width()});
  check(source.connected,error,at::kBool,{rows,int64_t(l.fields().size())});
  auto owners=at::zeros({rows},error.options().dtype(at::kLong));
  for(size_t j=0;j<l.fields().size();++j){const auto& f=l.fields()[j];const auto width=f.width*(f.ragged?l.capacity():1);
    auto value=weighted_columns(p,target.value,f.offset,width),flag=weighted_columns(p,target.connected,j,1).reshape({rows});
    add(p,{value,flag},{weighted_columns(p,source.value,f.offset,width),weighted_columns(p,source.connected,j,1).reshape({rows})},owners,error);
    weighted_copy(p,target.value.narrow(1,f.offset,width),value);weighted_copy(p,target.connected.select(1,j),flag);
  }
}
void parameter(DeviceProgram& p,std::vector<WeightedParameterPartial>& out,const WeightedBanks& banks,
    const std::string& name,const at::Tensor& value,const at::Tensor& connected){
  const auto& role=banks.roles().at(name);
  if(value.sizes()!=role.bank.sizes()||connected.sizes()!=at::IntArrayRef{1})throw std::invalid_argument("weighted role adjoint mismatch: "+name);
  auto original=at::zeros(role.original_shape,value.options());p.zero(original);
  weighted_copy(p,original.reshape({-1}).narrow(0,role.offset,value.numel()),value.reshape({-1}));
  out.push_back({role.parameter,original,connected});
}
void parameters(DeviceProgram& p,std::vector<WeightedParameterPartial>& out,const WeightedBanks& banks,
    const std::string& prefix,const WeightedFieldAdjoints& gradients){
  for(const auto& [name,g]:gradients)parameter(p,out,banks,prefix+name,g.value,g.connected);
}
void source(DeviceProgram& p,TaskServices& services,const WeightedFlowNode& n,const WeightedAggregateGradient& g,
    std::vector<WeightedParameterPartial>& out,const at::Tensor& error,const WeightedFlowBudget& remaining){
  if(!g.parameters)return;
  for(size_t j=0;j<n.source.adaptations.size();++j){const auto role="source."+std::to_string(j);const auto h=n.banks.roles().at(role+".a").bank.size(1);
    auto take=[&](const at::Tensor& x){auto y=at::zeros(x.sizes(),x.options());weighted_copy(p,y,x);return y;};
    WeightedSourceParameters part{take(g.parameters->a.narrow(0,j,1).narrow(1,0,h)),
      take(g.parameters->b.narrow(0,j,1).narrow(2,0,h)),take(g.parameters->embedding.narrow(0,j,1).narrow(1,0,h))};
    auto flags=take(g.owner_connected.narrow(0,j,1));
    if(n.source.adaptations[j]){auto& worker=services.begin();
      auto adapted=append_weighted_source_adaptation_vjp(worker,*n.source.adaptations[j],part,flags,error,remaining(worker));services.end(worker);
      part=adapted.base;flags=adapted.connected;
      parameter(p,out,n.banks,role+".a_left",adapted.factors.a_left,flags);parameter(p,out,n.banks,role+".a_right",adapted.factors.a_right,flags);
      parameter(p,out,n.banks,role+".b_left",adapted.factors.b_left,flags);parameter(p,out,n.banks,role+".b_right",adapted.factors.b_right,flags);
    }
    parameter(p,out,n.banks,role+".a",part.a,flags);parameter(p,out,n.banks,role+".b",part.b,flags);parameter(p,out,n.banks,role+".embedding",part.embedding,flags);
  }
}
}
