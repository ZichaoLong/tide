// 0770593 allocation §2–6. Column-to-column region functions and adjoints.
#include "tide/weighted_columns.h"
#include "weighted_allocation_batch.h"
#include "weighted_sending_batch.h"
#include "weighted_placement.h"
#include "tide/operator_profile.h"
#include <ATen/record_function.h>

namespace tide::weighted {
RegionColumns region_columns(const RegionWeights& w,const TensorFields& fields,const RegionalConfig& c,
                             const Tensor& node_ids,std::optional<at::Device> selection){
  RECORD_FUNCTION("tide::stage::region_math", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::RegionMath);
  auto recv=fields.at("w"),caps=fields.at("mu"),scores=fields.at("score"),nodes=node_ids;
  detail::require(scores.dim()==2&&scores.size(0)>0&&scores.size(1)>0,"region columns require nonempty R,N");
  for(const auto& [name,t]:fields)detail::require(t.dim()>=2&&t.sizes().slice(0,2)==scores.sizes()
    &&t.device()==scores.device()&&t.scalar_type()==scores.scalar_type(),"region column metadata mismatch");
  detail::require(recv.dim()==2&&caps.dim()==2,"region w/mu columns must be scalar fields");
  if(c.selection_mode=="regional_total_zero"||c.hard())detail::require(nodes.defined()
    &&nodes.sizes()==scores.sizes()&&nodes.scalar_type()==at::kLong&&nodes.device()==scores.device(),
    "stable sending requires int64 R,N node IDs on descriptor device");
  if(c.region_score=="att"){
    auto weights=fields.at(c.query_weight=="received"?"w":"mu");
    auto x=(weights.unsqueeze(-1)*fields.at("x")).sum(1)/weights.sum(1).clamp_min(1).unsqueeze(1);
    auto h=at::silu(at::matmul(x,w.extra.at("query.in_weight").t())+w.extra.at("query.hidden_bias"));
    auto q=at::matmul(h,w.extra.at("query.out_weight").t())+w.extra.at("query.out_bias");
    scores=*c.alpha_loc*scores+*c.alpha_sel*(fields.at("key")*q.unsqueeze(1)).sum(-1)/std::sqrt(double(*c.selection_width));
  }
  auto received=recv.sum(1),mass=caps.sum(1);Tensor context,low,high,shift,values;
  if(c.region_control_mode=="context"){
    auto z=(recv.unsqueeze(-1)*fields.at("evidence")).sum(1)/received.clamp_min(1).unsqueeze(1);
    auto x=at::cat({z,received.log1p().unsqueeze(1),mass.log1p().unsqueeze(1)},1);
    auto h=at::silu(at::matmul(x,w.extra.at("head.matrix").t())+w.extra.at("head.hidden_bias"));
    context=at::matmul(h,w.extra.at("head.vector"))+w.extra.at("head.bias");
  }
  if(c.total()){
    auto rho=context.defined()?context.sigmoid():at::full_like(received,*c.total_fraction);
    low=received.clamp_max(1)+rho*(mass.clamp_max(*c.k)-received.clamp_max(1));high=low;
  }else if(!c.attention()&&c.selection_mode!="proportional"){
    shift=context.defined()?context:at::full_like(received,*c.eta);
    low=(c.regional()?received:mass).clamp_max(1);high=at::full_like(received,*c.k);
  }
  if(selection)for(auto* t:{&recv,&caps,&scores,&nodes,&received,&low,&high,&shift})
    if(t->defined())*t=placement_copy(*t,*selection,t->scalar_type());
  RegionColumns out;out.mode=c.selection_mode;
  if(c.selection_mode=="regional_total_zero")out.controls=batch::sorted_send(scores,caps,nodes,low);
  else if(c.attention())out.controls=batch::attention(scores,caps,recv,c.selection_mode,*c.allocation_base,
                                                     at::full_like(received,*c.allocation_temperature));
  else if(c.selection_mode=="proportional")out.controls=batch::proportional(caps,*c.k);
  else if(c.hard())out.controls=batch::sorted_send(scores-shift.unsqueeze(1),caps,nodes,at::zeros_like(received),*c.k);
  else{
    out.allocation=allocate_columns(scores,caps,low,high,at::full({},*c.allocation_temperature,scores.options()),
                                    *c.regularizer,c.ent_alpha,shift);
    out.controls=out.allocation->controls;
  }
  return out;
}
std::vector<std::string> region_column_dependencies(const RegionColumns& r){
  if(!r.allocation)return std::vector<std::string>(r.controls.size(0),r.mode);
  auto host=r.allocation->branch.to(at::kCPU).contiguous();auto data=host.const_data_ptr<Index>();
  std::vector<std::string> result;for(Index i=0;i<host.numel();++i)result.push_back(batch::branches.at(data[i]));return result;
}
std::vector<bool> region_column_differentiable(const RegionColumns& r){
  if(!r.allocation)return std::vector<bool>(r.controls.size(0),r.controls.requires_grad());
  auto host=r.allocation->branch.to(at::kCPU).contiguous();auto data=host.const_data_ptr<Index>();
  const auto& xs=r.allocation->inputs;std::vector<bool> result;
  for(Index i=0;i<host.numel();++i){
    const auto b=data[i];bool grad=false;
    if(b==batch::Caps)grad=xs[1].requires_grad();
    else if(b!=batch::Zero){
      for(size_t k:{size_t(0),size_t(1),(b==batch::Lower||b==batch::Equality)?size_t(2):size_t(3),size_t(4)})
        grad|=xs[k].requires_grad();
      grad|=xs[5].defined()&&xs[5].requires_grad();
    }
    result.push_back(grad);
  }
  return result;
}
std::vector<std::pair<Tensor,Tensor>> region_column_vjp_sources(const RegionColumns& r,const Tensor& bar,const Tensor& rows){
  std::vector<std::pair<Tensor,Tensor>> result;
  if(!r.allocation){if(r.controls.requires_grad())result.emplace_back(r.controls,bar);return result;}
  auto grads=allocation_columns_vjp(*r.allocation,bar,rows);
  for(size_t i=0;i<6;++i){const auto& t=r.allocation->inputs[i];
    if(t.defined()&&t.requires_grad()&&grads[i].defined())result.emplace_back(t,grads[i]);}
  return result;
}
}
