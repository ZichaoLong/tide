// Allocation §2–6: complete query/context/normalization/allocation batches.
#include "weighted_region_batch.h"
#include "weighted_sending_batch.h"
#include "weighted_detail.h"
#include <sstream>

namespace tide::weighted::batch {
std::vector<Selection> regions(const RegionWeights& w,const std::vector<RegionInput>& requests,const RegionalConfig& c,
                               std::vector<std::string>* dependencies){
  std::vector<Selection> result(requests.size());std::map<std::string,std::vector<size_t>> groups;
  if(dependencies)dependencies->resize(requests.size());
  for(size_t i=0;i<requests.size();++i){
    detail::require(!requests[i].candidates.empty(),"region batch requires nonempty complete candidates");
    const auto& first=requests[i].candidates[0].fields.at("w");std::ostringstream key;
    key<<requests[i].candidates.size()<<first.device()<<int(first.scalar_type());groups[key.str()].push_back(i);
  }
  for(const auto& [key,ids]:groups){
    std::map<std::string,Tensor> fields;
    for(const auto& [name,unused]:requests[ids[0]].candidates[0].fields){
      std::vector<Tensor> rows;
      for(auto i:ids){std::vector<Tensor> row;for(const auto& d:requests[i].candidates)row.push_back(d.fields.at(name));rows.push_back(at::stack(row));}
      fields[name]=at::stack(rows);
    }
    auto recv=fields.at("w"),caps=fields.at("mu"),scores=fields.at("score");std::vector<Tensor> node_rows;
    for(auto i:ids){std::vector<Index> row;for(const auto& d:requests[i].candidates)row.push_back(d.node);node_rows.push_back(at::tensor(row,recv.options().dtype(at::kLong)));}
    auto nodes=at::stack(node_rows);
    if(c.region_score=="att"){
      auto weights=fields.at(c.query_weight=="received"?"w":"mu");
      auto x=(weights.unsqueeze(-1)*fields.at("x")).sum(1)/weights.sum(1).clamp_min(1).unsqueeze(1);
      auto hidden=at::silu(at::matmul(x,w.extra.at("query.in_weight").t())+w.extra.at("query.hidden_bias"));
      auto query=at::matmul(hidden,w.extra.at("query.out_weight").t())+w.extra.at("query.out_bias");
      scores=*c.alpha_loc*scores+*c.alpha_sel*(fields.at("key")*query.unsqueeze(1)).sum(-1)/std::sqrt(static_cast<double>(*c.selection_width));
    }
    auto received=recv.sum(1),mass=caps.sum(1);Tensor context,low,high,shift,values;
    if(c.region_control_mode=="context"){
      auto z=(recv.unsqueeze(-1)*fields.at("evidence")).sum(1)/received.clamp_min(1).unsqueeze(1);
      auto x=at::cat({z,received.log1p().unsqueeze(1),mass.log1p().unsqueeze(1)},1);
      auto hidden=at::silu(at::matmul(x,w.extra.at("head.matrix").t())+w.extra.at("head.hidden_bias"));
      context=at::matmul(hidden,w.extra.at("head.vector"))+w.extra.at("head.bias");
    }
    if(c.total()){
      auto rho=context.defined()?context.sigmoid():at::full_like(received,*c.total_fraction);
      low=received.clamp_max(1)+rho*(mass.clamp_max(*c.k)-received.clamp_max(1));high=low;
      if(c.selection_mode=="regional_total_zero")values=sorted_send(scores,caps,nodes,low);
    }else if(c.attention())values=attention(scores,caps,recv,c.selection_mode,*c.allocation_base,at::full_like(received,*c.allocation_temperature));
    else if(c.selection_mode=="proportional")values=proportional(caps,*c.k);
    else{
      shift=context.defined()?context:at::full_like(received,*c.eta);
      if(c.hard())values=sorted_send(scores-shift.unsqueeze(1),caps,nodes,at::zeros_like(received),*c.k);
      low=(c.regional()?received:mass).clamp_max(1);high=at::full_like(received,*c.k);
    }
    std::vector<Tensor> rows;
    if(!values.defined()){
      std::vector<AllocationRow> inputs;
      for(size_t j=0;j<ids.size();++j)inputs.push_back({scores[j],caps[j],low[j],high[j],at::full({},*c.allocation_temperature,scores.options()),shift.defined()?shift[j]:Tensor{}});
      auto allocations=allocate_batch(inputs,*c.regularizer,c.ent_alpha);
      for(size_t j=0;j<ids.size();++j){rows.push_back(allocations[j].controls);if(dependencies)(*dependencies)[ids[j]]=allocations[j].constraint;}
    }else{
      rows=values.unbind();if(dependencies)for(auto i:ids)(*dependencies)[i]=c.selection_mode;
    }
    for(size_t j=0;j<ids.size();++j){
      auto i=ids[j];auto& out=result[i];out.history.last_time=requests[i].time;
      for(size_t k=0;k<requests[i].candidates.size();++k){
        auto node=requests[i].candidates[k].node;out.controls[node]=rows[j][k];if(rows[j][k].item<double>()>0)out.active.insert(node);
      }
    }
  }
  return result;
}
}
