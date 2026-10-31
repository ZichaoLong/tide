// KV §4 / recurrent §3: only selected comparison snapshots, unscaled (c,G).
#include "weighted_query_batch.h"

namespace tide::weighted::query {
using namespace detail;
using packed::project;
namespace {
Tensor mix(const NodeWeights& w,const States& states,const Times& times,const Tensor& u,const KVConfig& c){
  Visible visible(states,times,c);std::vector<Tensor> result(states.size());
  auto queries=project(normalize(u,c.epsilon),weight(w,"full_program.content.query")).reshape({-1,c.heads,c.width/c.heads});
  const std::vector<double> empty;
  for(const auto& [length,rows]:visible.groups){
    work::packing::bucket("kv_visible_full",length,rows.size());
    const Index n=rows.size(),d=c.width/c.heads;auto ids=at::tensor(rows,u.options().dtype(at::kLong));
    auto keys=visible.slot(states,rows,"keys").reshape({n,length,c.heads,d});
    auto values=visible.slot(states,rows,"values").reshape_as(keys);
    auto weights=visible.slot(states,rows,"weights"),total=weights.sum(1);
    auto displacement=-time_values(visible.age(rows),u)/c.position_scale;
    std::vector<Tensor> reads;
    for(Index h=0;h<c.heads;++h){
      auto k=rotate(keys.select(2,h).reshape({-1,d}),displacement.flatten(),c.full_frequencies.empty()?empty:c.full_frequencies[h]).reshape({n,length,d});
      auto q=queries.index_select(0,ids).select(1,h);
      auto logits=(k*q.unsqueeze(1)).sum(-1)/std::sqrt(d);
      if(!c.distance_slopes.empty())logits=logits+c.distance_slopes[h]*displacement;
      reads.push_back(weighted_read(logits,values.select(2,h),weights,total));
    }
    auto combined=project(at::cat(reads,-1),weight(w,"full_program.content.out"));
    for(size_t i=0;i<rows.size();++i)result[rows[i]]=combined[i];
  }
  return c.alpha_att*at::stack(result);
}
Tensor mix(const NodeWeights& w,const States& states,const Times& times,const Tensor& u,const RecurrentConfig& c){
  const std::string p="full_program.content.";
  auto H=effective(states,times,c)[0];Tensor combined;
  if(c.basic()){
    auto q=project(normalize(u,c.epsilon),weight(w,p+"query"));
    auto reads=at::matmul(H,q.unsqueeze(1).unsqueeze(-1));
    combined=at::matmul(weight(w,p+"out"),reads).squeeze(-1).sum(1);
  }else{
    auto f=features(w,states,times,u,c);auto read=at::matmul(H,f.query.unsqueeze(-1)).squeeze(-1);
    if(c.memory_family=="gated_delta"){
      auto rms=scaled_rms(read,weight(w,p+"rho"),c.output_epsilon);
      auto gate=at::silu(project(normalize(u,c.epsilon),weight(w,p+"output_gate")));
      combined=project(rms.flatten(1)*gate,weight(w,p+"out"));
    }else{
      auto y=(read+weight(w,p+"direct").unsqueeze(-1)*f.value).flatten(1)*at::silu(f.gate);
      combined=project(scaled_rms(y,weight(w,p+"rho"),c.output_epsilon),weight(w,p+"out"));
    }
  }
  return c.alpha_mem*combined;
}
template<class Config> std::vector<FullResult> evaluate(const NodeWeights& w,const std::vector<FullInput>& requests,Index slots,const Options& options,const Config& c){
  require(options.mode=="hard","weighted Full uses exact VJP; legacy Emit mode must be hard");
  if(requests.empty())return {};
  States states;Times times;std::vector<Tensor> rows,controls;
  for(const auto& r:requests){states.push_back(*r.comparison);times.push_back(r.time);rows.push_back(r.content.value.slice(0,1));controls.push_back(r.control);}
  auto u=at::stack(rows),x=u+mix(w,states,times,u,c),normal=normalize(x,c.epsilon);
  auto fresh=x+c.alpha_ff*project(at::silu(project(normal,weight(w,"full_program.content.gate")))*
    project(normal,weight(w,"full_program.content.up")),weight(w,"full_program.content.down"));
  auto payload=at::cat({at::stack(controls).unsqueeze(-1),fresh},-1);
  std::vector<FullResult> result;
  for(size_t i=0;i<requests.size();++i){
    FullResult r{payload[i],{}};for(Index j=0;j<slots;++j)r.emitted.push_back({j,r.value});result.push_back(std::move(r));
  }
  return result;
}
}
std::vector<FullResult> full(const NodeWeights& w,const std::vector<FullInput>& r,Index slots,const Options& o,const KVConfig& c){return evaluate(w,r,slots,o,c);}
std::vector<FullResult> full(const NodeWeights& w,const std::vector<FullInput>& r,Index slots,const Options& o,const RecurrentConfig& c){return evaluate(w,r,slots,o,c);}
}
