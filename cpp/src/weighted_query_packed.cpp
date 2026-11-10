#include "tide/operator_profile.h"
#include <ATen/record_function.h>
// Batched raw snapshots with exact support; independent VJP roots live above.
#include "weighted_query_layout.h"
#include <limits>
namespace tide::weighted::query {
using namespace detail;
using packed::project;
namespace {
Tensor read(const Tensor& logits,const Tensor& values,const Tensor& weights,const Tensor& total){
  auto positive=weights>0,active=positive.any(-1);
  if(!active.any().item<bool>())return at::zeros({values.size(0),values.size(-1)},values.options());
  auto safe=at::where(positive,weights,at::ones_like(weights));
  auto masked=(logits+safe.log()).masked_fill(~positive,-std::numeric_limits<double>::infinity());
  masked=at::where(active.unsqueeze(-1),masked,at::zeros_like(masked));
  // Match scalar first-positive centering, including when the first stored
  // value is expired/zero-weight. Padding cannot alter the VJP anchor.
  auto first=positive.to(at::kLong).argmax(-1).reshape({-1,1,1});
  auto anchor=values.gather(1,first.expand({-1,1,values.size(-1)}));
  auto safe_values=at::where(positive.unsqueeze(-1),values,anchor);
  auto value=attention_mean(masked,safe_values)*total.clamp_max(1).unsqueeze(-1);
  return at::where(active.unsqueeze(-1),value,at::zeros_like(value));
}
SnapshotLayout layout(const States& states,const Times& times,const std::vector<Index>& ids,Index window){
  States s;Times t;for(auto i:ids){s.push_back(states[i]);t.push_back(times[i]);}
  return SnapshotLayout(std::move(s),t,window);
}
}
std::array<Tensor,4> packed_memory(const NodeWeights& w,const States& states,const Times& times,const Tensor& u,const KVConfig& c){
  RECORD_FUNCTION("tide::stage::read_memory", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::ReadMemory);
  std::vector<std::array<Tensor,4>> result(states.size());
  auto queries=c.memory_read=="qry"?project(normalize(u,c.epsilon),weight(w,"kernel.read_q")):Tensor{};
  for(const auto& ids:snapshot_chunks(states,c.width)){
    auto packed=layout(states,times,ids,c.window);auto long_w=packed.weights();
    std::array<Tensor,2> weights{long_w*passive_decay(packed.age,c.beta_short,u),long_w};
    std::array<Tensor,4> values;Tensor logits,projected,summaries;
    if(c.memory_read=="qry"){
      auto k=packed.project_slot("keys",weight(w,"kernel.read_k")).reshape({-1,c.summary_width});
      k=rotate(k,(-time_values(packed.age,u)/c.position_scale).flatten(),c.read_frequencies).reshape({packed.rows,packed.length,c.summary_width});
      auto q=queries.index_select(0,at::tensor(ids,u.options().dtype(at::kLong)));
      logits=(k*q.unsqueeze(1)).sum(-1)/std::sqrt(c.summary_width);
      projected=packed.project_slot("values",weight(w,"kernel.read_v"));
    }else summaries=packed.visible_slot("summaries");
    for(int j=0;j<2;++j){values[j+2]=weights[j].sum(1);
      values[j]=c.memory_read=="qry"?read(logits,projected,weights[j],values[j+2]):
        (weights[j].unsqueeze(-1)*summaries).sum(1)/values[j+2].clamp_min(1).unsqueeze(-1);}
    std::array<std::vector<Tensor>,4> rows;
    for(int k=0;k<4;++k)rows[k]=values[k].unbind();
    for(size_t j=0;j<ids.size();++j)for(int k=0;k<4;++k)result[ids[j]][k]=rows[k][j];
  }
  std::array<Tensor,4> output;
  for(int j=0;j<4;++j){std::vector<Tensor> rows;for(const auto& r:result)rows.push_back(r[j]);output[j]=at::stack(rows);}
  return output;
}
Tensor packed_full(const NodeWeights& w,const States& states,const Times& times,const Tensor& u,const KVConfig& c){
  RECORD_FUNCTION("tide::stage::full_memory", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::FullMemory);
  std::vector<Tensor> result(states.size());const Index d=c.width/c.heads;
  auto queries=project(normalize(u,c.epsilon),weight(w,"full_program.content.query")).reshape({-1,c.heads,d});
  const std::vector<double> empty;
  for(const auto& ids:snapshot_chunks(states,c.width)){
    auto packed=layout(states,times,ids,c.window);auto weights=packed.weights(),total=weights.sum(1);
    auto keys=packed.visible_slot("keys").reshape({packed.rows,packed.length,c.heads,d}),values=packed.visible_slot("values").reshape_as(keys);
    auto q=queries.index_select(0,at::tensor(ids,u.options().dtype(at::kLong)));
    auto displacement=-time_values(packed.age,u)/c.position_scale;std::vector<Tensor> reads;
    for(Index h=0;h<c.heads;++h){
      auto k=rotate(keys.select(2,h).reshape({-1,d}),displacement.flatten(),c.full_frequencies.empty()?empty:c.full_frequencies[h]).reshape({packed.rows,packed.length,d});
      auto logits=(k*q.select(1,h).unsqueeze(1)).sum(-1)/std::sqrt(d);
      if(!c.distance_slopes.empty())logits=logits+c.distance_slopes[h]*displacement;
      reads.push_back(read(logits,values.select(2,h),weights,total));
    }
    auto rows=at::cat(reads,-1).unbind();for(size_t j=0;j<ids.size();++j)result[ids[j]]=rows[j];
  }
  return project(at::stack(result),weight(w,"full_program.content.out"));
}
}
