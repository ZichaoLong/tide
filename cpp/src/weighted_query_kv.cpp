// KV Mem §2/3: ragged visible snapshots, positive weighted attention, int64 ages.
#include "weighted_query_batch.h"

namespace tide::weighted::query {
using namespace detail;
using packed::project;
Visible::Visible(const States& states,const Times& times,const KVConfig& c){
  for(size_t i=0;i<states.size();++i){
    auto [mask,age]=visible(states[i],times[i],c);masks.push_back(mask);ages.push_back(age.index({mask}));
    groups[ages.back().numel()].push_back(i);
  }
}
Tensor Visible::slot(const States& states,const std::vector<Index>& rows,const std::string& name) const {
  std::vector<Tensor> result;
  for(auto i:rows)result.push_back(states[i].slots.at(name).index({masks[i]}));return at::stack(result);
}
Tensor Visible::age(const std::vector<Index>& rows) const {
  std::vector<Tensor> result;for(auto i:rows)result.push_back(ages[i]);return at::stack(result);
}
Tensor weighted_read(const Tensor& logits,const Tensor& values,const Tensor& weights,const Tensor& totals){
  std::vector<Tensor> selected,result(weights.size(0));std::map<Index,std::vector<Index>> groups;
  for(Index i=0;i<weights.size(0);++i){selected.push_back(at::nonzero(weights[i]>0).flatten());groups[selected.back().numel()].push_back(i);}
  for(const auto& [length,rows]:groups){
    if(!length){for(auto i:rows)result[i]=at::zeros({values.size(-1)},values.options());continue;}
    std::vector<Tensor> columns;for(auto i:rows)columns.push_back(selected[i]);
    auto ids=at::tensor(rows,weights.options().dtype(at::kLong)),cols=at::stack(columns);
    auto l=logits.index({ids.unsqueeze(-1),cols}),w=weights.index({ids.unsqueeze(-1),cols});
    auto v=values.index({ids.unsqueeze(-1),cols});
    auto reads=totals.index_select(0,ids).clamp_max(1).unsqueeze(-1)*attention_mean(l+w.log(),v);
    for(size_t i=0;i<rows.size();++i)result[rows[i]]=reads[i];
  }
  return at::stack(result);
}
std::array<Tensor,4> memory(const NodeWeights& w,const States& states,const Times& times,const Tensor& u,const KVConfig& c){
  Visible visible(states,times,c);std::vector<std::array<Tensor,4>> result(states.size());
  for(const auto& [length,rows]:visible.groups){
    const Index n=rows.size();auto age=visible.age(rows).to(u.scalar_type());
    auto writes=visible.slot(states,rows,"weights");
    std::array<Tensor,2> weights{writes*at::exp(-c.beta_short*age),writes};
    std::array<Tensor,4> values;Tensor logits,projected;
    if(c.memory_read=="qry"){
      auto ids=at::tensor(rows,u.options().dtype(at::kLong));
      auto q=project(normalize(u.index_select(0,ids),c.epsilon),weight(w,"kernel.read_q"));
      auto k=project(visible.slot(states,rows,"keys").reshape({-1,c.width}),weight(w,"kernel.read_k"));
      k=rotate(k,(-age/c.position_scale).flatten(),c.read_frequencies).reshape({n,length,c.summary_width});
      logits=(k*q.unsqueeze(1)).sum(-1)/std::sqrt(c.summary_width);
      projected=project(visible.slot(states,rows,"values").reshape({-1,c.width}),weight(w,"kernel.read_v")).reshape({n,length,c.summary_width});
    }
    for(int j=0;j<2;++j){
      values[j+2]=weights[j].sum(1);
      values[j]=c.memory_read=="sum"?(weights[j].unsqueeze(-1)*visible.slot(states,rows,"summaries")).sum(1)/values[j+2].clamp_min(1).unsqueeze(-1)
                                    :weighted_read(logits,projected,weights[j],values[j+2]);
    }
    for(size_t i=0;i<rows.size();++i)for(int j=0;j<4;++j)result[rows[i]][j]=values[j][i];
  }
  std::array<Tensor,4> batch;
  for(int j=0;j<4;++j){std::vector<Tensor> rows;for(const auto& r:result)rows.push_back(r[j]);batch[j]=at::stack(rows);}
  return batch;
}
}
