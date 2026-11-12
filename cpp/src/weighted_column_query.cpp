// KV §3–4, architecture §3: complete B/L query tensors and selected Full.
#include "weighted_column_queries.h"
#include "weighted_describe_heads.h"
#include "weighted_kv_block.h"
#include <numeric>
namespace tide::weighted::columns {
using detail::require;using detail::weight;
Tensor rotate_queries(const Tensor& keys,const Tensor& displacement,const std::vector<double>& frequencies){
  if(frequencies.empty())return keys;
  return detail::rotate(keys.reshape({-1,keys.size(-1)}),displacement.flatten(),frequencies).reshape(keys.sizes());
}
QueryInputs query_inputs(const KVWindow& window,const Layout& layout,const TensorFields& bank,
                const std::vector<Index>& rows,bool before,Index horizon,bool summary){
  auto selected=window.layout.select(rows);auto device=window.content.device();
  auto [qi,reverse]=padded(selected,device);auto [ki,unused]=padded(layout,device);
  auto ids=indices(rows,device);QueryInputs q;
  q.u=gather(window.content.index_select(0,ids),qi);q.mass=gather(window.mass.index_select(0,ids),qi);q.reverse=reverse;
  auto times=gather(indices(selected.times,device),qi),stamps=gather(indices(layout.times,device),ki);
  q.age=times.unsqueeze(2)-stamps.unsqueeze(1);
  auto visible=(before?q.age>0:q.age>=0)&(ki.unsqueeze(1)<static_cast<Index>(layout.times.size()))
    &(qi.unsqueeze(2)<static_cast<Index>(rows.size()));
  if(horizon)visible=visible&(q.age<horizon);
  q.weights=at::where(visible,gather(bank.at("weights"),ki).unsqueeze(1),at::zeros({},q.u.options()));
  if(summary)q.raw["summaries"]=gather(bank.at("summaries"),ki);
  else for(const auto& name:{"keys","values"})q.raw[name]=gather(bank.at(name),ki);
  return q;
}
TensorFields read(const NodeWeights& w,const KVConfig& c,const RegionalConfig& r,const KVWindow& window,
                  const std::string& mode,const KVState& initial){
  require(supported(c),"unsupported KV column memory");
  require(mode=="content"||mode=="old"||mode=="proposal","invalid KV column Read snapshot");
  require(!window.layout.times.empty(),"Read columns require actual rows");
  std::vector<Index> rows(window.layout.times.size());std::iota(rows.begin(),rows.end(),0);
  auto q=query_inputs(window,mode=="content"?initial.layout:window.bank_layout,
                 mode=="content"?initial.fields:window.bank,rows,mode=="old",c.window,c.memory_read=="sum");
  auto sw=q.weights*detail::passive_decay(q.age.clamp_min(0),c.beta_short,q.u);
  auto st=sw.sum(-1),lt=q.weights.sum(-1);Tensor shortv,longv;
  if(c.memory_read=="sum"){
    shortv=at::matmul(sw,q.raw.at("summaries"))/st.clamp_min(1).unsqueeze(-1);
    longv=at::matmul(q.weights,q.raw.at("summaries"))/lt.clamp_min(1).unsqueeze(-1);
  }else if((q.weights>0).any().item<bool>()){
    auto query=project(normalize(q.u,c.epsilon),weight(w,"kernel.read_q")).unsqueeze(1);
    auto k=project(q.raw.at("keys"),weight(w,"kernel.read_k")).unsqueeze(1);
    auto v=project(q.raw.at("values"),weight(w,"kernel.read_v")).unsqueeze(1);
    if(!c.read_frequencies.empty()){
      auto displacement=-detail::time_values(q.age,q.u)/c.position_scale;
      auto keys=rotate_queries(k.select(1,0).unsqueeze(1).expand({-1,q.u.size(1),-1,-1}),displacement,c.read_frequencies);
      auto head=query.select(1,0).unsqueeze(2);
      if(head.scalar_type()==at::kHalf){head=head.to(at::kFloat);keys=keys.to(at::kFloat);}
      auto logits=(head*keys).sum(-1).unsqueeze(1)/std::sqrt(double(c.summary_width));
      shortv=logit_attention(logits,v,sw).select(1,0)*st.clamp_max(1).unsqueeze(-1);
      longv=logit_attention(logits,v,q.weights).select(1,0)*lt.clamp_max(1).unsqueeze(-1);
    }else{
      shortv=block::attention(query,k,v,sw).select(1,0)*st.clamp_max(1).unsqueeze(-1);
      longv=block::attention(query,k,v,q.weights).select(1,0)*lt.clamp_max(1).unsqueeze(-1);
    }
  }else shortv=longv=at::zeros({q.u.size(0),q.u.size(1),c.summary_width},q.u.options());
  auto cur=normalize(q.u,c.epsilon),current=project(cur,weight(w,"kernel.summary"));
  auto stat=at::stack({q.mass.log1p(),st.log1p(),lt.log1p()},-1);
  detail::Groups groups{cur,at::cat({shortv,longv},-1),stat};
  auto features=at::cat({cur,stat.slice(-1,0,1),shortv,longv,stat.slice(-1,1),current*shortv,current*longv},-1);
  auto fields=query::describe_columns(w,r,q.mass,features,groups,c.epsilon);
  for(auto& [name,t]:fields)t=t.flatten(0,1).index_select(0,q.reverse);
  return fields;
}
Message full(const NodeWeights& w,const KVConfig& c,const KVWindow& window,const Tensor& controls){
  require(supported(c),"unsupported KV column Full");
  require(controls.sizes()==window.mass.sizes()&&controls.device()==window.mass.device()
    &&controls.scalar_type()==window.mass.scalar_type()&&at::isfinite(controls).all().item<bool>()
    &&!(controls<0).any().item<bool>(),"invalid KV column controls");
  auto ids=(controls.detach()>0).nonzero().flatten();auto rows=host_ids(ids);
  Message out{rows,controls.index_select(0,ids),at::empty({0,c.width},window.content.options())};
  if(rows.empty())return out;
  auto q=query_inputs(window,window.bank_layout,window.bank,rows,false,c.window);
  const std::string p="full_program.content.";const Index b=q.u.size(0),length=q.u.size(1),dh=c.width/c.heads;
  Tensor mixed;
  if((q.weights>0).any().item<bool>()){
    auto query=project(normalize(q.u,c.epsilon),weight(w,p+"query")).reshape({b,length,c.heads,dh}).transpose(1,2);
    auto k=q.raw.at("keys").reshape({b,-1,c.heads,dh}).transpose(1,2);
    auto v=q.raw.at("values").reshape({b,-1,c.heads,dh}).transpose(1,2);
    bool positional=false;for(const auto& f:c.full_frequencies)positional|=!f.empty();
    for(auto slope:c.distance_slopes)positional|=slope!=0;
    if(positional){
      auto displacement=-detail::time_values(q.age,q.u)/c.position_scale;std::vector<Tensor> logits;
      const std::vector<double> empty;
      for(Index h=0;h<c.heads;++h){
        auto key=rotate_queries(k.select(1,h).unsqueeze(1).expand({-1,length,-1,-1}),displacement,
                               c.full_frequencies.empty()?empty:c.full_frequencies[h]);
        auto head=query.select(1,h).unsqueeze(2);
        if(head.scalar_type()==at::kHalf){head=head.to(at::kFloat);key=key.to(at::kFloat);}
        auto l=(head*key).sum(-1)/std::sqrt(double(dh));
        if(!c.distance_slopes.empty())l=l+c.distance_slopes[h]*displacement;logits.push_back(l);
      }
      mixed=logit_attention(at::stack(logits,1),v,q.weights);
    }else mixed=block::attention(query,k,v,q.weights);
    mixed=mixed.transpose(1,2).reshape({b,length,c.width});
  }else mixed=at::zeros_like(q.u);
  auto x=q.u+c.alpha_att*project(mixed*q.weights.sum(-1).clamp_max(1).unsqueeze(-1),weight(w,p+"out"));
  auto n=normalize(x,c.epsilon);
  auto y=x+c.alpha_ff*project(at::silu(project(n,weight(w,p+"gate")))*project(n,weight(w,p+"up")),weight(w,p+"down"));
  out.content=y.flatten(0,1).index_select(0,q.reverse);return out;
}
}
