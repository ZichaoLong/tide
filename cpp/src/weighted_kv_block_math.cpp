#include "weighted_kv_block.h"
#include "weighted_describe_heads.h"
namespace tide::weighted::block {
namespace {
Tensor gather(const Tensor& rows,const Tensor& indices){
  auto shape=indices.sizes().vec();shape.insert(shape.end(),rows.sizes().begin()+1,rows.sizes().end());
  auto zero_shape=rows.sizes().vec();zero_shape[0]=1;
  return at::cat({rows,at::zeros(zero_shape,rows.options())}).index_select(0,indices.flatten()).reshape(shape);
}
TensorFields banks(const Plan& p,const std::vector<TensorFields>& old,const Tensor& values,
    const Tensor& summaries,const Tensor& contents){
  std::map<std::string,std::vector<Tensor>> fields;Index offset=0;
  for(size_t b=0;b<old.size();++b){
    const Index n=p.banks[b]->values.size();auto kv=values.slice(0,offset,offset+n).chunk(2,-1);
    TensorFields fresh{{"keys",kv[0]},{"values",kv[1]},{"summaries",summaries.slice(0,offset,offset+n)},
      {"weights",contents.slice(0,offset,offset+n).select(1,0).clamp_max(1)}};
    for(const auto& [name,t]:fresh){auto joined=at::cat({old[b].at(name),t});auto shape=joined.sizes().vec();shape[0]=p.length-joined.size(0);
      fields[name].push_back(at::cat({joined,at::zeros(shape,joined.options())}));}
    offset+=n;
  }
  TensorFields result;for(const auto& [name,rows]:fields)result[name]=at::stack(rows);return result;
}
Tensor project(const Tensor& x,const Tensor& matrix){return at::matmul(x,matrix.t());}
}
TensorFields compute(const NodeWeights& w,const KVConfig& c,const RegionalConfig& r,const Plan& p,
    const std::vector<TensorFields>& old,const Tensor& values,const Tensor& summaries,const Tensor& contents,
    const Tensor& payload,const Tensor& controls){
  using detail::weight;
  auto raw=banks(p,old,values,summaries,contents);auto h=gather(payload,p.indices),mass=h.select(-1,0),u=h.slice(-1,1);
  auto writes=at::where(p.visible,raw.at("weights").unsqueeze(1),at::zeros({},h.options()));
  auto used=(writes>0).any(1).unsqueeze(-1),total=writes.sum(-1);
  auto k=at::where(used,raw.at("keys"),at::zeros({},h.options()));
  auto v=at::where(used,raw.at("values"),at::zeros({},h.options()));
  TensorFields fields;
  if(!p.full){
    auto shortw=writes*detail::passive_decay(p.ages,c.beta_short,u),short_total=shortw.sum(-1);
    Tensor short_value,long_value;
    if(c.memory_read=="qry"){
      auto q=project(normalize(u,c.epsilon),weight(w,"kernel.read_q")).unsqueeze(1);
      k=project(k,weight(w,"kernel.read_k")).unsqueeze(1);v=project(v,weight(w,"kernel.read_v")).unsqueeze(1);
      short_value=attention(q,k,v,shortw).select(1,0)*short_total.clamp_max(1).unsqueeze(-1);
      long_value=attention(q,k,v,writes).select(1,0)*total.clamp_max(1).unsqueeze(-1);
    }else{
      auto summary=at::where(used,raw.at("summaries"),at::zeros({},h.options()));
      short_value=at::matmul(shortw,summary)/short_total.clamp_min(1).unsqueeze(-1);
      long_value=at::matmul(writes,summary)/total.clamp_min(1).unsqueeze(-1);
    }
    auto cur=normalize(u,c.epsilon),current=project(cur,weight(w,"kernel.summary"));
    auto stat=at::stack({mass.log1p(),short_total.log1p(),total.log1p()},-1);
    detail::Groups groups{cur,at::cat({short_value,long_value},-1),stat};
    auto features=at::cat({cur,stat.slice(-1,0,1),short_value,long_value,stat.slice(-1,1),current*short_value,current*long_value},-1);
    fields=query::describe_columns(w,r,mass,features,groups,c.epsilon);
  }else{
    const std::string prefix="full_program.content.";const Index b=u.size(0),qcount=u.size(1),d=c.width/c.heads;
    auto q=project(normalize(u,c.epsilon),weight(w,prefix+"query")).reshape({b,qcount,c.heads,d}).transpose(1,2);
    k=k.reshape({b,-1,c.heads,d}).transpose(1,2);v=v.reshape({b,-1,c.heads,d}).transpose(1,2);
    auto mix=attention(q,k,v,writes).transpose(1,2).reshape({b,qcount,c.width})*total.clamp_max(1).unsqueeze(-1);
    auto x=u+c.alpha_att*project(mix,weight(w,prefix+"out")),normal=normalize(x,c.epsilon);
    auto fresh=x+c.alpha_ff*project(at::silu(project(normal,weight(w,prefix+"gate")))*project(normal,weight(w,prefix+"up")),weight(w,prefix+"down"));
    fields["value"]=at::cat({gather(controls,p.indices).unsqueeze(-1),fresh},-1);
  }
  for(auto& [name,t]:fields)t=t.flatten(0,1).index_select(0,p.reverse);
  return fields;
}
}
