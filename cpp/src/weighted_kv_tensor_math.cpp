// Fixed 0770593 KV §3–4 and allocation §3: group tensors are the numerical unit.
#include "weighted_kv_tensor.h"
#include "weighted_column_queries.h"
#include "tide/packing_work.h"
#include <cmath>

namespace tide::weighted::columns::tensor {
namespace {
struct Math {
  const Packed& p;
  const Tensor& at(const std::string& name)const{return p.values.at(name);}
  Tensor bias(const std::string& name)const{
    auto x=at(name);if(p.shared.count(name))return x;
    auto shape=x.sizes().vec();shape.insert(shape.begin()+1,2,1);return x.reshape(shape);
  }
  Tensor project(const Tensor& x,const std::string& name)const{
    const auto& w=at(name);const bool shared=p.shared.count(name);
    work::packing::projection(shared?"kv_tensor_shared":"kv_tensor",x.numel()/x.size(-1));
    if(shared)return at::matmul(x,w.t());
    auto shape=x.sizes().vec();shape.back()=w.size(1);
    return at::bmm(x.reshape({p.nodes,-1,x.size(-1)}),w.transpose(1,2)).reshape(shape);
  }
  Tensor head(const std::string& name,const Tensor& x)const{
    auto h=at::silu(project(x,name+".matrix")+bias(name+".hidden_bias"));
    return bias(name+".bias")+(h*bias(name+".vector")).sum(-1);
  }
  Tensor encode(const std::string& name,const Tensor& x,double e)const{
    auto h=at::silu(project(x,name+".in_weight")+bias(name+".hidden_bias"));
    return normalize(project(h,name+".out_weight")+bias(name+".out_bias"),e);
  }
  std::array<Tensor,3> encoded(const std::string& name,const std::array<Tensor,3>& x,double e)const{
    return {encode(name+".encoders.cur",x[0],e),encode(name+".encoders.mem",x[1],e),encode(name+".encoders.stat",x[2],e)};
  }
  Tensor heads(const Tensor& value,Index count)const{
    return value.reshape({p.nodes,p.batch,value.size(2),count,value.size(3)/count}).transpose(2,3);
  }
  Tensor rotate(const Tensor& value,const Tensor& position,const std::vector<std::vector<double>>& frequencies,double scale)const{
    if(!position.defined())return value;
    auto coordinates=detail::time_values(position,value)/scale;std::vector<Tensor> rotated;
    for(Index h=0;h<value.size(2);++h){auto x=value.select(2,h);
      rotated.push_back(rotate_queries(x,coordinates,frequencies[h]));}
    return at::stack(rotated,2);
  }
  Tensor logits(const Tensor& q,const Tensor& k,const Request& r)const{
    auto frequencies=r.full?r.config.full_frequencies:std::vector<std::vector<double>>{r.config.read_frequencies};
    auto query=rotate(q,p.query_position,frequencies,r.config.position_scale);
    auto key=rotate(k,p.key_position,frequencies,r.config.position_scale);
    auto value=at::matmul(query,key.transpose(-1,-2))/std::sqrt(double(q.size(-1)));
    bool distance=false;if(r.full)for(auto slope:r.config.distance_slopes)distance|=slope!=0;
    if(distance){auto displacement=-detail::time_values(p.age,q)/r.config.position_scale;
      auto slopes=detail::constants(r.config.distance_slopes,q).reshape({1,1,-1,1,1});value=value+slopes*displacement.unsqueeze(2);}
    return value;
  }
  Tensor attention(const Tensor& logits,const Tensor& values,const Tensor& weights)const{
    const Index batch=p.nodes*p.batch,heads=logits.size(2),dim=values.size(-1);
    return logit_attention(logits.reshape({batch,heads,p.queries,p.keys}),values.reshape({batch,heads,p.keys,dim}),
      weights.reshape({batch,p.queries,p.keys})).reshape({p.nodes,p.batch,heads,p.queries,dim}).transpose(2,3)
      .reshape({p.nodes,p.batch,p.queries,heads*dim});
  }
};
TensorFields describe(const Request& r,const Math& m,const Tensor& mass,const Tensor& f,const std::array<Tensor,3>& g){
  const auto& c=r.regional;const double e=r.config.epsilon;auto recv=mass.clamp_max(1);Tensor mu,score;
  if(c.regional())mu=(c.gamma_max*recv).clamp_max(1);
  else {
    auto gamma=at::full({},c.fixed_gain.value_or(1),mass.options());
    if(c.send_gain=="context")gamma=1+(c.gamma_max-1)*m.head("read_program.gain",f).sigmoid();
    else if(c.gain_logit)gamma=1+(c.gamma_max-1)*m.bias("read_program.gain_logit").sigmoid();
    mu=(gamma*mass).clamp_max(1);
  }
  const std::string s="read_program.score";
  if(c.local_score=="cat")score=m.head(s,f);
  else {auto v=m.encoded(s,g,e);auto joined=at::cat({v[0],v[1],v[2]},-1);
    auto gate=at::silu(m.project(joined,s+".gate")+m.bias(s+".gate_bias"));auto up=m.project(joined,s+".up")+m.bias(s+".up_bias");
    score=m.bias(s+".bias")+*c.alpha_match*(v[0]*v[1]).sum(-1)/std::sqrt(static_cast<double>(*c.selection_width))
      +*c.alpha_res*(gate*up*m.bias(s+".vector")).sum(-1);
  }
  TensorFields fields{{"w",recv},{"mu",mu},{"score",score}};
  if(c.regional()){
    if(c.region_control_mode=="fixed"){auto shape=f.sizes().vec();shape.back()=*c.region_control_width;fields["evidence"]=at::zeros(shape,f.options());}
    else if(c.region_evidence=="cat")fields["evidence"]=normalize(m.project(f,"read_program.evidence_weight")+m.bias("read_program.evidence_bias"),e);
    else {const std::string name="read_program.evidence";auto v=m.encoded(name,g,e);
      fields["evidence"]=normalize(m.project(v[0],name+".cur")+m.project(v[1],name+".mem")+m.project(v[2],name+".stat")
        +m.project(v[0]*v[1],name+".interaction")+m.bias(name+".bias"),e);}
  }
  if(c.region_score=="att"){
    fields["x"]=m.project(g[0],"read_program.current_weight")+m.bias("read_program.current_bias");
    fields["key"]=c.region_key=="content"?m.encode("read_program.key",g[1],e):
      m.project(at::cat({g[1],g[2].slice(-1,1)},-1),"read_program.key_weight")+m.bias("read_program.key_bias");
  }
  return fields;
}
}
TensorFields formula(const Request& r,const Packed& p){
  const auto& c=r.config;Math m{p};const auto& u=m.at("$u");
  auto weights=at::where(p.visible,m.at("$weights").unsqueeze(-2),at::zeros({},u.options()));
  if(r.full){
    Tensor mixed=at::zeros_like(u);
    if(r.active){auto q=m.heads(m.project(normalize(u,c.epsilon),"full_program.content.query"),c.heads),k=m.heads(m.at("$keys"),c.heads);
      mixed=m.attention(m.logits(q,k,r),m.heads(m.at("$values"),c.heads),weights);}
    const std::string prefix="full_program.content.";
    auto x=u+c.alpha_att*m.project(mixed*weights.sum(-1).clamp_max(1).unsqueeze(-1),prefix+"out");
    auto n=normalize(x,c.epsilon);
    auto y=x+c.alpha_ff*m.project(at::silu(m.project(n,prefix+"gate"))*m.project(n,prefix+"up"),prefix+"down");
    return {{"content",y}};
  }
  auto sw=weights*detail::passive_decay(p.age.clamp_min(0),c.beta_short,u),st=sw.sum(-1),lt=weights.sum(-1);
  auto cur=normalize(u,c.epsilon);Tensor shortv,longv;
  if(c.memory_read=="sum"){
    shortv=at::matmul(sw,m.at("$summaries"))/st.clamp_min(1).unsqueeze(-1);
    longv=at::matmul(weights,m.at("$summaries"))/lt.clamp_min(1).unsqueeze(-1);
  }else if(r.active){
    auto q=m.heads(m.project(cur,"kernel.read_q"),1),k=m.heads(m.project(m.at("$keys"),"kernel.read_k"),1),v=m.heads(m.project(m.at("$values"),"kernel.read_v"),1);
    auto logits=m.logits(q,k,r);
    shortv=m.attention(logits,v,sw)*st.clamp_max(1).unsqueeze(-1);longv=m.attention(logits,v,weights)*lt.clamp_max(1).unsqueeze(-1);
  }else shortv=longv=at::zeros({p.nodes,p.batch,p.queries,c.summary_width},u.options());
  auto current=m.project(cur,"kernel.summary");
  auto stat=at::stack({m.at("$mass").log1p(),st.log1p(),lt.log1p()},-1);
  std::array<Tensor,3> groups{cur,at::cat({shortv,longv},-1),stat};
  auto features=at::cat({cur,stat.slice(-1,0,1),shortv,longv,stat.slice(-1,1),current*shortv,current*longv},-1);
  return describe(r,m,m.at("$mass"),features,groups);
}
}
