// Fixed 0770593 allocation §3.1–3.6: shared matrices over arbitrary row axes.
#include "weighted_describe_heads.h"
#include "weighted_packed.h"

namespace tide::weighted::query {
using namespace detail;
namespace {
Tensor project_columns(const Tensor& x,const Tensor& matrix){
  if(x.dim()==2)return packed::project(x,matrix);
  auto shape=x.sizes().vec();shape.back()=matrix.size(0);
  return packed::project(x.reshape({-1,x.size(-1)}),matrix).reshape(shape);
}
Tensor head(const NodeWeights& w,const std::string& p,const Tensor& x){
  auto hidden=at::silu(project_columns(x,weight(w,p+".matrix"))+weight(w,p+".hidden_bias"));
  return weight(w,p+".bias")+(hidden*weight(w,p+".vector")).sum(-1);
}
Tensor encode(const NodeWeights& w,const std::string& p,const Tensor& x,double epsilon){
  auto hidden=at::silu(project_columns(x,weight(w,p+".in_weight"))+weight(w,p+".hidden_bias"));
  return normalize(project_columns(hidden,weight(w,p+".out_weight"))+weight(w,p+".out_bias"),epsilon);
}
Groups encoded(const NodeWeights& w,const std::string& p,const Groups& g,double e){
  return {encode(w,p+".encoders.cur",g[0],e),encode(w,p+".encoders.mem",g[1],e),encode(w,p+".encoders.stat",g[2],e)};
}
Tensor score(const NodeWeights& w,const RegionalConfig& r,const Tensor& f,const Groups& g,double e){
  const std::string p="read_program.score";
  if(r.local_score=="cat")return head(w,p,f);
  auto v=encoded(w,p,g,e);auto joined=at::cat({v[0],v[1],v[2]},-1);
  auto gate=at::silu(project_columns(joined,weight(w,p+".gate"))+weight(w,p+".gate_bias"));
  auto up=project_columns(joined,weight(w,p+".up"))+weight(w,p+".up_bias");
  return weight(w,p+".bias")+*r.alpha_match*(v[0]*v[1]).sum(-1)/std::sqrt(static_cast<double>(*r.selection_width))
    +*r.alpha_res*(gate*up*weight(w,p+".vector")).sum(-1);
}
Tensor evidence(const NodeWeights& w,const RegionalConfig& r,const Tensor& f,const Groups& g,double e){
  if(r.region_control_mode=="fixed"){
    auto shape=f.sizes().vec();shape.back()=*r.region_control_width;return at::zeros(shape,f.options());
  }
  if(r.region_evidence=="cat")return normalize(project_columns(f,weight(w,"read_program.evidence_weight"))+weight(w,"read_program.evidence_bias"),e);
  const std::string p="read_program.evidence";auto v=encoded(w,p,g,e);
  auto result=project_columns(v[0],weight(w,p+".cur"))+project_columns(v[1],weight(w,p+".mem"))+project_columns(v[2],weight(w,p+".stat"));
  return normalize(result+project_columns(v[0]*v[1],weight(w,p+".interaction"))+weight(w,p+".bias"),e);
}
Tensor key(const NodeWeights& w,const RegionalConfig& r,const Groups& g,double e){
  if(r.region_key=="content")return encode(w,"read_program.key",g[1],e);
  return project_columns(at::cat({g[1],g[2].slice(-1,1)},-1),weight(w,"read_program.key_weight"))+weight(w,"read_program.key_bias");
}
}
TensorFields describe_columns(const NodeWeights& w,const RegionalConfig& r,const Tensor& mass,
    const Tensor& f,const Groups& g,double epsilon){
  auto recv=mass.clamp_max(1);Tensor mu;
  if(r.regional())mu=(r.gamma_max*recv).clamp_max(1);
  else {
    auto gamma=at::full({},r.fixed_gain.value_or(1),mass.options());
    if(r.send_gain=="context")gamma=1+(r.gamma_max-1)*head(w,"read_program.gain",f).sigmoid();
    else if(r.gain_logit)gamma=1+(r.gamma_max-1)*weight(w,"read_program.gain_logit").sigmoid();
    mu=(gamma*mass).clamp_max(1);
  }
  TensorFields fields{{"w",recv},{"mu",mu},{"score",score(w,r,f,g,epsilon)}};
  if(r.regional())fields["evidence"]=evidence(w,r,f,g,epsilon);
  if(r.region_score=="att"){
    fields["x"]=project_columns(g[0],weight(w,"read_program.current_weight"))+weight(w,"read_program.current_bias");
    fields["key"]=key(w,r,g,epsilon);
  }
  return fields;
}
}
