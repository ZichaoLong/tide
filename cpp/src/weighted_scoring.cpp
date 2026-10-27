// Independent C++ equations (1–5), pinned 0770593. No state or scheduling.
#include "weighted_scoring.h"

namespace tide::weighted::detail {
Tensor encode_group(const NodeWeights& w,const std::string& p,const Tensor& x,double epsilon){
  auto hidden=at::silu(at::matmul(weight(w,p+".in_weight"),x)+weight(w,p+".hidden_bias"));
  return normalize(at::matmul(weight(w,p+".out_weight"),hidden)+weight(w,p+".out_bias"),epsilon);
}
namespace {
Groups encoded(const NodeWeights& w,const std::string& p,const Groups& g,double e){
  return {encode_group(w,p+".encoders.cur",g[0],e),encode_group(w,p+".encoders.mem",g[1],e),encode_group(w,p+".encoders.stat",g[2],e)};
}
void encoder_shape(const NodeWeights& w,const std::string& p,Index input,Index hidden,Index output){
  shape(w,p+".in_weight",{hidden,input});shape(w,p+".hidden_bias",{hidden});
  shape(w,p+".out_weight",{output,hidden});shape(w,p+".out_bias",{output});
}
void group_shape(const NodeWeights& w,const std::string& p,Index d,Index ds,const RegionalConfig& r){
  encoder_shape(w,p+".encoders.cur",d,r.hidden_width,*r.selection_width);
  encoder_shape(w,p+".encoders.mem",2*ds,r.hidden_width,*r.selection_width);
  encoder_shape(w,p+".encoders.stat",3,r.hidden_width,*r.selection_width);
}
void scalar_shape(const NodeWeights& w,const std::string& p,Index f,Index hidden){
  shape(w,p+".matrix",{hidden,f});shape(w,p+".hidden_bias",{hidden});shape(w,p+".vector",{hidden});shape(w,p+".bias",{});
}
}
Tensor local_score(const NodeWeights& w,const RegionalConfig& r,const Tensor& f,const Groups& g,double epsilon){
  if(r.local_score=="cat")return head(w.extra,"read_program.score",f);
  const std::string p="read_program.score";auto v=encoded(w,p,g,epsilon);auto features=at::cat({v[0],v[1],v[2]});
  auto gate=at::silu(at::matmul(weight(w,p+".gate"),features)+weight(w,p+".gate_bias"));
  auto up=at::matmul(weight(w,p+".up"),features)+weight(w,p+".up_bias");
  return weight(w,p+".bias")+*r.alpha_match*(v[0]*v[1]).sum()/std::sqrt(static_cast<double>(*r.selection_width))
    +*r.alpha_res*(gate*up*weight(w,p+".vector")).sum();
}
Tensor region_evidence(const NodeWeights& w,const RegionalConfig& r,const Tensor& f,const Groups& g,double epsilon){
  if(r.region_control_mode=="fixed")return at::zeros({*r.region_control_width},f.options());
  if(r.region_evidence=="cat")return normalize(at::matmul(weight(w,"read_program.evidence_weight"),f)+weight(w,"read_program.evidence_bias"),epsilon);
  const std::string p="read_program.evidence";auto v=encoded(w,p,g,epsilon);
  auto result=at::matmul(weight(w,p+".cur"),v[0])+at::matmul(weight(w,p+".mem"),v[1])+at::matmul(weight(w,p+".stat"),v[2]);
  return normalize(result+at::matmul(weight(w,p+".interaction"),v[0]*v[1])+weight(w,p+".bias"),epsilon);
}
Tensor memory_key(const NodeWeights& w,const RegionalConfig& r,const Groups& g,double epsilon){
  if(r.region_key=="content")return encode_group(w,"read_program.key",g[1],epsilon);
  return at::matmul(weight(w,"read_program.key_weight"),at::cat({g[1],g[2].slice(0,1)}))+weight(w,"read_program.key_bias");
}
void validate_describe(const NodeWeights& w,const RegionalConfig& r,Index d,Index ds){
  const Index f=d+4*ds+3;const std::string p="read_program.score";
  if(r.local_score=="cat")scalar_shape(w,p,f,r.hidden_width);
  else {
    group_shape(w,p,d,ds,r);
    for(auto name:{"gate","up"}){shape(w,p+"."+name,{r.hidden_width,3*(*r.selection_width)});shape(w,p+"."+name+"_bias",{r.hidden_width});}
    shape(w,p+".vector",{r.hidden_width});shape(w,p+".bias",{});
  }
  if(r.region_control_mode=="context"){
    if(r.region_evidence=="cat"){
      shape(w,"read_program.evidence_weight",{*r.region_control_width,f});shape(w,"read_program.evidence_bias",{*r.region_control_width});
    } else {
      const std::string ep="read_program.evidence";group_shape(w,ep,d,ds,r);
      for(auto name:{"cur","mem","stat","interaction"})shape(w,ep+"."+name,{*r.region_control_width,*r.selection_width});
      shape(w,ep+".bias",{*r.region_control_width});
    }
  }
  if(r.send_gain=="context")scalar_shape(w,"read_program.gain",f,r.hidden_width);
  else if(r.gain_logit)shape(w,"read_program.gain_logit",{});
  if(r.region_score=="att"){
    shape(w,"read_program.current_weight",{*r.selection_width,d});shape(w,"read_program.current_bias",{*r.selection_width});
    if(r.region_key=="content_stats"){
      shape(w,"read_program.key_weight",{*r.selection_width,2*ds+2});shape(w,"read_program.key_bias",{*r.selection_width});
    } else encoder_shape(w,"read_program.key",2*ds,r.hidden_width,*r.selection_width);
  }
}
Tensor region_scores(const RegionWeights& w,const RegionalConfig& r,const std::map<std::string,Tensor>& fields){
  if(r.region_score=="ind")return fields.at("score");
  auto weights=fields.at(r.query_weight=="received"?"w":"mu");
  auto x=(weights.unsqueeze(-1)*fields.at("x")).sum(0)/weights.sum().clamp_min(1);
  auto hidden=at::silu(at::matmul(w.extra.at("query.in_weight"),x)+w.extra.at("query.hidden_bias"));
  auto query=at::matmul(w.extra.at("query.out_weight"),hidden)+w.extra.at("query.out_bias");
  return *r.alpha_loc*fields.at("score")+*r.alpha_sel*(fields.at("key")*query).sum(-1)/std::sqrt(static_cast<double>(*r.selection_width));
}
void validate_query(const RegionWeights& w,const RegionalConfig& r){
  if(r.region_score!="att")return;
  for(auto name:{"in_weight","out_weight"})require(w.extra.at(std::string("query.")+name).sizes()==at::IntArrayRef({*r.selection_width,*r.selection_width}),"invalid query matrix shape");
  for(auto name:{"hidden_bias","out_bias"})require(w.extra.at(std::string("query.")+name).sizes()==at::IntArrayRef({*r.selection_width}),"invalid query bias shape");
}
}
