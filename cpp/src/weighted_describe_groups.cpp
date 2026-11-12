// Allocation §3: compatible Describe heads in whole-node block projections.
#include "weighted_describe_heads.h"
#include "weighted_column_query_ops.h"
namespace tide::weighted::query {
using detail::weight;using detail::Groups;
namespace {
using Weights=std::vector<const NodeWeights*>;using Values=std::vector<Tensor>;
Values project(const Weights& w,const Values& x,const std::string& name){
  Values matrices;for(auto p:w)matrices.push_back(weight(*p,name));return columns::project_nodes(x,matrices);
}
Values head(const Weights& w,const Values& x,const std::string& p){
  auto hidden=project(w,x,p+".matrix");
  for(size_t i=0;i<w.size();++i)hidden[i]=weight(*w[i],p+".bias")+
    (at::silu(hidden[i]+weight(*w[i],p+".hidden_bias"))*weight(*w[i],p+".vector")).sum(-1);
  return hidden;
}
Values encode(const Weights& w,const Values& x,const std::string& p,double epsilon){
  auto hidden=project(w,x,p+".in_weight");
  for(size_t i=0;i<w.size();++i)hidden[i]=at::silu(hidden[i]+weight(*w[i],p+".hidden_bias"));
  auto out=project(w,hidden,p+".out_weight");
  for(size_t i=0;i<w.size();++i)out[i]=normalize(out[i]+weight(*w[i],p+".out_bias"),epsilon);
  return out;
}
std::vector<Groups> encoded(const Weights& w,const std::vector<Groups>& groups,const std::string& p,double e){
  std::vector<Groups> out(w.size());const std::vector<std::string> names{"cur","mem","stat"};
  for(size_t j=0;j<3;++j){Values xs;for(const auto& g:groups)xs.push_back(g[j]);
    auto values=encode(w,xs,p+".encoders."+names[j],e);for(size_t i=0;i<w.size();++i)out[i][j]=values[i];}
  return out;
}
}
std::vector<TensorFields> describe_nodes(const std::vector<const NodeWeights*>& w,const RegionalConfig& r,
    const std::vector<Tensor>& mass,const std::vector<Tensor>& f,const std::vector<Groups>& g,double e){
  if(w.empty())return {};
  std::vector<TensorFields> out(w.size());Values scores;
  if(r.local_score=="cat")scores=head(w,f,"read_program.score");
  else {
    const std::string p="read_program.score";auto v=encoded(w,g,p,e);Values joined;
    for(const auto& row:v)joined.push_back(at::cat({row[0],row[1],row[2]},-1));
    auto gate=project(w,joined,p+".gate"),up=project(w,joined,p+".up");
    for(size_t i=0;i<w.size();++i)scores.push_back(weight(*w[i],p+".bias")+
      *r.alpha_match*(v[i][0]*v[i][1]).sum(-1)/std::sqrt(double(*r.selection_width))+
      *r.alpha_res*(at::silu(gate[i]+weight(*w[i],p+".gate_bias"))*(up[i]+weight(*w[i],p+".up_bias"))*weight(*w[i],p+".vector")).sum(-1));
  }
  Values gains;if(!r.regional()&&r.send_gain=="context")gains=head(w,f,"read_program.gain");
  for(size_t i=0;i<w.size();++i){auto recv=mass[i].clamp_max(1);Tensor mu;
    if(r.regional())mu=(r.gamma_max*recv).clamp_max(1);
    else {
      auto gamma=at::full({},r.fixed_gain.value_or(1),mass[i].options());
      if(!gains.empty())gamma=1+(r.gamma_max-1)*gains[i].sigmoid();
      else if(r.gain_logit)gamma=1+(r.gamma_max-1)*weight(*w[i],"read_program.gain_logit").sigmoid();
      mu=(gamma*mass[i]).clamp_max(1);
    }
    out[i]={{"w",recv},{"mu",mu},{"score",scores[i]}};
  }
  if(r.regional()){
    Values evidence;
    if(r.region_control_mode=="fixed")for(const auto& x:f){auto shape=x.sizes().vec();shape.back()=*r.region_control_width;evidence.push_back(at::zeros(shape,x.options()));}
    else if(r.region_evidence=="cat"){
      evidence=project(w,f,"read_program.evidence_weight");
      for(size_t i=0;i<w.size();++i)evidence[i]=normalize(evidence[i]+weight(*w[i],"read_program.evidence_bias"),e);
    }else{
      const std::string p="read_program.evidence";auto v=encoded(w,g,p,e);std::vector<Values> maps;
      const std::vector<std::string> names{"cur","mem","stat","interaction"};
      for(size_t j=0;j<4;++j){Values xs;for(const auto& row:v)xs.push_back(j<3?row[j]:row[0]*row[1]);maps.push_back(project(w,xs,p+"."+names[j]));}
      for(size_t i=0;i<w.size();++i)evidence.push_back(normalize(maps[0][i]+maps[1][i]+maps[2][i]+maps[3][i]+weight(*w[i],p+".bias"),e));
    }
    for(size_t i=0;i<w.size();++i)out[i]["evidence"]=evidence[i];
  }
  if(r.region_score=="att"){
    Values cur,memory;for(const auto& row:g){cur.push_back(row[0]);memory.push_back(row[1]);}
    auto xs=project(w,cur,"read_program.current_weight");Values keys;
    if(r.region_key=="content")keys=encode(w,memory,"read_program.key",e);
    else {
      Values inputs;for(const auto& row:g)inputs.push_back(at::cat({row[1],row[2].slice(-1,1)},-1));
      keys=project(w,inputs,"read_program.key_weight");
      for(size_t i=0;i<w.size();++i)keys[i]=keys[i]+weight(*w[i],"read_program.key_bias");
    }
    for(size_t i=0;i<w.size();++i){out[i]["x"]=xs[i]+weight(*w[i],"read_program.current_bias");out[i]["key"]=keys[i];}
  }
  return out;
}
}
