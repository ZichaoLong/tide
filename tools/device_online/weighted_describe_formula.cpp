// Independent literal allocation §3/5.1 formulas; no resident algebra helpers.
#include "weighted_describe_formula.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
namespace tide::device_online::describe_check {
using at::Tensor;
namespace {
Tensor normalized(const Tensor& x){return x/(x.square().mean()+.13).sqrt();}
Tensor affine(const Fields& p,const std::string& name,const Tensor& x){auto y=at::mv(p.at(name+".matrix"),x);if(p.count(name+".bias"))y=y+p.at(name+".bias");return y;}
Tensor head(const Fields& p,const std::string& name,const Tensor& x){auto a=at::mv(p.at(name+".matrix"),x)+p.at(name+".hidden_bias");return at::mv(p.at(name+".projection"),a*a.sigmoid())+p.at(name+".bias");}
}
Formula formula(const Fields& inputs,const Tensor& owners,const WeightedDescribeParameters& weights,const weighted::RegionalConfig& c){
  at::AutoGradMode enabled(true);Formula f;std::vector<Fields> xs(R),params(B);
  for(const auto& [name,x]:inputs)for(int r=0;r<R;++r){auto leaf=x[r].clone().set_requires_grad(true);xs[r][name]=leaf;f.leaves.push_back(leaf);f.labels.push_back({name,r,false});}
  for(const auto& [name,x]:weights)for(int b=0;b<B;++b){auto leaf=x[b].clone().set_requires_grad(true);params[b][name]=leaf;f.leaves.push_back(leaf);f.labels.push_back({name,b,true});}
  for(int r=0;r<R;++r){Fields output{{"w",at::zeros({1},at::kFloat)},{"mu",at::zeros({1},at::kFloat)},{"score",at::zeros({1},at::kFloat)}};
    if(c.regional())output["evidence"]=at::zeros({*c.region_control_width},at::kFloat);
    if(c.region_score=="att"){output["x"]=at::zeros({*c.selection_width},at::kFloat);output["key"]=at::zeros({*c.selection_width},at::kFloat);}
    if(owners[r].item<int64_t>()<0){f.fields.push_back(output);continue;}
    const auto& x=xs[r];const auto& p=params[owners[r].item<int64_t>()];
    auto mass=x.at("mass"),cur=normalized(x.at("content")),ms=x.at("short_memory"),ml=x.at("long_memory"),ws=x.at("short_total"),wl=x.at("long_total");
    auto mem=at::cat({ms,ml}),stat=at::cat({mass.log1p(),ws.log1p(),wl.log1p()}),current=affine(p,"summary",cur);
    auto feature=at::cat({cur,stat.slice(0,0,1),ms,ml,stat.slice(0,1),current*ms,current*ml}),one=at::ones({1},at::kFloat);
    auto encoders=[&](const std::string& prefix){return std::vector<Tensor>{normalized(head(p,prefix+".cur",cur)),normalized(head(p,prefix+".mem",mem)),normalized(head(p,prefix+".stat",stat))};};
    auto received=mass.clamp_max(1);output["w"]=received;
    if(c.regional())output["mu"]=(c.gamma_max*received).clamp_max(1);
    else {Tensor gamma;if(c.send_gain=="context")gamma=1+(c.gamma_max-1)*head(p,"gain",feature).sigmoid();
      else if(c.gain_logit)gamma=1+(c.gamma_max-1)*affine(p,"gain.logit",one).sigmoid();else gamma=at::full({1},*c.fixed_gain,at::kFloat);
      output["mu"]=(gamma*mass).clamp_max(1);}
    if(c.local_score=="cat")output["score"]=head(p,"score",feature);
    else {auto e=encoders("score.encoders");auto combined=at::cat(e),gate=affine(p,"score.gate",combined),up=affine(p,"score.up",combined);
      auto match=(e[0]*e[1]).sum().reshape({1})/std::sqrt(*c.selection_width);
      output["score"]=affine(p,"score.bias",one)+*c.alpha_match*match+*c.alpha_res*affine(p,"score.vector",gate*gate.sigmoid()*up);}
    if(c.region_control_mode=="context"){
      if(c.region_evidence=="cat")output["evidence"]=normalized(affine(p,"evidence",feature));
      else {auto e=encoders("evidence.encoders");output["evidence"]=normalized(affine(p,"evidence.cur",e[0])+affine(p,"evidence.mem",e[1])+
        affine(p,"evidence.stat",e[2])+affine(p,"evidence.interaction",e[0]*e[1])+affine(p,"evidence.bias",one));}
    }
    if(c.region_score=="att"){output["x"]=affine(p,"current",cur);
      output["key"]=c.region_key=="content"?normalized(head(p,"key",mem)):affine(p,"key",at::cat({mem,stat.slice(0,1)}));}
    f.fields.push_back(output);
  }return f;
}
}
