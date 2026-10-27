#include "weighted_describe.h"
#include <array>
#include <cmath>
namespace tide::device_online {
WeightedDescribeTape append_weighted_describe(DeviceProgram& p,const WeightedDescribeInput& x,const at::Tensor& owners,
    const WeightedDescribeParameters& w,const weighted::RegionalConfig& c,double epsilon,const at::Tensor& error,int64_t available){
  if(!x.content.defined()||x.content.dim()!=2||!x.short_memory.defined()||x.short_memory.dim()!=2||!w.count("summary.matrix"))
    throw std::invalid_argument("invalid Describe inputs/parameters");
  const auto rows=x.content.size(0),d=x.content.size(1),ds=x.short_memory.size(1),banks=w.at("summary.matrix").size(0);
  const auto shapes=weighted_describe_parameter_shapes(banks,d,ds,c);
  if(w.size()!=shapes.size())throw std::invalid_argument("Describe parameter roles differ from configuration");
  for(const auto& [name,shape]:shapes){auto found=w.find(name);if(found==w.end()||!found->second.defined()||found->second.sizes()!=at::IntArrayRef(shape)
    ||found->second.device()!=x.content.device()||found->second.scalar_type()!=at::kFloat||!found->second.is_contiguous()||found->second.requires_grad())
      throw std::invalid_argument("invalid Describe parameter: "+name);}
  for(const auto& value:{x.mass,x.short_total,x.long_total})if(!value.defined()||value.sizes()!=at::IntArrayRef{rows})throw std::invalid_argument("Describe scalar field shape");
  if(x.long_memory.sizes()!=at::IntArrayRef{rows,ds})throw std::invalid_argument("Describe memory vector shape");
  WeightedDescribeTape t;t.graph=std::make_shared<WeightedRowGraph>(p,owners,banks,error,available);auto& g=*t.graph;
  const auto mass=g.input(x.mass.reshape({rows,1}),true),u=g.input(x.content),ms=g.input(x.short_memory),ml=g.input(x.long_memory),
    ws=g.input(x.short_total.reshape({rows,1}),true),wl=g.input(x.long_total.reshape({rows,1}),true);
  t.inputs={{"mass",mass},{"content",u},{"short_memory",ms},{"long_memory",ml},{"short_total",ws},{"long_total",wl}};
  auto projection=[&](const std::string& name,size_t input){return g.projection(input,{w.at(name+".matrix"),w.count(name+".bias")?w.at(name+".bias"):at::Tensor{}},name);};
  auto head=[&](const std::string& name,size_t input){return g.head(input,{w.at(name+".matrix"),w.at(name+".hidden_bias"),w.at(name+".projection"),w.at(name+".bias")},name);};
  const auto cur=g.norm(u,epsilon),mem=g.cat({ms,ml}),stat=g.cat({g.log1p(mass),g.log1p(ws),g.log1p(wl)}),current=projection("summary",cur);
  const auto features=g.cat({cur,g.slice(stat,0,1),ms,ml,g.slice(stat,1,2),g.product(current,ms),g.product(current,ml)});
  const auto one=g.constant(1,1);
  auto encoders=[&](const std::string& prefix){return std::array<size_t,3>{g.norm(head(prefix+".cur",cur),epsilon),
    g.norm(head(prefix+".mem",mem),epsilon),g.norm(head(prefix+".stat",stat),epsilon)};};
  const auto received=g.clamp_max(mass,1);size_t caps,score;
  if(c.regional())caps=g.clamp_max(g.scale(received,c.gamma_max),1);
  else {size_t gamma;
    if(c.send_gain=="context")gamma=g.scale(g.sigmoid(head("gain",features)),c.gamma_max-1,1);
    else if(c.gain_logit)gamma=g.scale(g.sigmoid(projection("gain.logit",one)),c.gamma_max-1,1);
    else gamma=g.constant(1,c.fixed_gain.value_or(1));
    caps=g.clamp_max(g.product(gamma,mass),1);
  }
  if(c.local_score=="cat")score=head("score",features);
  else {auto encoded=encoders("score.encoders");auto f=g.cat({encoded[0],encoded[1],encoded[2]});
    auto gated=g.product(g.silu(projection("score.gate",f)),projection("score.up",f));
    auto match=g.scale(g.sum(g.product(encoded[0],encoded[1])),*c.alpha_match/std::sqrt(*c.selection_width));
    auto residual=g.scale(projection("score.vector",gated),*c.alpha_res);score=g.add(g.add(projection("score.bias",one),match),residual);}
  t.outputs={{"w",received},{"mu",caps},{"score",score}};
  if(c.regional()){
    size_t evidence;
    if(c.region_control_mode=="fixed")evidence=g.constant(*c.region_control_width,0);
    else if(c.region_evidence=="cat")evidence=g.norm(projection("evidence",features),epsilon);
    else {auto encoded=encoders("evidence.encoders");auto result=projection("evidence.cur",encoded[0]);
      result=g.add(result,projection("evidence.mem",encoded[1]));result=g.add(result,projection("evidence.stat",encoded[2]));
      result=g.add(result,projection("evidence.interaction",g.product(encoded[0],encoded[1])));
      result=g.add(result,projection("evidence.bias",one));evidence=g.norm(result,epsilon);}
    t.outputs["evidence"]=evidence;
  }
  if(c.region_score=="att"){
    t.outputs["x"]=projection("current",cur);
    t.outputs["key"]=c.region_key=="content"?g.norm(head("key",mem),epsilon):projection("key",g.cat({mem,g.slice(stat,1,2)}));
  }
  for(const auto& [name,index]:t.outputs)t.fields[name]=g.value(index);return t;
}
WeightedDescribeGradient append_weighted_describe_vjp(DeviceProgram& p,const WeightedDescribeTape& t,
    const std::map<std::string,WeightedRowAdjoint>& roots,const at::Tensor& error,int64_t available){
  if(!t.graph)throw std::invalid_argument("missing Describe tape");std::vector<WeightedRowRoot> inputs;
  for(const auto& [name,root]:roots){auto found=t.outputs.find(name);if(found==t.outputs.end())throw std::invalid_argument("inapplicable Describe root: "+name);
    inputs.push_back({found->second,root.value,root.connected});}
  auto g=t.graph->reverse(p,inputs,error,available);WeightedDescribeGradient result;result.parameters=std::move(g.parameters);
  for(const auto& [name,index]:t.inputs)result.inputs[name]=g.rows.at(index);return result;
}
}
