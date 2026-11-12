// Allocation §2–3: one call-local parameter view, literal interior quad and roots.
#include "../src/weighted_column_placement.h"
#include "../src/weighted_regional_groups.h"
#include "../src/weighted_placement.h"
#include "tide/autograd_census.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <set>

namespace {
using namespace tide;
namespace col=weighted::columns;
void require(bool value,const std::string& name){if(!value)throw std::runtime_error(name);}
void close(const Tensor& a,const Tensor& b,const std::string& name){
  require(a.defined()==b.defined(),name+" connectivity");if(!a.defined())return;
  require(a.sizes()==b.sizes(),name+" shape");const bool fp32=a.scalar_type()==at::kFloat;
  require(at::allclose(a,b,fp32?2e-5:1e-8,fp32?2e-6:1e-10),name+" numeric");
}
Tensor leaf(std::vector<Index> shape,double start,at::ScalarType dtype){
  Index n=1;for(auto d:shape)n*=d;
  return (at::arange(n,at::kDouble)*.007+start).reshape(shape).to(dtype).set_requires_grad(true);
}
std::map<Index,Tensor> literal(const RegionWeights& weights,const weighted::RegionalConfig& c,
                             const col::RegionalPacking& packing,const col::RegionalFields& fields){
  std::set<Index> rows;std::map<Index,std::vector<Tensor>> controls;
  for(const auto& [node,domain]:packing.domains)rows.insert(domain.begin(),domain.end());
  auto weight=[&](const std::string& name){return weights.extra.at(name).to(at::kDouble);};
  for(auto row:rows){std::vector<Index> ids;std::vector<Tensor> recv,caps,scores,evidence;
    for(const auto& [node,domain]:packing.domains){auto p=std::lower_bound(domain.begin(),domain.end(),row);
      if(p==domain.end()||*p!=row)continue;const auto i=p-domain.begin();const auto& f=fields.at(node);
      ids.push_back(node);recv.push_back(f.at("w")[i]);caps.push_back(f.at("mu")[i]);scores.push_back(f.at("score")[i]);evidence.push_back(f.at("evidence")[i]);}
    auto w=at::stack(recv),mu=at::stack(caps),s=at::stack(scores),e=at::stack(evidence);
    auto received=w.sum(),mass=mu.sum(),z=(w.unsqueeze(-1)*e).sum(0)/received.clamp_min(1);
    auto input=at::cat({z,received.log1p().reshape({1}),mass.log1p().reshape({1})});
    auto h=at::matmul(weight("head.matrix"),input)+weight("head.hidden_bias");
    auto context=(h*at::sigmoid(h)*weight("head.vector")).sum()+weight("head.bias");
    auto lower=received.clamp_max(1),target=lower+context.sigmoid()*(mass.clamp_max(*c.k)-lower);
    // All selected fixture rows are strictly interior: this is the literal
    // Lagrange multiplier elimination, not a call to either implementation.
    auto value=(s-s.mean())/(*c.allocation_temperature)+target/double(ids.size());
    require(value.detach().gt(0).all().item<bool>()&&value.detach().lt(mu.detach()).all().item<bool>(),"literal interior domain");
    for(size_t i=0;i<ids.size();++i)controls[ids[i]].push_back(value[i]);
  }
  std::map<Index,Tensor> out;for(auto& [node,values]:controls)out[node]=at::stack(values);return out;
}
void check(bool aligned,bool grad){
  RegionWeights weights;weights.extra={{"head.matrix",leaf({3,4},.02,at::kFloat)},
    {"head.hidden_bias",leaf({3},.03,at::kFloat)},{"head.bias",leaf({},.01,at::kFloat)},
    {"unused",leaf({3},.03,at::kFloat)}};
  weights.extra["head.vector"]=weights.extra.at("head.hidden_bias");
  weighted::RegionalConfig config;config.k=2;config.hidden_width=3;config.region_control_width=2;
  config.allocation_temperature=2.;config.validate();
  col::RegionalPacking packing({{0,{0,2}},{1,{0,2}},{2,{1,3}},{3,{1,3}},{4,aligned?std::vector<Index>{1,3}:std::vector<Index>{2,3}}});
  require(packing.groups.size()==2,"fixture requires two cardinality groups");col::RegionalFields fields;
  for(Index node=0;node<5;++node)fields[node]={{"w",leaf({2},.2+.01*node,at::kDouble)},
    {"mu",leaf({2},.8,at::kDouble)},{"score",leaf({2},.02*node,at::kDouble)},
    {"evidence",leaf({2,2},.1+.02*node,at::kDouble)}};
  std::vector<Tensor> roots;std::set<const void*> seen;
  for(const auto& [name,t]:weights.extra)if(seen.insert(t.unsafeGetTensorImpl()).second)roots.push_back(t);
  for(const auto& [node,domain]:fields)for(const auto& [name,t]:domain)roots.push_back(t);
  ExecutionPlacement request;request.preset="cpu";const auto placement=resolve_placement(request,at::kCPU);
  autograd_census::reset(true);std::map<Index,Tensor> actual;
  {at::AutoGradMode mode(grad);actual=col::region_groups(weights,config,packing,fields,placement);}
  std::vector<Tensor> outputs;for(const auto& [node,value]:actual)outputs.push_back(value);
  autograd_census::graph("observed",outputs);auto counts=autograd_census::metrics();autograd_census::reset(false);
  Index copies=0;for(const auto& [name,n]:counts)if(name.rfind("total/node_types/",0)==0&&name.find("::Copies>")!=std::string::npos)copies+=n;
  require(copies==(grad?1:0),"parameter conversions were repeated per cardinality group");
  auto expected=literal(weights,config,packing,fields);
  for(const auto& [node,value]:actual){close(value,expected.at(node),"literal controls");require(value.requires_grad()==grad,"output grad mode");}
  if(grad){
    for(Index node:{0,2,4})for(double factor:{1.,0.}){
      auto a=torch::autograd::grad({actual[node].square().sum()*factor},roots,{},true,false,true);
      auto b=torch::autograd::grad({expected[node].square().sum()*factor},roots,{},true,false,true);
      for(size_t i=0;i<roots.size();++i)close(a[i],b[i],"literal independent VJP");}
    if(!aligned){bool rejected=false;
      try{torch::autograd::grad({actual[0].sum()},roots,{},true,true,true);}catch(const std::exception& e){rejected=std::string(e.what()).find("first-order")!=std::string::npos;}
      require(rejected,"regional view accepted higher-order VJP");
      {at::NoGradGuard guard;weights.extra.at("unused").add_(.1);}
      rejected=false;try{torch::autograd::grad({actual[0].sum()},roots,{},false,false,true);}catch(const std::exception&){rejected=true;}
      require(rejected,"unused original parameter version was not guarded");
    }
  }
}
void views(){
  RegionWeights weights;auto x=leaf({2},.2,at::kDouble);weights.extra={{"a",x},{"b",x},{"independent",x.detach().clone().set_requires_grad(true)}};
  col::RegionParameterViews view(weights,at::kCPU);
  require(&view.parameters(at::kDouble)==&weights,"same-location parameter view was copied");
  const auto& first=view.parameters(at::kFloat);const auto& again=view.parameters(at::kFloat);
  require(&first==&again&&first.extra.at("a").is_same(first.extra.at("b"))
    &&!first.extra.at("a").is_same(first.extra.at("independent")),"parameter identity/cache differs");
  auto a=first.extra.at("a"),b=first.extra.at("b"),unused=first.extra.at("independent");
  auto grads=torch::autograd::grad({(a+2*b).sum()},{x,weights.extra.at("independent")},{},true,false,true);
  close(grads[0],at::full_like(x,3),"shared view contributions");require(!grads[1].defined(),"equal parameters merged");
  {at::NoGradGuard guard;x.add_(.1);}
  col::RegionParameterViews fresh(weights,at::kCPU);
  require(!at::equal(first.extra.at("a"),fresh.parameters(at::kFloat).extra.at("a")),"view leaked across calls");
}
}
int main(){try{at::set_num_threads(1);views();for(bool aligned:{false,true})for(bool grad:{false,true}){
  try{check(aligned,grad);}catch(const std::exception& e){throw std::runtime_error("aligned="+std::to_string(aligned)+" grad="+std::to_string(grad)+": "+e.what());}}
  std::cout<<"Regional transport CPU: one parameter view across groups, literal quad, aliases, roots, versions and grad modes passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
