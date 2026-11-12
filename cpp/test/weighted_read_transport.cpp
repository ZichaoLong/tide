// Cross-node Read transport: real alias reuse and independent column adjoints.
#include "weighted_graph_fixture.h"
#include "weighted_column_placement.h"
#include "tide/packing_work.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>

namespace {
using namespace tide;
namespace col=weighted::columns;
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b,const std::string& name){
  require(a.defined()==b.defined(),name+": disconnected root differs");if(!a.defined())return;
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),name+": shape/dtype differs");
  require(at::allclose(a,b,2e-5,2e-6),name+": numeric value differs");
}
Tensor leaf(std::vector<Index> shape,double offset){
  Index n=1;for(auto size:shape)n*=size;
  return (at::arange(n,at::kFloat)*.03+offset).reshape(shape).set_requires_grad(true);
}
struct Fixture {
  Model model;
  std::vector<col::PlacedReadRequest> requests;
  std::vector<Tensor> roots;
};
Fixture material(const std::string& mode,bool different=false,bool independent=false){
  Fixture f;f.model=weighted_check::model(weighted_check::graph("fc","kv"),"kv",at::kCPU,at::kFloat);
  auto& first=f.model.nodes[0].extra;
  first["read_program.score.vector"]=first.at("read_program.score.hidden_bias");
  for(Index g=1;g<3;++g)for(const auto& [name,value]:first)
    if(name=="kernel.summary"||name.rfind("read_program.",0)==0)
      f.model.nodes[g].extra[name]=independent&&g==2?value.detach().clone().set_requires_grad(true):value;
  for(Index g=0;g<3;++g){auto& owner=f.model.nodes[g];
    auto binding=weighted::node_binding(owner);
    auto& config=std::get<weighted::KVConfig>(binding.memory);config.memory_read="sum";
    col::KVWindow window;window.layout={{0,2,4},{2,4,3,5}};window.bank_layout={{0,2,4},{0,2,0,3}};
    window.mass=leaf({4},.3);window.content=leaf({4,2},.1+.1*g);
    col::KVState old;old.layout={{0,1,2},{0,0}};
    old.inert=at::zeros({2,3},at::kFloat);old.last_times={0,0};old.observations={1,1};
    for(const auto* name:{"weights","keys","values","summaries"}){
      const bool scalar=std::string(name)=="weights";
      window.bank[name]=leaf(scalar?std::vector<Index>{4}:std::vector<Index>{4,2},.2+.1*g);
      old.fields[name]=leaf(scalar?std::vector<Index>{2}:std::vector<Index>{2,2},.3+.1*g);
    }
    ExecutionPlacement p;p.preset="cpu";p.scoring_dtype=different&&g==2?"float32":"float64";
    f.requests.push_back({{&owner,binding,window,mode,old},resolve_placement(p,at::kCPU)});
  }
  auto root=[&](const Tensor& value){if(!value.requires_grad())return;
    for(const auto& t:f.roots)if(t.is_same(value))return;f.roots.push_back(value);};
  for(const auto& request:f.requests){for(const auto& [name,value]:request.query.weights->extra)root(value);
    const auto& window=std::get<col::KVWindow>(request.query.window);
    root(window.mass);root(window.content);for(const auto& [name,value]:window.bank)root(value);
    for(const auto& [name,value]:std::get<col::KVState>(request.query.initial).fields)root(value);}
  return f;
}
void values_and_roots(const std::string& mode,bool different,bool independent){
  auto f=material(mode,different,independent);
  work::packing::reset(true);
  auto actual=col::read_placed_many(f.requests,Index{1}<<28);
  auto metrics=work::packing::metrics();work::packing::reset(false);
  if(!different&&!independent){
    require(metrics["packing/projection/kv_tensor_shared/calls"]>=3,"transport lost shared matrix identity");
    require(metrics["packing/projection/kv_tensor/calls"]==0,"shared Read unnecessarily stacked node matrices");}
  std::vector<TensorFields> expected;
  for(const auto& r:f.requests){const auto& q=r.query;
    expected.push_back(col::read_placed(*q.weights,q.binding,q.window,q.mode,q.initial,r.placement));}
  for(size_t g=0;g<actual.size();++g)for(const auto& [name,value]:actual[g]){
    close(value,expected[g].at(name),"node field");
    require(value.requires_grad()==expected[g].at(name).requires_grad(),"field history differs");}
  for(double factor:{1.,0.}){
    auto a=actual[2].at("score"),e=expected[2].at("score");
    auto bar=(at::arange(a.numel(),a.options()).reshape(a.sizes())*.02+.3)*factor;
    auto ga=torch::autograd::grad({a},f.roots,{bar},true,false,true);
    auto ge=torch::autograd::grad({e},f.roots,{bar},true,false,true);
    for(size_t i=0;i<ga.size();++i)close(ga[i],ge[i],"non-first root VJP");
    auto unused=std::get<col::KVWindow>(f.requests[0].query.window).content;
    for(size_t i=0;i<f.roots.size();++i)if(f.roots[i].is_same(unused))require(!ga[i].defined(),"unused node root connected");
  }
  {at::NoGradGuard guard;auto inference=col::read_placed_many(f.requests,Index{1}<<28);
    for(const auto& fields:inference)for(const auto& [name,value]:fields)require(!value.requires_grad(),"inference retained a private graph");}
}
void original_versions(bool unused){
  auto f=material("proposal");auto values=col::read_placed_many(f.requests,Index{1}<<28);
  auto& q=f.requests[unused?0:2].query;
  auto target=unused?std::get<col::KVWindow>(q.window).bank.at("weights"):std::get<col::KVWindow>(q.window).content;
  {at::NoGradGuard guard;target.add_(.01);}
  bool refused=false;try{torch::autograd::grad({values[2].at("score").sum()},f.roots,{},false,false,true);}
  catch(const std::exception& e){refused=std::string(e.what()).find("modified by an inplace")!=std::string::npos;}
  require(refused,"changed original copy input was not rejected");
}
}
int main(){try{
  at::set_num_threads(1);at::set_num_interop_threads(1);
  for(const std::string mode:{"content","old","proposal"}){
    values_and_roots(mode,false,false);values_and_roots(mode,true,false);values_and_roots(mode,false,true);}
  original_versions(false);original_versions(true);
  std::cout<<"Read transport: 9 CPU value/root/alias cases and 2 version guards passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
