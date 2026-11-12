// KV §1.2: ragged positive writes, exact raw retention and independent roots.
#include "weighted_graph_fixture.h"
#include "../src/weighted_kv_tensor.h"
#include "tide/autograd_census.h"
#include "tide/packing_work.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <numeric>

namespace {
using namespace tide;
namespace col=weighted::columns;
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b,const std::string& label){
  require(a.defined()==b.defined(),label+" connection");if(!a.defined())return;
  require(a.sizes()==b.sizes()&&a.requires_grad()==b.requires_grad(),label+" metadata/history");
  const bool fp32=a.scalar_type()==at::kFloat;
  require(at::allclose(a,b,fp32?2e-5:2e-9,fp32?2e-6:2e-11),label+" value");
}
Tensor leaf(std::vector<Index> shape,double offset,at::ScalarType dtype){
  Index n=1;for(auto d:shape)n*=d;
  return (.13*at::sin(at::arange(n,at::kDouble)*.31+offset)).reshape(shape).to(dtype).set_requires_grad(true);
}
void root(std::vector<Tensor>& roots,const Tensor& value){
  if(!value.requires_grad())return;for(const auto& r:roots)if(r.is_same(value))return;roots.push_back(value);
}
void vjp(const Tensor& a,const Tensor& b,const std::vector<Tensor>& roots,const std::string& label){
  require(a.requires_grad()==b.requires_grad(),label+" output connection");if(!a.requires_grad())return;
  for(double factor:{1.,0.}){auto bar=(at::arange(a.numel(),a.options()).reshape(a.sizes())*.031+.2)*factor;
    auto ga=torch::autograd::grad({a},roots,{bar},true,false,true),gb=torch::autograd::grad({b},roots,{bar},true,false,true);
    for(size_t j=0;j<roots.size();++j)close(ga[j],gb[j],label+" root "+std::to_string(j));}
}
struct Fixture {std::vector<NodeWeights> owners;std::vector<col::KVPrepareRequest> requests;std::vector<Tensor> roots,write_roots;};
Fixture fixture(at::ScalarType dtype,bool frozen=false){
  Fixture f;auto model=weighted_check::model(weighted_check::graph("fc","kv"),"kv",at::kCPU,dtype);
  f.owners.assign(model.nodes.begin(),model.nodes.begin()+5);const Index t=Index{1}<<54;
  std::vector<col::Layout> layouts{{{0,3,5,6},{t+10,t+13,t+20,t+11,t+25,t+18}},{{0,2,3},{t+12,t+22,t+15}},
    {{0,2,4},{t+14,t+21,t+17,t+23}},{{0,2,5},{t+11,t+17,t+10,t+19,t+29}},{{0,3,5,6},{t+10,t+13,t+20,t+11,t+25,t+18}}};
  std::vector<std::vector<double>> masses{{.3,0.,.8,.2,1.2,0.},{.4,0.,.9},{0.,0.,0.,0.},{.3,.4,0.,.6,0.},{.3,0.,.8,.2,1.2,0.}};
  for(Index i=0;i<5;++i){auto& w=f.owners[i];w.extra["kernel.base"]=f.owners[0].extra.at("kernel.base");
    w.extra["kernel.adapter_v"]=w.extra.at("kernel.summary");w.extra["kernel.adapter_u"]=leaf({4,2},.4+i,dtype);
    if(frozen)for(auto& [name,p]:w.extra)p=p.detach();
    auto c=std::get<weighted::KVConfig>(weighted::node_binding(w).memory);c.adapter_rank=2;c.window=9;
    std::vector<State> old;
    for(Index b=0;b<layouts[i].samples();++b){auto s=w.kernel->initial(w);s.last_time=t+6;s.observations=2;
      s.value=s.value.detach().set_requires_grad(true);root(f.roots,s.value);
      s.slots["times"]=at::tensor(std::vector<Index>{t+1,t+6},at::kLong);
      for(const auto* name:{"keys","values","summaries","weights"}){
        auto value=std::string(name)=="weights"?at::tensor({.2,.4},at::kDouble).to(dtype).set_requires_grad(true):leaf({2,2},.2+i+b,dtype);
        s.slots[name]=value;root(f.roots,value);}
      old.push_back(std::move(s));
    }
    auto mass=at::tensor(masses[i],at::kDouble).to(dtype).set_requires_grad(true);
    auto u=leaf({static_cast<Index>(layouts[i].times.size()),2},.6+i,dtype);if(frozen)u=u.detach();
    if(i==4)u=f.requests[0].content;
    f.requests.push_back({&w,c,layouts[i],mass,u,col::KVState::from_states(old)});
  }
  for(const auto& w:f.owners)for(const auto& [name,p]:w.extra)root(f.write_roots,p);
  for(const auto& r:f.requests){root(f.write_roots,r.mass);root(f.write_roots,r.content);}
  for(const auto& r:f.write_roots)root(f.roots,r);return f;
}
TensorFields literal(const col::KVPrepareRequest& r){
  auto ids=(r.mass.detach()>0).nonzero().flatten(),u=r.content.index_select(0,ids);
  auto n=u/at::sqrt(r.config.epsilon+u.square().mean(-1,true));const auto& w=r.weights->extra;
  // Independent algebra: form the low-rank matrix before applying the rows.
  auto matrix=w.at("kernel.base")+at::matmul(w.at("kernel.adapter_u"),w.at("kernel.adapter_v"));
  auto projected=at::matmul(n,matrix.t());return {{"keys",projected.slice(1,0,2)},{"values",projected.slice(1,2)},
    {"summaries",at::matmul(n,w.at("kernel.summary").t())},{"weights",r.mass.index_select(0,ids).clamp_max(1)}};
}
void check(at::ScalarType dtype,bool frozen=false){
  auto f=fixture(dtype,frozen);const Index budget=16*1024*1024;
  work::packing::reset(true);auto actual=col::prepare_many(f.requests,budget);auto counts=work::packing::metrics();work::packing::reset(false);
  require(counts.at("packing/bucket/kv_tensor_writes/rows")==4&&counts.at("packing/bucket/kv_tensor_writes/calls")==1,"unequal write counts not grouped");
  require(counts.at("packing/bucket/kv_tensor_writes/length/4/rows/4/calls")==1,"actual padded write dimensions");
  auto bounded=col::prepare_many(f.requests,1);std::vector<col::KVWindow> expected;
  for(const auto& r:f.requests)expected.push_back(col::prepare(*r.weights,r.config,r.layout,r.mass,r.content,r.left));
  for(size_t i=0;i<actual.size();++i){const auto& r=f.requests[i];const auto& a=actual[i];const auto& e=expected[i];
    require(a.bank_layout.times==e.bank_layout.times&&a.bank_layout.offsets==e.bank_layout.offsets,"raw bank int64 layout");
    require(a.terminal.layout.times==e.terminal.layout.times&&a.terminal.layout.offsets==e.terminal.layout.offsets,"terminal int64 layout");
    require(a.terminal.last_times==e.terminal.last_times&&a.terminal.observations==e.terminal.observations,"raw clocks/counts");
    close(a.terminal.inert,e.terminal.inert,"inert");
    for(const auto& [name,value]:a.bank){close(value,e.bank.at(name),"bank "+name);close(value,bounded[i].bank.at(name),"budget bank "+name);
      vjp(value,e.bank.at(name),f.roots,"bank "+name+" node "+std::to_string(i));
      close(a.terminal.fields.at(name),e.terminal.fields.at(name),"terminal "+name);
      vjp(a.terminal.fields.at(name),e.terminal.fields.at(name),f.roots,"terminal "+name+" node "+std::to_string(i));}
    if(i==2){for(const auto& [name,value]:a.bank)require(value.is_same(r.left.fields.at(name)),"zero-write raw identity");continue;}
    std::vector<Index> fresh;for(Index b=0;b<a.bank_layout.samples();++b){const auto old=r.left.layout.offsets[b+1]-r.left.layout.offsets[b];
      for(Index j=a.bank_layout.offsets[b]+old;j<a.bank_layout.offsets[b+1];++j)fresh.push_back(j);}
    auto index=at::tensor(fresh,at::kLong);auto direct=literal(r);
    for(const auto& [name,value]:direct){auto value_actual=a.bank.at(name).index_select(0,index);close(value_actual.detach(),value.detach(),"literal "+name);
      // Selecting fresh rows through the full raw bank connects old columns to
      // zeros. This independent formula covers write roots; raw roots above.
      if(value.requires_grad())vjp(value_actual,value,f.write_roots,"literal "+name);
      else {auto gradients=torch::autograd::grad({value_actual.sum()},f.write_roots,{},true,false,true);
        for(const auto& gradient:gradients)require(!gradient.defined(),"frozen fresh writes acquired a root");}}
  }
  vjp(actual[0].bank.at("keys").sum()+actual[1].bank.at("values").sum()+actual[4].bank.at("summaries").sum(),
    expected[0].bank.at("keys").sum()+expected[1].bank.at("values").sum()+expected[4].bank.at("summaries").sum(),f.roots,"shared aliases");
  // The silent third sample retains its old entries even though other samples
  // advanced and expired history; a subsequent entirely silent window is exact.
  auto next=f.requests[0];next.left=actual[0].terminal;next.layout={{0,1,2,3},{(Index{1}<<54)+100,(Index{1}<<54)+101,(Index{1}<<54)+102}};
  next.mass=at::zeros({3},next.mass.options());next.content=at::zeros({3,2},next.content.options());
  auto silent=col::prepare_many({next},budget)[0];require(silent.terminal.layout.times==next.left.layout.times,"silent expiry changed raw timestamps");
  for(const auto& [name,value]:silent.terminal.fields)require(value.is_same(next.left.fields.at(name)),"silent continuation changed raw fields");
  {at::NoGradGuard guard;f.requests[0].mass.add_(.01);}
  bool rejected=false;try{torch::autograd::grad({actual[0].bank.at("weights").sum()},f.roots,{},false,false,true);}catch(const std::exception&){rejected=true;}
  require(rejected,"original write input version not guarded");
}
void admission(){
  auto f=fixture(at::kDouble);work::packing::reset(true);
  auto limited=col::prepare_many(f.requests,7000);auto counts=work::packing::metrics();work::packing::reset(false);
  require(counts.at("packing/bucket/kv_tensor_writes/calls")==2&&counts.at("packing/bucket/kv_tensor_writes/rows")==4,
    "write admission ignored group maximum padding");
  std::vector<col::KVWindow> scalar;
  for(size_t i=0;i<f.requests.size();++i){const auto& r=f.requests[i];scalar.push_back(col::prepare(*r.weights,r.config,r.layout,r.mass,r.content,r.left));
    for(const auto& [name,value]:limited[i].bank)close(value,scalar.back().bank.at(name),"budget split "+name);}
  vjp(limited[0].bank.at("keys").sum()+limited[3].bank.at("values").sum(),
    scalar[0].bank.at("keys").sum()+scalar[3].bank.at("values").sum(),f.roots,"cross-group shared roots");
  f.requests[1].config.epsilon=.7;
  work::packing::reset(true);auto mixed=col::prepare_many(f.requests,16*1024*1024);counts=work::packing::metrics();work::packing::reset(false);
  require(counts.at("packing/bucket/kv_tensor_writes/rows")==3,"different epsilon entered one formula group");
  for(size_t i=0;i<f.requests.size();++i){const auto& r=f.requests[i];auto scalar=col::prepare(*r.weights,r.config,r.layout,r.mass,r.content,r.left);
    for(const auto& [name,value]:mixed[i].bank)close(value,scalar.bank.at(name),"epsilon group "+name);}
}
void graph_size(){
  auto f=fixture(at::kDouble);Index expected=-1;
  for(const auto size:{std::pair<Index,Index>{2,2},{5,7}}){const auto [nodes,rows]=size;
    std::vector<Index> ids(rows);std::iota(ids.begin(),ids.end(),0);
    auto u=leaf({rows+1,2},.2,at::kDouble),mass=leaf({rows+1},.3,at::kDouble);
    {at::NoGradGuard guard;mass[rows]=0;}
    std::vector<col::tensor::WriteRequest> group(nodes,{&f.owners[0],mass,u,&ids});
    autograd_census::reset(true);auto output=col::tensor::prepare(group,.3);auto metrics=autograd_census::metrics();autograd_census::reset(false);
    const auto count=metrics.at("graphs/private/kv.tensor.writes/unique_nodes");if(expected<0)expected=count;
    else require(count==expected,"write private graph grew with nodes/rows");
  }
  std::cout<<"KV write private graph nodes (G/R independent): "<<expected<<'\n';
}
}
int main(){try{at::set_num_threads(1);check(at::kDouble);check(at::kFloat);check(at::kDouble,true);admission();graph_size();
  std::cout<<"KV tensor writes: ragged data/roots, shared roles, literal formula, raw retention and continuation passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
