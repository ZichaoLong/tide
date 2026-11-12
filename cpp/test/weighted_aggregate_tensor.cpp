// Architecture §2: literal physical atoms, distinct roots and regional batching.
#include "weighted_graph_fixture.h"
#include "../src/weighted_column_sources.h"
#include "../src/weighted_aggregate_layout.h"
#include "tide/packing_work.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <set>

namespace {
using namespace tide;
namespace col=weighted::columns;
void require(bool ok,const std::string& name){if(!ok)throw std::runtime_error(name);}
void close(const Tensor& a,const Tensor& b,const std::string& name){
  require(a.defined()==b.defined(),name+" connection");if(!a.defined())return;
  require(a.sizes()==b.sizes()&&a.requires_grad()==b.requires_grad(),name+" metadata/history");
  const auto dtype=a.scalar_type();const double rtol=dtype==at::kHalf?2e-2:dtype==at::kFloat?2e-5:2e-9;
  const double atol=dtype==at::kHalf?2e-3:dtype==at::kFloat?2e-6:2e-11;
  require(at::allclose(a,b,rtol,atol),name+" value");
}
void vjp(const Tensor& a,const Tensor& b,const std::vector<Tensor>& roots,const std::string& label){
  require(a.requires_grad()==b.requires_grad(),label+" history");if(!a.requires_grad())return;
  for(double factor:{1.,0.}){auto bar=(at::arange(a.numel(),at::kDouble)*.03+.2).reshape(a.sizes()).to(a.options())*factor;
    auto ga=torch::autograd::grad({a},roots,{bar},true,false,true),gb=torch::autograd::grad({b},roots,{bar},true,false,true);
    for(size_t j=0;j<roots.size();++j)close(ga[j],gb[j],label+" root "+std::to_string(j));}
}
struct Fixture {Model model;std::shared_ptr<weighted::NodeBinding> binding;std::vector<col::Message> messages;std::vector<col::AggregateRequest> requests;std::vector<Tensor> roots;};
Fixture fixture(at::ScalarType dtype,double alpha=0){
  Fixture f;f.model=weighted_check::model(weighted_check::graph("fc","kv"),"kv",at::kCPU,dtype);
  f.binding=std::make_shared<weighted::NodeBinding>(weighted::node_binding(f.model.nodes[0]));f.binding->aggregate_alpha=alpha;
  for(Index label:{1,2})for(const auto* name:{"a","b","embedding"})
    f.model.nodes[0].extra["aggregate_program.transforms."+std::to_string(label)+"."+name]=f.model.nodes[0].extra.at(std::string("aggregate_program.transforms.0.")+name);
  auto message=[&](std::vector<Index> rows,std::vector<double> weights){
    auto m=at::tensor(weights,at::kDouble).to(dtype).set_requires_grad(true);
    auto z=(at::sin(at::arange(rows.size()*2,at::kDouble)*.3+.17)).reshape({static_cast<Index>(rows.size()),2}).to(dtype).set_requires_grad(true);
    f.messages.push_back({std::move(rows),m,z});f.roots.insert(f.roots.end(),{m,z});
  };
  message({0,2,5},{.3,0.,.7});message({1,2},{.5,.4});message({0,2},{0.,0.});message({3,7},{.2,.6});
  message({0,2,5},{.3,0.,.7});message({4,6},{-.2,.5});message({},{});
  // Same class and exact same payload may occur as two physical atoms.
  const std::vector<std::vector<std::pair<Index,Index>>> groups{{{0,0},{1,0},{0,0}},{{3,1}},{{2,2}},
    {{0,0},{1,0},{0,0}},{{4,0},{1,0},{4,0}},{{3,0},{5,0}},{{6,0}}};
  for(const auto& group:groups){col::AggregateRequest r{&f.model.nodes[0],f.binding.get(),{}};
    for(auto [index,label]:group)r.sources.push_back({&f.messages[index],label});f.requests.push_back(std::move(r));}
  for(const auto* name:{"a","b","embedding"})f.roots.push_back(f.model.nodes[0].extra.at(std::string("aggregate_program.transforms.0.")+name));
  return f;
}
Tensor source(const col::AggregateRequest& r,const col::Message& p,Index label){
  if(!r.binding->aggregate_alpha)return p.content;
  const auto prefix="aggregate_program.transforms."+std::to_string(label)+".";const auto& w=r.weights->extra;
  const auto epsilon=std::visit([](const auto& c){return c.epsilon;},r.binding->memory);
  auto n=p.content/at::sqrt(epsilon+p.content.square().mean(-1,true));
  auto h=at::matmul(n,w.at(prefix+"a").t())+w.at(prefix+"embedding");
  return p.content+r.binding->aggregate_alpha*at::matmul(h*at::sigmoid(h),w.at(prefix+"b").t());
}
col::Received literal(const col::AggregateRequest& r){
  col::Received result;std::set<Index> rows;std::vector<Tensor> values;
  for(const auto& [p,label]:r.sources){rows.insert(p->rows.begin(),p->rows.end());values.push_back(p->rows.empty()?Tensor{}:source(r,*p,label));}
  result.rows={rows.begin(),rows.end()};if(rows.empty())return result;
  std::vector<Tensor> masses,means;
  for(auto row:rows){std::vector<Tensor> m,z;
    for(size_t j=0;j<r.sources.size();++j){const auto* p=r.sources[j].first;auto found=std::lower_bound(p->rows.begin(),p->rows.end(),row);
      if(found!=p->rows.end()&&*found==row){const auto index=found-p->rows.begin();m.push_back(p->strength[index].clamp_min(0));z.push_back(values[j][index]);}}
    auto total=at::stack(m).sum();masses.push_back(total);
    if(total.detach().item<double>()>0){std::vector<Tensor> terms;for(size_t j=0;j<m.size();++j)terms.push_back(m[j]*z[j]);means.push_back(at::stack(terms).sum(0)/total);}
    else means.push_back(at::zeros({2},m[0].options()));
  }
  result.mass=at::stack(masses);result.content=at::stack(means);return result;
}
void check(at::ScalarType dtype,double alpha=0){
  auto f=fixture(dtype,alpha);
  work::packing::reset(true);auto actual=col::aggregate_columns_many(f.requests,Index{1}<<24);auto counts=work::packing::metrics();work::packing::reset(false);
  if(dtype==at::kHalf)require(!counts.count("packing/bucket/aggregate_tensor/calls"),"half entered new mean kernel");
  else require(counts.at("packing/bucket/aggregate_tensor/calls")==1&&counts.at("packing/bucket/aggregate_tensor/rows")==5,"real aggregate groups/cache identity");
  require(counts.at("packing/bucket/aggregate_exact_reuse/rows")==2,"exact receiver alias was not reused");
  auto bounded=col::aggregate_columns_many(f.requests,1);std::vector<col::Received> expected;
  for(const auto& r:f.requests)expected.push_back(literal(r));
  for(size_t g=0;g<actual.size();++g){require(actual[g].rows==expected[g].rows,"physical row support differs");
    close(actual[g].mass,expected[g].mass,"literal mass");close(actual[g].content,expected[g].content,"literal content");
    close(actual[g].mass,bounded[g].mass,"bounded mass");close(actual[g].content,bounded[g].content,"bounded content");
    if(actual[g].rows.empty())continue;
    vjp(actual[g].mass,expected[g].mass,f.roots,"literal mass VJP");vjp(actual[g].content,expected[g].content,f.roots,"literal content VJP");
    vjp(actual[g].content,bounded[g].content,f.roots,"column-root VJP");
  }
  vjp(actual[0].mass.sum()+actual[3].mass.sum()+actual[4].content.square().sum(),
    expected[0].mass.sum()+expected[3].mass.sum()+expected[4].content.square().sum(),f.roots,"independent/equal alias uses");
  auto mass_only=torch::autograd::grad({actual[0].mass.sum()},f.roots,{},true,false,true);
  require(!mass_only[1].defined()&&!mass_only[3].defined(),"mass output connected source values");
  require(!actual[2].content.requires_grad(),"zero-mass receiver content gained history");
  if(!alpha){auto single=torch::autograd::grad({actual[1].content.sum()},f.roots,{},true,false,true);
    require(single[6].defined()&&single[6].count_nonzero().item<Index>()==0,"single-source mass derivative not connected exact zero");}
  {at::NoGradGuard guard;auto plain=col::aggregate_columns_many(f.requests,Index{1}<<24);
    for(size_t i=0;i<plain.size();++i)if(!plain[i].rows.empty()){close(plain[i].mass,expected[i].mass.detach(),"no-grad mass");close(plain[i].content,expected[i].content.detach(),"no-grad content");}}
}
void guards(){
  auto f=fixture(at::kDouble);auto output=col::aggregate_columns_many(f.requests,Index{1}<<24);
  bool first_order=false;try{torch::autograd::grad({output[0].content.sum()},{f.messages[0].strength},{},true,true,true);}catch(const std::exception& e){first_order=std::string(e.what()).find("first-order")!=std::string::npos;}
  require(first_order,"aggregate boundary accepted higher-order VJP");
  {at::NoGradGuard guard;f.messages[2].content.add_(.01);}
  bool version=false;try{torch::autograd::grad({output[0].content.sum()},{f.messages[0].strength},{},false,false,true);}catch(const std::exception&){version=true;}
  require(version,"unobserved original aggregate version not guarded");
}
void budget_and_direction(){
  auto f=fixture(at::kDouble);work::packing::reset(true);auto bounded=col::aggregate_columns_many(f.requests,24000);
  auto counts=work::packing::metrics();work::packing::reset(false);
  require(counts.at("packing/bucket/aggregate_tensor/calls")==1&&counts.at("packing/bucket/aggregate_tensor/rows")==4,"unique source/common-row budget was not enforced");
  f.messages[3].strength.set_requires_grad(false);f.messages[3].content.set_requires_grad(false);
  auto output=col::aggregate_columns_many(f.requests,Index{1}<<24);
  require(!output[1].mass.requires_grad()&&!output[1].content.requires_grad(),"frozen receiver gained roots");
  auto mass=f.messages[0].strength,value=f.messages[0].content;
  auto gradients=torch::autograd::grad({output[0].content.square().sum()},{mass,value});
  auto md=at::tensor({.03,0.,-.02},at::kDouble),vd=at::full_like(value,.02);const double step=1e-5;
  auto initial_m=mass.detach().clone(),initial_v=value.detach().clone();std::vector<Tensor> samples;
  {at::NoGradGuard guard;for(double sign:{1.,-1.}){mass.copy_(initial_m+sign*step*md);value.copy_(initial_v+sign*step*vd);
    samples.push_back(col::aggregate_columns_many(f.requests,Index{1}<<24)[0].content.square().sum());}}
  close((gradients[0]*md).sum()+(gradients[1]*vd).sum(),(samples[0]-samples[1])/(2*step),"smooth directional derivative");
}
void source_owners(){
  auto f=fixture(at::kDouble,.25);std::vector<col::SourceRequest> requests;std::vector<std::vector<size_t>> uses;
  for(const auto& r:f.requests){std::vector<size_t> slots;for(const auto& [p,label]:r.sources)if(!p->rows.empty()){
    slots.push_back(requests.size());requests.push_back({p->content,col::source_parameters(*r.weights,label),.25,.1});}uses.push_back(std::move(slots));}
  auto owners=col::source_blocks(requests,Index{1}<<24);
  require(owners.values.size()==6&&owners.owners.size()==13,"source blocks published physical-use owners");
  std::vector<col::AggregateTensorRequest> entries;
  for(size_t i=0;i<f.requests.size();++i){col::AggregateTensorRequest r;size_t j=0;std::set<Index> rows;
    for(const auto& [p,label]:f.requests[i].sources)if(!p->rows.empty()){
      r.sources.push_back(p);r.values.push_back(owners.values[owners.owners[uses[i][j++]]]);rows.insert(p->rows.begin(),p->rows.end());}
    if(!rows.empty()){r.rows={rows.begin(),rows.end()};entries.push_back(std::move(r));}}
  std::vector<const col::AggregateTensorRequest*> pointers;for(const auto& r:entries)pointers.push_back(&r);
  auto plan=col::aggregate_source_layout(pointers);
  require(plan.groups==6&&plan.sources==6&&plan.rows==8&&plan.originals.size()==12,"incidence incorrectly merged independent source roots");
  require(plan.incidence[0][0]==2&&plan.incidence[0][1]==1&&plan.incidence[3]==plan.incidence[0],"physical repeated uses lost multiplicity");
  plan.materialize();require(plan.index.sizes()==at::IntArrayRef({6,8})&&plan.counts.sizes()==at::IntArrayRef({6,6})
    &&plan.counts.scalar_type()==at::kLong&&plan.inverse.scalar_type()==at::kLong,"source-owner row map differs");
}
}
int main(){try{at::set_num_threads(1);for(auto dtype:{at::kDouble,at::kFloat}){check(dtype);check(dtype,.25);}check(at::kHalf);guards();budget_and_direction();source_owners();
  std::cout<<"Aggregate tensor CPU: literal atoms, ragged rows/sources, exact aliases, roots, centered VJP and directional derivative passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
