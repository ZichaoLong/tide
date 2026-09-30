#include "portable_torch/runtime.hpp"
#include "tide/placement.h"
#include "tide/greedy.h"
#include "tide/stream.h"
#include "tide/read.h"
#include "tide/region.h"
#include "tide/optimizer.h"
#include "tide/parameters.h"
#include "../bench/streaming.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void close(const Tensor& a, const Tensor& b) {
  require(a.defined() == b.defined(), "placement gradient connectivity");
  if (!a.defined()) return;
  require(a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes(), "placement tensor metadata");
  const bool fp64 = a.scalar_type() == at::kDouble;
  require(at::allclose(a.detach().cpu(), b.detach().cpu(), fp64 ? 1e-8 : 1e-5, fp64 ? 1e-10 : 1e-6), "placement tensor value");
}
template<class F> void refuses(F action, const std::string& marker) {
  try { action(); } catch (const std::exception& e) { if (std::string(e.what()).find(marker) != std::string::npos) return; throw; }
  throw std::runtime_error("missing placement refusal: " + marker);
}
Graph graph_for(const std::string& selector, const std::string& read) {
  Graph g; g.nodes = {{0},{0},{1},{2}}; g.regions = {{1,true,true,read,selector},{1},{1}};
  g.edges = {{0,2,2},{0,2,2},{1,2,1},{2,0,3}}; g.inputs = {0,1,3}; g.outputs = {0,2,3};
  for (auto& n : g.nodes) { n.memory = "ema"; n.full = "tanh"; }
  g.nodes[1].clear = true; g.compile(); return g;
}
Model model_for(const Graph& g, at::Device device, at::ScalarType dtype) {
  const auto opts = at::TensorOptions().device(device).dtype(dtype);
  auto leaf = [](Tensor x) { return x.set_requires_grad(true); };
  Model m;
  for (size_t n = 0; n < g.nodes.size(); ++n)
    m.nodes.push_back({leaf(at::zeros({2},opts)),leaf(at::eye(2,opts)*.125),
                       leaf(at::full({2},.03125,opts)),leaf(at::ones({2},opts))});
  m.regions.resize(g.regions.size());
  if (g.regions[0].selector == "tensor-history-v1") {
    m.regions[0].extra["alpha"] = leaf(at::full({},.5,opts));
    m.regions[0].extra["bias"] = leaf(at::tensor({-.2,.2},at::kDouble).to(opts));
  }
  for (size_t i=0;i<g.inputs.size();++i) m.input_scale.push_back(leaf(at::full({},.5,opts)));
  for (size_t i=0;i<g.outputs.size();++i) m.output_scale.push_back(leaf(at::full({},.5,opts)));
  for (size_t i=0;i<g.edges.size();++i) {
    m.agg_scale.push_back(leaf(at::full({},.5,opts)));m.edge_scale.push_back(leaf(at::full({},.25,opts)));
  }
  return m;
}
Continuation initial(const Graph& g, const Model& m) {
  Continuation q; q.identity=g.identity;
  auto kernel=make_region_kernel(g.regions[0]);
  auto h=kernel->initial(m.regions[0],region_layout(g,0),m.nodes[0].bias);
  const Index big=Index(1)<<54;
  h.node_maps["selected"]={{0,big+1},{1,big}};
  if(g.regions[0].selector=="lh-count-affect-v1")h.node_maps["affected"]={{0,big},{1,big+1}};
  q.history[{0,0}]=h;return q;
}
void detach(Continuation& q) {
  for(auto& [_,s]:q.states){s.value=s.value.detach();for(auto& [__,v]:s.slots)v=v.detach();}
  for(auto& [_,h]:q.history)for(auto& [__,v]:h.tensors)v=v.detach();
  for(auto& a:q.pending)a.value=a.value.detach();
}
Tensor objective(const Result& r) {
  std::vector<Tensor> terms;
  for(const auto& x:r.outputs)terms.push_back(x.value.square().sum());
  for(const auto& [_,s]:r.continuation.states)terms.push_back(s.value.square().sum()*.25);
  for(const auto& x:r.continuation.pending)terms.push_back(x.value.square().sum()*.125);
  return at::stack(terms).sum();
}
void schedules(at::Device device,at::ScalarType dtype,const ExecutionPlacement& placement,
               const std::string& selector,const std::string& read,bool packed) {
  auto g=graph_for(selector,read);
  // CPU FP64 Read with accelerator payload is a valid explicit mixed contract.
  if(placement.scoring_dtype=="float64")for(auto& n:g.nodes)n.readout="norm-fp64-v1";
  g.compile();auto cpu=model_for(g,at::Device(at::kCPU),dtype),target=model_for(g,device,dtype);
  auto placed=place_model(g,target,placement);const auto resolved=resolve_placement(placement,device);
  auto cp=cpu.parameters(),tp=placed.parameters();
  require(tp.alias_partitions()==target.parameters().alias_partitions(),"placement changed aliases");
  for(const auto& owner:tp.owners())require(owner.value.is_same(target.parameters().value(owner.canonical)),"placement copied a parameter leaf");
  OptimizerGroup group;group.lr=.0001;group.momentum=.25;
  for(const auto& p:cp.owners())group.parameters.push_back(p.canonical);
  SGD co(cp,{group}),to(tp,{group});Options opts;opts.mode="hst";
  Streaming reference(g,cpu,opts);opts.packed=packed;Greedy candidate(g,placed,opts);Streaming stream(g,placed,opts);
  auto a=initial(g,cpu),b=initial(g,target);
  for(Index cycle=0;cycle<3;++cycle) {
    co.zero_grad();to.zero_grad();std::vector<Tensor> x,y;
    std::vector<External> xs,ys;
    for(Index port=0;port<3;++port) {
      auto v=at::tensor({.25+port*.125,-.125+cycle*.03125},at::kDouble).to(dtype).set_requires_grad(true);
      auto w=v.detach().to(device).clone().set_requires_grad(true);x.push_back(v);y.push_back(w);
      xs.push_back({0,port,cycle,cycle*4,v});ys.push_back({0,port,cycle,cycle*4,w});
    }
    auto expected=reference.run(a,xs,(cycle+1)*4,(cycle+1)*4);
    auto actual=packed?candidate.run(b,ys,(cycle+1)*4,(cycle+1)*4):stream.run(b,ys,(cycle+1)*4,(cycle+1)*4);
    tide_bench::compare(actual,expected,true,dtype);
    for(const auto& e:actual.trace)require(e.descriptor.device()==resolved.read&&e.content.device()==device&&e.control.device()==device,"placement runtime devices");
    // Independent region roots must preserve None versus connected zero.
    for(const auto& roots:{std::pair<Tensor,Tensor>{expected.trace[0].descriptor,actual.trace[0].descriptor},
                           std::pair<Tensor,Tensor>{expected.trace[0].control,actual.trace[0].control}}) {
      auto gx=torch::autograd::grad({roots.first},x,{},true,false,true);
      auto gy=torch::autograd::grad({roots.second},y,{},true,false,true);
      for(size_t i=0;i<x.size();++i)close(gy[i],gx[i]);
      require(!gy[2].defined(),"placement joined independent region autograd roots");
    }
    auto wanted=objective(expected),got=objective(actual);close(got,wanted);wanted.backward();got.backward();
    for(size_t i=0;i<x.size();++i)close(y[i].grad(),x[i].grad());
    for(const auto& p:cp.owners())close(tp.value(p.canonical).grad(),p.value.grad());
    a=expected.continuation;b=actual.continuation;detach(a);detach(b);co.step();to.step();
    for(const auto& p:cp.owners())close(tp.value(p.canonical),p.value);
    require(to.state().size()==co.state().size(),"placement optimizer owner set");
    for(const auto& [name,s]:co.state()) {
      const auto& t=to.state().at(name);require(s.step==t.step,"placement optimizer counter");close(t.momentum_buffer,s.momentum_buffer);
    }
  }
}
void selection(at::Device device,at::ScalarType dtype,const ExecutionPlacement& p) {
  for(const std::string profile:{"count-v1","positive-v1","lh-count-affect-v1"})for(Index budget:{1,2,4}) {
    Graph g;g.nodes={{0},{0},{0},{0}};g.regions={{budget,true,true,"content",profile}};g.compile();
    auto m=model_for(g,device,dtype),placed=place_model(g,m,p);auto layout=region_layout(g,0);
    auto base=make_region_kernel(g.regions[0]);auto old=base->initial(m.regions[0],layout,m.nodes[0].bias);
    const Index big=Index(1)<<54;old.node_maps["selected"]={{0,big+1},{1,big},{2,big},{3,big}};
    if(profile=="lh-count-affect-v1")old.node_maps["affected"]={{0,big+2},{1,big},{2,big+1},{3,big}};
    auto source=at::tensor({1000.,0.,1.,1.},at::TensorOptions().dtype(dtype));
    std::vector<Candidate> reference,actual;auto desc=resolve_placement(p,device).read;
    for(Index i=0;i<4;++i){reference.push_back({i,source[i]});actual.push_back({i,source[i].to(desc)});}
    const auto expected=base->step(m.regions[0],{old,0,reference,layout,source.options()});
    const auto got=placed.regions[0].kernel->step(m.regions[0],{old,0,actual,layout,m.nodes[0].bias.options()});
    require(got.active==expected.active&&got.history.node_maps==expected.history.node_maps,"placement stable exact int64 selection");
    for(const auto& [node,value]:expected.controls)close(got.controls.at(node),value);
    // Layout, budget and count priority belong to the call's graph, not the
    // parameter view. Reuse one placed model under another compatible policy.
    auto alternate=g.regions[0];alternate.budget=budget==1?4:1;
    if(profile!="lh-count-affect-v1")alternate.count_priority=false;
    RegionLayout changed{alternate,layout.members};
    const auto want=base->step(m.regions[0],{old,0,reference,changed,source.options()});
    const auto have=placed.regions[0].kernel->step(m.regions[0],{old,0,actual,changed,m.nodes[0].bias.options()});
    require(have.active==want.active&&have.history.node_maps==want.history.node_maps,"placement cached graph-owned region policy");
  }
}
class CustomRead final:public ReadKernel {
 public:Tensor step(const NodeWeights&,const ReadInput& r)const override{return r.content.value.sum();}
  void validate_weights(const NodeWeights&)const override{}
};
void refusals(at::Device device,at::ScalarType dtype) {
  auto g=graph_for("count-v1","content");auto m=model_for(g,device,dtype);ExecutionPlacement p;
  m.nodes[0].read_kernel=std::make_shared<CustomRead>();
  refuses([&]{place_model(g,m,p);},"custom");m.nodes[0].read_kernel.reset();
  p.preset="unknown";refuses([&]{place_model(g,m,p);},"unknown");p.preset="native";
  for(auto& n:g.nodes)n.readout="norm-fp32-v1";g.compile();p.scoring_dtype="float64";
  refuses([&]{place_model(g,m,p);},"conflicts");p.scoring_dtype="profile";
  if(!device.is_cpu()) {
    p.preset="resident";refuses([&]{place_model(g,m,p);},"device-resident");p.preset="mixed-c";
    for(auto& n:g.nodes)n.readout="norm-fp64-v1";g.compile();refuses([&]{place_model(g,m,p);},"FP64");
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    bool profile_smoke=false;std::vector<char*> arguments{argv[0]};
    for(int i=1;i<argc;++i)if(std::string(argv[i])=="--profile-smoke")profile_smoke=true;else arguments.push_back(argv[i]);
    auto args=portable_torch::parse_cli(arguments.size(),arguments.data(),true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);std::cout<<"--profile-smoke  bounded representative placement trace (not the full gate)\n";return 0;}
    const auto device=portable_torch::resolve_device(args);
    if(args.dtype!=at::kFloat&&args.dtype!=at::kDouble)throw std::invalid_argument("placement gate requires FP32/FP64");
    at::set_num_threads(1);at::set_num_interop_threads(1);
    std::vector<ExecutionPlacement> choices;
    for(const std::string preset:device.is_cpu()?std::vector<std::string>{"native","cpu"}:std::vector<std::string>{"mixed-a","mixed-b","mixed-c"}) {
      ExecutionPlacement p;p.preset=preset;choices.push_back(p);
    }
    if(!device.is_cpu()) {
      ExecutionPlacement a;a.read="cpu";a.control="payload";a.selection="payload";choices.push_back(a);
      a.read="payload";a.control="cpu";choices.push_back(a);
    }
    if(profile_smoke){ExecutionPlacement p;p.preset=device.is_cpu()?"cpu":"mixed-c";choices={p};}
    Index cases=0;
    for(const auto& p:choices) {
      selection(device,args.dtype,p);
      for(const std::string selector:{"count-v1","positive-v1","lh-count-affect-v1","tensor-history-v1"})
      for(const std::string read:{"content","old","proposal"})for(bool packed:{false,true}) {
        if(profile_smoke && (read!="proposal"||!packed||(selector!="count-v1"&&selector!="tensor-history-v1")))continue;
        try{schedules(device,args.dtype,p,selector,read,packed);}catch(...){std::cerr<<"placement="<<p.preset<<" read="<<p.read<<" control="<<p.control<<" selection="<<p.selection<<" selector="<<selector<<" mode="<<read<<" packed="<<packed<<'\n';throw;}++cases;
      }
    }
    ExecutionPlacement fp64;fp64.read=fp64.control=fp64.selection="cpu";fp64.scoring_dtype="float64";
    schedules(device,args.dtype,fp64,"count-v1","proposal",true);++cases;
    refusals(device,args.dtype);portable_torch::synchronize(device);runtime.close();
    std::cout<<"standalone-placement: passed schedules="<<cases<<" updates="<<cases*3<<" exact_int64_selection_and_refusals=passed scope="<<(profile_smoke?"profile-smoke":"full")<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
