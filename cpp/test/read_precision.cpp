// Independent standalone Read-precision gates; no Python adapter at runtime.
#include "portable_torch/runtime.hpp"
#include "tide/greedy.h"
#include "tide/stream.h"
#include "tide/read.h"
#include "tide/checkpoint.h"
#include "../bench/streaming.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>

namespace {
using namespace tide;
void require(bool value,const char* reason) { if(!value)throw std::runtime_error(reason); }
void close(const Tensor& a,const Tensor& b) {
  require(a.defined()==b.defined(),"norm32 gradient connectivity");
  if(!a.defined())return;
  require(a.scalar_type()==b.scalar_type()&&a.sizes()==b.sizes(),"norm32 tensor metadata");
  const double tolerance=a.scalar_type()==at::kHalf?1e-3:1e-6;
  require(at::allclose(a.detach().to(at::kCPU),b.detach().to(at::kCPU),1e-5,tolerance),"norm32 tensor value");
}
void analytic(at::Device device,at::ScalarType dtype) {
  auto options=at::TensorOptions().device(device).dtype(dtype);
  Node node{0};node.readout="norm-fp32-v1";auto kernel=make_read_kernel(node);
  require(kernel->descriptor_dtype(dtype)==at::kFloat&&kernel->joint_batch(),"norm32 declared precision/batch");
  NodeWeights weights;weights.read=at::ones({2},options).set_requires_grad(true);
  auto x=at::tensor({3.f,4.f},at::kFloat).to(options).set_requires_grad(true);
  auto zero=at::zeros({2},options).set_requires_grad(true);
  std::vector<ReadInput> requests{{nullptr,0,{x}},{nullptr,1,{zero}}};
  auto values=kernel->batch(weights,requests);
  require(values.size()==2,"norm32 batch shape");
  for(size_t i=0;i<values.size();++i) {
    require(values[i].scalar_type()==at::kFloat&&values[i].device()==device,"norm32 descriptor placement");
    close(values[i],kernel->step(weights,requests[i]));
    close(values[i],at::full({},i?0:5,options.dtype(at::kFloat)));
    auto grad=torch::autograd::grad({values[i]},{x,zero,weights.read},{},true,false,true);
    // Direct batch has ordinary tensor connectivity. The public packed replay
    // below separately checks per-event disconnected owners.
    close(grad[0],(i?at::zeros({2},at::kFloat):at::tensor({.6f,.8f},at::kFloat)).to(options));
    close(grad[1],at::zeros({2},options));
    require(!grad[2].defined(),"norm32 unused linear weights acquired a gradient");
  }
}
Model model_for(const Graph& g,at::Device device,at::ScalarType dtype) {
  auto options=at::TensorOptions().device(device).dtype(dtype);
  auto leaf=[](Tensor x){return x.set_requires_grad(true);};Model m;
  for(size_t n=0;n<g.nodes.size();++n) {
    NodeWeights w{leaf(at::zeros({2},options)),leaf(at::eye(2,options)),
      leaf(at::zeros({2},options)),leaf(at::ones({2},options))};
    w.extra["add_retention"]=leaf(at::full({},.5,options));m.nodes.push_back(w);
  }
  for(auto& scales:{&m.input_scale,&m.output_scale})
    for(int i=0;i<2;++i)scales->push_back(leaf(at::full({},.5,options)));
  for(size_t i=0;i<g.edges.size();++i) {
    m.agg_scale.push_back(leaf(at::full({},.5,options)));
    m.edge_scale.push_back(leaf(at::full({},.25,options)));
  }
  return m;
}
void detach(Continuation& q) {
  for(auto& [_,s]:q.states)s.value=s.value.detach();
  for(auto& a:q.pending)a.value=a.value.detach();
}
Tensor loss(const Result& result) {
  std::vector<Tensor> terms;
  for(const auto& [_,s]:result.continuation.states)terms.push_back(s.value.square().sum());
  for(const auto& x:result.outputs)terms.push_back(x.value.square().sum());
  for(const auto& x:result.continuation.pending)terms.push_back(x.value.square().sum());
  return at::stack(terms).sum();
}
void schedules(at::Device device,at::ScalarType dtype,const std::string& mode,bool packed) {
  Graph g;g.nodes={{0},{0},{1}};g.regions={{1,false,false,mode},{1,true,true,mode}};
  g.edges={{0,2,2},{0,2,2},{1,2,1},{2,0,3}};g.inputs={0,1};g.outputs={0,2};
  for(auto& n:g.nodes){n.memory="lh-add-repeat-v1";n.full="identity";n.readout="norm-fp32-v1";}
  g.nodes[0].clear=true;g.compile();
  auto cpu=model_for(g,at::Device(at::kCPU),dtype),target=model_for(g,device,dtype);
  auto cp=cpu.parameters(),tp=target.parameters();OptimizerGroup group;group.lr=.0001;group.eps=1e-5;
  for(const auto& p:cp.owners())group.parameters.push_back(p.canonical);
  AdamW co(cp,{group}),to(tp,{group});
  Options options;options.mode="hst";Streaming reference(g,cpu,options);
  options.packed=packed;options.workers=packed?2:1;Greedy candidate(g,target,options);
  Continuation a,b;a.identity=b.identity=g.identity;
  for(Index cycle=0;cycle<3;++cycle) {
    co.zero_grad();to.zero_grad();
    auto x=at::tensor({.25,.5,.75,.125},at::TensorOptions().dtype(dtype)).reshape({2,2}).set_requires_grad(true);
    auto y=x.detach().to(device).clone().set_requires_grad(true);
    auto xs=std::vector<External>{{0,0,cycle,cycle*5,x[0]},{0,1,cycle,cycle*5,x[1]}};
    auto ys=std::vector<External>{{0,0,cycle,cycle*5,y[0]},{0,1,cycle,cycle*5,y[1]}};
    auto expected=reference.run(a,xs,(cycle+1)*5,(cycle+1)*5);
    auto actual=candidate.run(b,ys,(cycle+1)*5,(cycle+1)*5);
    tide_bench::compare(actual,expected,true,dtype,device);
    for(const auto& e:actual.trace)require(e.descriptor.scalar_type()==at::kFloat,"norm32 trace precision");
    // An isolated descriptor must not connect the other external owner.
    const auto& ed=expected.trace[0].descriptor;const auto& ad=actual.trace[0].descriptor;
    require(ed.requires_grad()==ad.requires_grad(),"norm32 descriptor root connectivity");
    if(ed.requires_grad()) {
      auto gx=torch::autograd::grad({ed},{x},{},true,false,true).at(0);
      auto gy=torch::autograd::grad({ad},{y},{},true,false,true).at(0);close(gy,gx);
    }
    auto wanted=loss(expected),got=loss(actual);close(got,wanted);wanted.backward();got.backward();
    close(y.grad(),x.grad());
    for(const auto& owner:cp.owners())close(tp.value(owner.canonical).grad(),owner.value.grad());
    for(const auto& w:target.nodes)require(!w.read.grad().defined(),"norm32 graph connected linear Read weights");
    a=expected.continuation;b=actual.continuation;detach(a);detach(b);co.step();to.step();
    for(const auto& owner:cp.owners())close(tp.value(owner.canonical),owner.value);
    require(to.state().size()==co.state().size(),"norm32 optimizer owner count");
    for(const auto& [name,s]:co.state()) {
      const auto& t=to.state().at(name);require(t.step==s.step,"norm32 optimizer counter");
      close(t.exp_avg,s.exp_avg);close(t.exp_avg_sq,s.exp_avg_sq);
    }
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    auto device=portable_torch::resolve_device(args);
    if(args.dtype!=at::kFloat&&args.dtype!=at::kDouble)throw std::invalid_argument("norm32 schedule gate requires FP32/FP64 payload");
    at::set_num_threads(1);at::set_num_interop_threads(1);
    analytic(device,args.dtype);analytic(device,at::kHalf);
    for(const std::string mode:{"content","old","proposal"})for(bool packed:{false,true})schedules(device,args.dtype,mode,packed);
    portable_torch::synchronize(device);runtime.close();
    std::cout<<"standalone-norm32: passed schedules=6 updates=18 analytic_payloads=2\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
