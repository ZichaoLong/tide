#include "portable_torch/runtime.hpp"
#include "tide/ownership.h"
#include "tide/placement.h"
#include "tide/settle.h"
#include "tide/optimizer.h"
#include "../bench/streaming.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <iostream>

namespace {
using namespace tide;
void require(bool value,const char* why) { if(!value)throw std::runtime_error(why); }
void close(const Tensor& a,const Tensor& b) {
  require(a.defined()==b.defined(),"owner gradient presence mismatch");
  if(!a.defined())return;
  require(a.scalar_type()==b.scalar_type()&&a.sizes()==b.sizes(),"owner tensor metadata mismatch");
  const bool fp64=a.scalar_type()==at::kDouble;
  require(at::allclose(a.detach().cpu(),b.detach().cpu(),fp64?1e-8:1e-5,fp64?1e-10:1e-6),"owner tensor value mismatch");
}
Graph graph_for(const std::string& family,const std::string& memory) {
  Graph g;g.nodes={{0},{0},{1},{2}};
  for(auto& n:g.nodes)n.memory=memory;
  g.regions={{2,true,true,"proposal","tensor-history-v1"},{1},{1}};
  g.edges={{0,2,2},{0,2,2},{1,2,family=="settle"?2:1}};
  if(family=="pdg")g.edges.push_back({2,0,3});
  g.inputs={0,1,3};g.outputs={0,2,3};g.compile();return g;
}
Model model_for(const Graph& g,at::ScalarType dtype) {
  const auto opts=at::TensorOptions().dtype(dtype).device(at::kCPU);
  auto leaf=[](Tensor x){return x.set_requires_grad(true);};
  Model m;
  for(const auto& n:g.nodes) {
    NodeWeights w{leaf(at::zeros({2},opts)),leaf(at::eye(2,opts)*.125),
                  leaf(at::full({2},.03125,opts)),leaf(at::ones({2},opts)*.25)};
    if(n.memory=="lh-add-repeat-v1")w.extra["add_retention"]=leaf(at::full({},.99,opts));
    else {
      w.extra["fiber_qkv"]=leaf(at::cat({at::eye(2,opts),at::eye(2,opts),at::eye(2,opts)},1)*.125);
      w.extra["fiber_out"]=leaf(at::eye(2,opts)*.125);
      w.extra["fiber_qkv_bias"]=leaf(at::zeros({6},opts));
      w.extra["fiber_out_bias"]=leaf(at::zeros({2},opts));
      w.extra["fiber_decay"]=leaf(at::full({},.01,opts));
    }
    m.nodes.push_back(w);
  }
  m.nodes[2].weight=m.nodes[1].weight;
  m.regions.resize(g.regions.size());
  m.regions[0].extra={{"alpha",leaf(at::full({},.5,opts))},{"bias",leaf(at::tensor({-.2,.2},at::kDouble).to(dtype))}};
  for(size_t i=0;i<g.inputs.size();++i)m.input_scale.push_back(leaf(at::full({},.5,opts)));
  m.input_scale[1]=m.input_scale[0];
  for(size_t i=0;i<g.outputs.size();++i)m.output_scale.push_back(leaf(at::full({},.5,opts)));
  for(size_t i=0;i<g.edges.size();++i) {
    m.edge_scale.push_back(leaf(at::full({},.25,opts)));
    m.agg_scale.push_back(leaf(at::full({},.5,opts)));
  }
  m.edge_scale[0]=m.agg_scale[0];return m;
}
Tensor loss(const Result& r) {
  // The test's loss explicitly gathers small scalar roots on CPU. This work is
  // not hidden in the runtime and these checks are not throughput measurements.
  std::vector<Tensor> terms;
  auto add=[&](const Tensor& v,double scale){terms.push_back(v.square().sum().cpu()*scale);};
  for(const auto& o:r.outputs)add(o.value,.7);
  for(const auto& [_,s]:r.continuation.states){add(s.value,.3);for(const auto& [__,v]:s.slots)add(v,.11);}
  for(const auto& [_,h]:r.continuation.history)for(const auto& [__,v]:h.tensors)add(v,.13);
  for(const auto& a:r.continuation.pending)add(a.value,.2);
  return at::stack(terms).sum()/100;
}
void detach(Continuation& q) {
  for(auto& [_,s]:q.states){s.value=s.value.detach();for(auto& [__,v]:s.slots)v=v.detach();}
  for(auto& [_,h]:q.history)for(auto& [__,v]:h.tensors)v=v.detach();
  for(auto& a:q.pending)a.value=a.value.detach();
}
void owners(const Graph& g,const Model& m,const Result& r,const ExecutionPlacement& p) {
  for(const auto& e:r.trace) {
    const auto d=m.nodes[e.node].bias.device();
    require(e.content.device()==d&&e.control.device()==d&&e.next.device()==d,"node did not execute on owner");
    require(e.descriptor.device()==resolve_placement(p,d).read,"Read did not execute at requested placement");
  }
  for(const auto& a:r.messages)require(a.value.device()==m.nodes[a.node].bias.device(),"message not at target owner");
  for(const auto& a:r.continuation.pending)require(a.value.device()==m.nodes[a.node].bias.device(),"pending not at target owner");
  for(const auto& [key,s]:r.continuation.states) {
    auto d=m.nodes[key.second].bias.device();require(s.value.device()==d,"state not at owner");
    for(const auto& [_,v]:s.slots)require(v.device()==d,"KV not at owner");
  }
  for(const auto& [key,h]:r.continuation.history)for(const auto& [_,v]:h.tensors)
    require(v.device()==region_reference(g,m,key.second).device(),"history not at region owner");
}
void check(at::Device first,at::Device peer,at::ScalarType dtype,const std::string& family,
           const std::string& memory,const ExecutionPlacement& placement,bool greedy) {
  auto g=graph_for(family,memory);auto cpu=model_for(g,dtype),original=model_for(g,dtype);
  auto target=place_payloads(g,original,{first,peer,peer,first});
  require(target.parameters().alias_partitions()==original.parameters().alias_partitions(),"placement lost aliases");
  target=place_model(g,target,placement);
  if(family=="settle") {
    SettleGraph spec(g,{1,3,2});cpu=spec.embed_model(cpu);target=spec.embed_model(target);g=spec.encoded_graph();
    target=place_model(g,target,placement);
  }
  auto cp=cpu.parameters(),tp=target.parameters();
  OptimizerGroup group;group.lr=.0001;group.eps=1e-5;group.weight_decay=.01;
  AdamW co(cp,{group}),to(tp,{group});
  Options opts;opts.mode="hst";opts.packed=false;Streaming reference(g,cpu,opts);
  opts.packed=true;opts.workers=2;Greedy candidate(g,target,opts);Streaming stream(g,target,opts);
  Continuation a,b;a.identity=b.identity=g.identity;
  const Index stride=family=="settle"?5:1;
  for(Index update=0;update<2;++update) {
    co.zero_grad();to.zero_grad();std::vector<Tensor> la,lb,x,y;
    for(Index window=0;window<2;++window) {
      const Index start=update*4+window*2,stop=(start+2)*stride;
      auto v=(at::arange(4,at::TensorOptions().dtype(dtype)).reshape({2,2})*.07+.25).set_requires_grad(true);
      auto w=v.detach().clone().set_requires_grad(true);x.push_back(v);y.push_back(w);
      std::vector<External> xs,ys;
      for(size_t p=0;p<g.inputs.size();++p)for(Index t=0;t<2;++t) {
        xs.push_back({0,static_cast<Index>(p),start+t,(start+t)*stride,v[t]});
        ys.push_back({0,static_cast<Index>(p),start+t,(start+t)*stride,w[t].to(target.nodes[g.inputs[p]].bias.device())});
      }
      auto expected=reference.run(a,xs,stop,stop);
      auto actual=greedy?candidate.run(b,ys,stop,stop):stream.run(b,ys,stop,stop);
      tide_bench::compare(actual,expected,true,dtype);owners(g,target,actual,placement);
      if(window==0) {
        size_t i=0;while(expected.trace.at(i).node!=0)++i;
        for(double factor:{1.,0.})for(bool control:{false,true}) {
          auto ar=control?expected.trace[i].control:expected.trace[i].descriptor;
          auto br=control?actual.trace[i].control:actual.trace[i].descriptor;
          std::vector<Tensor> av,bv;
          for(const auto& p:cp.owners()){av.push_back(p.value);bv.push_back(tp.value(p.canonical));}
          auto ag=torch::autograd::grad({ar.sum()*factor},av,{},true,false,true);
          auto bg=torch::autograd::grad({br.sum()*factor},bv,{},true,false,true);
          for(size_t j=0;j<ag.size();++j)close(ag[j],bg[j]);
          auto isolated=torch::autograd::grad({br.sum()*factor},{target.nodes[3].weight},{},true,false,true);
          require(!isolated[0].defined(),"cross-owner copy connected independent root");
        }
      }
      la.push_back(loss(expected));lb.push_back(loss(actual));a=expected.continuation;b=actual.continuation;
    }
    auto ax=torch::autograd::grad({la.back()},{x.front()},{},true,false,true);
    auto bx=torch::autograd::grad({lb.back()},{y.front()},{},true,false,true);
    close(ax[0],bx[0]);require(ax[0].defined()&&ax[0].abs().sum().item<double>()>0,"lost connected-window VJP");
    at::stack(la).sum().backward();at::stack(lb).sum().backward();
    for(size_t i=0;i<x.size();++i)close(x[i].grad(),y[i].grad());
    for(const auto& p:cp.owners())close(p.value.grad(),tp.value(p.canonical).grad());
    detach(a);detach(b);co.step();to.step();
    for(const auto& p:cp.owners())close(p.value,tp.value(p.canonical));
    require(co.state().size()==to.state().size(),"optimizer state owner mismatch");
    for(const auto& [name,s]:co.state()) {
      const auto& t=to.state().at(name);require(s.step==t.step,"optimizer counter mismatch");
      close(s.exp_avg,t.exp_avg);close(s.exp_avg_sq,t.exp_avg_sq);
      require(t.exp_avg.device()==tp.value(name).device(),"optimizer slot misplaced");
    }
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    bool profile_smoke=false;std::vector<char*> arguments{argv[0]};
    for(int i=1;i<argc;++i)if(std::string(argv[i])=="--profile-smoke")profile_smoke=true;else arguments.push_back(argv[i]);
    auto args=portable_torch::parse_cli(arguments.size(),arguments.data(),true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);std::cout<<"Requires two logical accelerator devices, or CPU FP32/FP64.\n--profile-smoke selects one bounded trace case, not the complete gate.\n";return 0;}
    auto first=portable_torch::resolve_device(args);
    auto peer=first.is_cpu()?first:at::Device(first.type(),first.index()+1);
    require(args.dtype==at::kFloat||args.dtype==at::kDouble,"owner gate requires FP32/FP64");
    at::set_num_threads(1);at::set_num_interop_threads(1);
    std::vector<c10::Stream> streams;
    if(!first.is_cpu()) {
      c10::impl::VirtualGuardImpl implementation(first.type());
      streams={implementation.getNewStream(first),implementation.getNewStream(peer)};
    }
    c10::MultiStreamGuard stream_guard(streams);
    if(!streams.empty()) {
      NodePool pool(2);pool.set_devices({first,peer});
      auto inherited=[&] {
        c10::impl::VirtualGuardImpl implementation(first.type());
        for(const auto& stream:streams)
          require(implementation.getStream(stream.device())==stream,"worker lost an owner stream");
      };
      pool.run({inherited,inherited});
    }
    Index count=0;
    for(const std::string preset:first.is_cpu()?std::vector<std::string>{"cpu"}:std::vector<std::string>{"mixed-a","mixed-b","mixed-c"})
    for(const std::string family:{"pdg","timed-dag","settle"})
    for(const std::string memory:{"lh-add-repeat-v1","lh-fiber-attention-sum-repeat-v1"})for(bool greedy:{false,true}) {
      if(profile_smoke&&((preset!="cpu"&&preset!="mixed-c")||family!="timed-dag"||memory=="lh-add-repeat-v1"||!greedy))continue;
      ExecutionPlacement p;p.preset=preset;
      try{check(first,peer,args.dtype,family,memory,p,greedy);}catch(...){std::cerr<<preset<<' '<<family<<' '<<memory<<" greedy="<<greedy<<'\n';throw;}
      ++count;
    }
    portable_torch::synchronize(first);portable_torch::synchronize(peer);runtime.close();
    std::cout<<"payload-ownership: passed cases="<<count<<" updates="<<count*2<<" connected_windows="<<count*4
             <<" scope="<<(profile_smoke?"profile-smoke":"full")<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
