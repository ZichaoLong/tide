#include "training_test.h"
#include "tide/fiber_attention.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace tide::device_online::test {
void train_trajectory(at::Device device,Fixture f,bool prefill,ResidentOptimizerKind kind,at::ScalarType dtype) {
  train_trajectory(device,std::move(f),prefill,kind,dtype,1e-8);
}
void train_trajectory(at::Device device,Fixture f,bool prefill,ResidentOptimizerKind kind,at::ScalarType dtype,double adam_epsilon,bool conditioned_controls,Options options) {
  at::NoGradGuard guard;const auto width=f.model.nodes.at(0).bias.numel();f.model=test::train_model(f.model,at::kFloat);
  auto cpu=f;cpu.model=test::train_model(f.model,dtype);cpu.initial=test::train_boundary(f.initial,dtype);
  auto registry=cpu.model.parameters(true);OptimizerGroup group;
  group.lr=.001;group.momentum=.875;group.nesterov=true;group.weight_decay=.0125;group.amsgrad=true;group.eps=adam_epsilon;
  for(const auto& owner:registry.owners())group.parameters.push_back(owner.canonical);
  std::unique_ptr<NamedOptimizer> optimizer;
  if(kind==ResidentOptimizerKind::sgd)optimizer=std::make_unique<SGD>(registry,std::vector<OptimizerGroup>{group});
  else optimizer=std::make_unique<AdamW>(registry,std::vector<OptimizerGroup>{group});
  ResidentTrainingLimits limits;limits.forward.prefill=prefill;limits.forward.trace=512;limits.forward.full_chunk_rows=3;
  limits.reverse_chunk_rows=3;
  if(std::any_of(f.graph.nodes.begin(),f.graph.nodes.end(),[](const auto& n){return !n.identity&&(n.memory=="attention"||is_fiber_attention_profile(n.memory));})) {
    limits.forward.queue=96;limits.forward.arrivals=192;limits.forward.outputs=192;limits.forward.trace=512;
    limits.forward.kv_rows=32;limits.forward.kv_trace_rows=8192;limits.forward.attention_chunk_rows=3;
    limits.forward.attention_key_rows=2;limits.forward.workspace_bytes=256*1024*1024;
    limits.backward_bytes=Index(1)*1024*1024*1024;
    if(std::any_of(f.graph.nodes.begin(),f.graph.nodes.end(),[](const auto& n){return !n.identity&&is_fiber_attention_profile(n.memory);}))
      limits.forward.kv_rows=128; // Persistent rows span all four updates.
  }
  limits.forward.mode=options.mode;limits.forward.zeta=options.zeta;
  if(width>3) {
    limits.forward.workspace_bytes=512*1024*1024;
    // Four retained reverse programs each reserve disjoint component budgets.
    // SwiGLU has three D x 2D banks plus matrix/transposed/partial workspaces.
    if(std::any_of(f.graph.nodes.begin(),f.graph.nodes.end(),[](const auto& n){return !n.identity&&n.full=="swiglu";}))
      limits.backward_bytes=Index(2)*1024*1024*1024;
  }
  auto session=std::make_unique<ResidentTrainingSession>(f.graph,f.model,f.initial,device,kind,std::vector<OptimizerGroup>{group},limits);
  const auto original=session->checkpoint();
  auto actual_parameters=original.parameters;
  std::map<std::pair<Index,Index>,Index> positions;
  for(const auto& x:f.input)++positions[{x.batch,x.port}];
  for(int step=0;step<4;++step) {
    const auto start=session->cut();const int mode=step==1?5:step==2?0:4;
    std::vector<External> all=f.input;for(auto& x:all){x.time+=step*11;x.position+=step*positions.at({x.batch,x.port});}
    cpu.input=all;auto ref=test::retained_reference(cpu,mode,dtype,options);
    std::vector<ResidentCotangents> roots;int w=0;Index cut=start;
    for(auto stop:test::retained_stops(start)) {
      std::vector<External> input;for(auto x:all)if(cut<=x.time&&x.time<stop){if(step==1)x.value=x.value.to(device);input.push_back(x);}
      auto window=session->advance(input,stop,stop);roots.push_back(test::train_roots(window,w,mode));
      if(dtype==at::kFloat)try{train_forward_compare(session->result(),ref.windows[w],cpu.graph,conditioned_controls);}
        catch(...){
          std::cerr<<"training phase=forward step="<<step<<" window="<<w<<" width="<<width<<'\n';
          const auto actual=session->result();const auto& expected=ref.windows[w];
          for(size_t i=0;i<std::min(actual.trace.size(),expected.trace.size());++i) {
            const auto& a=actual.trace[i];const auto& b=expected.trace[i];
            if(!at::allclose(a.descriptor,b.descriptor,1e-5,1e-6)||!at::allclose(a.control,b.control,1e-5,1e-6)) {
              const auto& mode=cpu.graph.regions[cpu.graph.nodes[b.node].region].read_mode;
              auto x=mode=="content"?a.content:mode=="old"?a.old.value:a.proposal;
              auto y=mode=="content"?b.content:mode=="old"?b.old.value:b.proposal;
              auto weight=actual_parameters.at(registry.canonical_name("nodes."+std::to_string(b.node)+".read"));
              std::cerr<<std::setprecision(17)<<"descriptor node="<<b.node<<" time="<<b.time<<" actual="<<a.descriptor.item<double>()
                <<" reference="<<b.descriptor.item<double>()<<" actual_control="<<a.control.item<double>()<<" reference_control="<<b.control.item<double>()
                <<" actual_dot64="<<(x.to(at::kDouble)*weight.to(at::kDouble)).sum().item<double>()
                <<" reference_dot64="<<(y.to(at::kDouble)*cpu.model.nodes[b.node].read.to(at::kDouble)).sum().item<double>()
                <<" absolute_terms="<<(x.to(at::kDouble)*weight.to(at::kDouble)).abs().sum().item<double>()<<'\n';
            }
          }
          throw;
        }
      ++w;cut=stop;
    }
    test::train_reject([&]{session->checkpoint();},"checkpoint accepted unconsumed windows");
    test::train_reject([&]{session->step();},"optimizer crossed outstanding windows");
    auto bad=roots;std::swap(bad[0],bad[1]);test::train_reject([&]{session->backward(bad);},"out-of-order roots accepted");
    auto gradient=session->backward(roots);
    try{test::train_gradients(gradient,ref,cpu);}catch(...){std::cerr<<"training phase=gradients step="<<step<<'\n';throw;}
    test::train_reject([&]{session->backward(roots);},"backward reused consumed roots");
    test::train_reject([&]{session->advance({},cut,cut);},"advance bypassed unconsumed gradients");
    for(const auto& o:registry.owners())o.value.mutable_grad()=ref.gradients.at(o.canonical);
    optimizer->step();const auto updated=session->step();test::train_require(updated.applied&&updated.generation==step+1,"public optimizer did not update");
    auto checkpoint=session->checkpoint();
    actual_parameters=checkpoint.parameters;for(auto& [_,x]:actual_parameters)x=x.clone();
    try{test::train_checkpoint(checkpoint,cpu.model,*optimizer);}catch(...){std::cerr<<"training phase=checkpoint step="<<step<<'\n';throw;}
    if(step==0)test::train_require(!at::equal(checkpoint.state.values,original.state.values)&&checkpoint.state.steps.gt(0).any().item<bool>(),"trajectory performed no real parameter updates");
    cpu.initial=test::train_boundary(ref.windows.back().continuation,dtype);
    if(step==1) {
      session->close();session=std::make_unique<ResidentTrainingSession>(f.graph,f.model,checkpoint,device,limits);
      auto restored=session->checkpoint();test::train_require(at::equal(checkpoint.state.corrections,restored.state.corrections),"Adam correction restart changed");
      for(auto& [_,p]:checkpoint.parameters)p.fill_(123); // Exports must own storage.
    }
  }
  session->close();session->close();test::train_reject([&]{session->checkpoint();},"closed training owner exported state");
}
} // namespace tide::device_online::test
