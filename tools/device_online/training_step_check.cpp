#include "device_backend.h"
#include "device_optimizer.h"
#include "graph_vjp_fixture.h"
#include "full_vjp_fixture.h"
#include "tide/stream.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>

namespace {
using namespace tide;using namespace tide::device_online;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
void alias(Model& m,int shape) {
  m.nodes[2].weight=m.nodes[0].weight;m.nodes[2].bias=m.nodes[0].decay;m.nodes[3].read=m.nodes[0].bias;
  m.input_scale[1]=m.agg_scale[0];m.output_scale[2]=m.edge_scale[1];
  if(shape==3)m.input_scale[0]=m.nodes[0].extra.at("add_retention");
}
Model clone(Model m,at::ScalarType dtype) {
  std::map<const void*,Tensor> copies;
  auto copy=[&](Tensor& x){auto key=x.unsafeGetTensorImpl();auto found=copies.find(key);
    if(found==copies.end())found=copies.emplace(key,x.detach().to(dtype).clone().set_requires_grad(true)).first;x=found->second;};
  for(auto& w:m.nodes){copy(w.decay);copy(w.weight);copy(w.bias);copy(w.read);for(auto& [_,x]:w.extra)copy(x);}
  for(auto* group:{&m.input_scale,&m.agg_scale,&m.edge_scale,&m.output_scale})for(auto& x:*group)copy(x);
  return m;
}
Continuation detached(Continuation q,at::ScalarType dtype) {
  auto copy=[&](Tensor& x){x=x.detach().to(dtype).clone();};
  for(auto& [_,s]:q.states){copy(s.value);for(auto& [__,x]:s.slots)copy(x);}
  for(auto& a:q.pending)copy(a.value);for(auto& [_,h]:q.history)for(auto& [__,x]:h.tensors)copy(x);return q;
}
Result comparison_copy(Result r) {
  auto cast=[](Tensor& x){if(x.defined()&&x.is_floating_point())x=x.detach().to(at::kFloat);};
  auto state=[&](State& s){cast(s.value);for(auto& [_,x]:s.slots)cast(x);};
  auto history=[&](History& h){for(auto& [_,x]:h.tensors)cast(x);};
  for(auto& [_,s]:r.continuation.states)state(s);for(auto& [_,h]:r.continuation.history)history(h);
  for(auto& a:r.continuation.pending)cast(a.value);for(auto& a:r.messages)cast(a.value);for(auto& o:r.outputs)cast(o.value);
  for(auto& e:r.trace) {
    state(e.old);state(e.proposed_state);state(e.comparison_state);state(e.next_state);history(e.history);
    for(auto x:{&e.content,&e.proposal,&e.descriptor,&e.control,&e.comparison,&e.next,&e.full})cast(*x);
    for(auto& a:e.fiber)cast(a.value);for(auto& x:e.emitted)cast(x.value);for(auto& x:e.contributions)cast(x.value);
    for(auto& x:e.sources){cast(x.atom.value);cast(x.scale);}
  }
  return r;
}
void trajectory(at::Device device,int shape,int64_t width,bool prefill,DeviceOptimizerKind kind,at::ScalarType dtype) {
  at::NoGradGuard guard;auto f=test::graph_vjp_fixture(shape,0,width);alias(f.model,shape);
  auto registry=f.model.parameters(false);auto cpu_model=clone(f.model,dtype);auto cpu_registry=cpu_model.parameters(false);
  auto q=detached(f.initial,dtype);std::vector<Tensor> leaves;for(auto& o:cpu_registry.owners())leaves.push_back(o.value);
  OptimizerGroup group;group.lr=.001;group.weight_decay=.0125;group.momentum=.875;group.nesterov=true;group.amsgrad=true;
  for(const auto& o:cpu_registry.owners())group.parameters.push_back(o.canonical);
  std::unique_ptr<NamedOptimizer> cpu_optimizer;
  if(kind==DeviceOptimizerKind::sgd)cpu_optimizer=std::make_unique<SGD>(cpu_registry,std::vector<OptimizerGroup>{group});
  else cpu_optimizer=std::make_unique<AdamW>(cpu_registry,std::vector<OptimizerGroup>{group});
  ContentLimits limits;limits.prefill=prefill;limits.trace=512;limits.full_chunk_rows=3;if(width>3)limits.workspace_bytes=512*1024*1024;
  ContentFlow flow(f.graph,f.model,f.initial,device,limits);std::unique_ptr<DeviceOptimizer> optimizer;
  for(int step=0;step<4;++step) {
    const int64_t stop=(step+1)*11;std::vector<External> input,cpu_input;
    for(auto x:f.input){x.time+=step*11;x.position+=step*(x.batch==0?3:2);input.push_back(x);x.value=x.value.to(dtype);cpu_input.push_back(x);}
    flow.advance_device(input,stop);auto t=flow.reverse_tape();auto opts=t.fiber_values.options();
    const float factor=step==1?0.f:1.f;
    GraphCotangents roots{at::full_like(t.outputs.values,factor*.0625f),at::zeros_like(t.outputs.valid),at::full_like(t.pending.values,factor*.015625f),
      at::zeros_like(t.pending.valid),at::full({2,4,width},factor*.03125f,opts),at::zeros({2,4},opts.dtype(at::kBool))};
    if(step!=2){roots.outputs_connected.copy_(t.outputs.valid);roots.pending_connected.copy_(t.pending.valid);roots.final_connected.fill_(true);}
    auto error=at::zeros({1},opts.dtype(at::kInt));DeviceProgram p(device);p.limit_workspace(64*1024*1024);
    auto graph=append_graph_vjp(p,t,roots,error,3,128*1024*1024);
    auto grad=append_parameter_vjp(p,f.graph,registry,graph,error,16*1024*1024);
    if(!optimizer)optimizer=std::make_unique<DeviceOptimizer>(grad,kind,std::vector<OptimizerGroup>{group},64*1024*1024);
    optimizer->append_step(p,grad,error);append_parameter_publish(p,flow.parameter_banks(),grad,optimizer->values(),error,1024*1024);
    p.finish();portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"resident training step refused");
    std::vector<Tensor> expected(leaves.size());Result result;
    {
      at::AutoGradMode enabled(true);Streaming cpu(f.graph,cpu_model,{});result=cpu.run(q,cpu_input,stop,stop);
      tide_bench::compare(flow.result(),comparison_copy(result),true,at::kFloat);
      if(step!=2){std::vector<Tensor> terms;
        for(const auto& x:result.outputs)terms.push_back(x.value.sum()*(factor*.0625));
        for(const auto& [_,s]:result.continuation.states)terms.push_back(s.value.sum()*(factor*.03125));
        for(const auto& a:result.continuation.pending)terms.push_back(a.value.sum()*(factor*.015625));
        expected=torch::autograd::grad({at::stack(terms).sum()},leaves,{},false,false,true);}
    }
    auto values=grad.values.cpu(),on=grad.connected.cpu();
    for(size_t i=0;i<leaves.size();++i) {
      const auto n=leaves[i].numel(),offset=grad.offsets[i];
      auto value=offset<0?at::zeros_like(leaves[i]).to(at::kFloat):values.narrow(0,offset,n).reshape(leaves[i].sizes());
      test::full_same(value,on[i],expected[i],grad.owners[i].canonical.c_str());
      leaves[i].mutable_grad()=expected[i];
    }
    cpu_optimizer->step();auto updated=optimizer->values().cpu();
    for(size_t i=0;i<leaves.size();++i)if(grad.offsets[i]>=0)
      test::full_same(updated.narrow(0,grad.offsets[i],leaves[i].numel()).reshape(leaves[i].sizes()),at::ones({},at::kBool),leaves[i],grad.owners[i].canonical.c_str());
    // Verify a differentiable bias update also reaches a HARD Read alias, even
    // if these four windows happen not to cross a selection boundary.
    auto banks=flow.parameter_banks();require(at::allclose(banks.read[3].cpu(),cpu_model.nodes[3].read.to(at::kFloat),1e-5,1e-6),"updated shared Read bank stale");
    // This checker explicitly truncates at every optimizer boundary. It does
    // not claim retained-window differentiation or an implicit detach policy.
    q=detached(result.continuation,dtype);
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("training step gate requires explicit NPU FP32");
    const auto device=portable_torch::resolve_device(args);if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("training step gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int shape:{0,3})for(bool prefill:{false,true})for(auto kind:{DeviceOptimizerKind::sgd,DeviceOptimizerKind::adamw})
      for(auto dtype:{at::kFloat,at::kDouble}) {
        try{trajectory(device,shape,3,prefill,kind,dtype);++cases;}
        catch(...){std::cerr<<"training step shape="<<shape<<" prefill="<<prefill<<" optimizer="<<int(kind)<<" reference="<<dtype<<'\n';throw;}
      }
    for(auto kind:{DeviceOptimizerKind::sgd,DeviceOptimizerKind::adamw}){trajectory(device,0,257,true,kind,at::kFloat);++cases;}
    std::cout<<"device-training-step: passed trajectories="<<cases<<" windows="<<cases*4<<" aliases=true continuation=explicit_truncation scope=internal_HARD_subset_not_public_training\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
