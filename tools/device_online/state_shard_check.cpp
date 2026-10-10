#include "device_backend.h"
#include "sharded_state.h"
#include "content_fixture.h"
#include "content_flow_internal.h"
#include "tide/stream.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;using namespace tide::device_online;
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
void same_states(const Continuation& actual,const Continuation& expected,bool exact) {
  require(actual.states.size()==expected.states.size(),"state owner created or lost an owner");
  auto same=[&](const Tensor& a,const Tensor& b) {
    require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"state/cache shape or dtype differs");
    if(exact)require(at::equal(a.contiguous().reshape({-1}).view(at::kByte),b.contiguous().reshape({-1}).view(at::kByte)),"state/KV partly committed on refusal");
    else require(at::allclose(a,b,a.scalar_type()==at::kHalf?2e-2:1e-5,a.scalar_type()==at::kHalf?2e-3:1e-6),"compact state/KV differs from independent CPU");
  };
  for(const auto& [owner,a]:actual.states) {
    const auto& b=expected.states.at(owner);require(a.last_time==b.last_time&&a.observations==b.observations,"compact clocks differ");same(a.value,b.value);
    require(a.slots.size()==b.slots.size(),"state cache slot names differ");for(const auto& [name,x]:a.slots)same(x,b.slots.at(name));
  }
}
test::Fixture fixture(at::ScalarType dtype,bool selective) {
  auto f=test::fixture(2,0);Graph fresh;fresh.nodes=f.graph.nodes;fresh.regions=f.graph.regions;f.graph=std::move(fresh);
  auto& g=f.graph;g.inputs={0,1,2,3,3};g.outputs={0,1,2,3};
  for(auto& r:g.regions){r.read_mode="proposal";r.observe_all=!selective;r.selector="count-v1";r.count_priority=false;}
  for(auto& n:g.nodes){n.identity=false;n.clear=false;n.full="identity";n.memory="ema";}
  g.nodes[1].memory="lh-add-repeat-v1";f.model.nodes[1].extra["add_retention"]=at::full({},.5f);
  g.nodes[2].memory="attention";g.nodes[2].query_heads=3;g.nodes[2].kv_heads=1;g.nodes[2].window=3;
  auto eye=at::eye(3,at::kFloat);auto& e=f.model.nodes[2].extra;
  e["attn_q"]=eye*.25f;e["attn_k"]=eye.narrow(1,0,1)*.5f;e["attn_v"]=eye.narrow(1,0,1).clone();e["attn_out"]=eye*.5f;
  g.nodes[3].memory="lh-fiber-attention-sum-repeat-v1";
  auto& a=f.model.nodes[3].extra;a["fiber_qkv"]=at::cat({eye*.25f,eye*.5f,eye},1);a["fiber_qkv_bias"]=at::zeros({9});
  a["fiber_out"]=eye*.5f;a["fiber_out_bias"]=at::zeros({3});a["fiber_decay"]=at::full({},.03125f);
  f.model.input_scale.assign(5,at::full({},.5f));f.model.output_scale.assign(4,at::full({},1.f));
  g.compile();f.initial.identity=g.identity;f.initial.history.clear();f.initial.batch_size=1;
  test::model_dtype(f.model,dtype);for(auto& [_,s]:f.initial.states)s.value=s.value.to(dtype);
  return f;
}
void transaction(at::Device d,at::ScalarType dtype,bool selective) {
  auto f=fixture(dtype,selective);auto peer=at::Device(d.type(),d.index()+1);
  ContentLimits l;l.queue=16;l.kv_rows=16;l.kv_trace_rows=256;l.trace=128;l.attention_chunk_rows=2;l.attention_key_rows=3;
  l.vectorized_read=l.vectorized_state=true;l.workspace_bytes=64*1024*1024;
  ContentProfile profile(f.graph,f.model,d,true);
  ShardedState shards(profile,{{d,peer},{0,1,1,0}},f.initial,d,l,48*1024*1024);profile.upload_routing(d);
  auto opts=at::TensorOptions().device(d).dtype(dtype),longs=opts.dtype(at::kLong);
  AtomBatch input{at::zeros({16,6},longs),at::zeros({16,3},opts),at::zeros({16},opts.dtype(at::kBool))};
  auto error=at::zeros({1},opts.dtype(at::kInt)),injected=at::zeros_like(error),observed=at::zeros_like(error);
  auto stop=at::zeros({1},longs),stage=at::zeros_like(stop),count=at::zeros_like(stop);
  DeviceReady planner(profile.owners,f.graph.regions.size(),profile.wires,1,d,true,profile.causal_regions);
  FrameSelector selector(profile.owners,profile.policies,1,d);auto history=selector.initial();
  DeviceProgram p(d);p.limit_workspace(8*1024*1024);
  auto ready=planner.append_stage(p,input,stop,error);auto content=append_content(p,profile,ready,error,true);
  shards.append_read(p,ready,content,stage,error,8*1024*1024);
  auto selection=selector.append_stage(p,ready,content.scores,history,error);
  auto update=shards.append_update(p,ready,content,selection,stage,count,error);
  p.copy(observed,error);p.copy(error,injected); // Explicit downstream refusal after all proposals.
  selector.append_commit(p,history,selection,error);shards.append_commit(p,error);shards.append_stop(p);p.finish();
  Streaming reference(f.graph,f.model,Options{});auto q=f.initial;int64_t windows=0,refusals=0;
  auto run=[&](const std::vector<External>& xs,int64_t until,int failure) {
    std::vector<Atom> atoms;for(const auto& x:xs)atoms.push_back({x.batch,f.graph.inputs[x.port],x.time,0,x.port,x.position,x.value});
    upload_atoms(atoms,input);error.zero_();observed.zero_();injected.fill_(failure);count.zero_();stop.fill_(until);shards.reset_window();
    shards.synchronize_inputs();c10::impl::VirtualGuardImpl(d.type()).synchronizeDevice(d.index());
    p.submit();shards.submit();std::exception_ptr problem;
    try{p.wait();}catch(...){problem=std::current_exception();}try{shards.wait();}catch(...){if(!problem)problem=std::current_exception();}
    if(problem)std::rethrow_exception(problem);
    require(!observed.cpu().item<int>(),"state proposal failed before injected downstream refusal");
    require(error.cpu().item<int>()==failure,"state consensus lost downstream refusal");++windows;
  };
  run({},0,0);Continuation initial;initial.batch_size=1;shards.export_states(initial);same_states(initial,f.initial,true);
  for(int t=0;t<4;++t) {
    std::vector<External> xs;for(int port=0;port<5;++port)
      xs.push_back({0,port,t,t,at::tensor({float(port+1),float(t+1),.125f}).to(dtype)*.125f});
    Continuation before;before.batch_size=1;shards.export_states(before);
    for(int failure:{1,7}) {run(xs,t+1,failure);Continuation after;after.batch_size=1;shards.export_states(after);same_states(after,before,true);++refusals;}
    run(xs,t+1,0);q=reference.run(q,xs,t+1,t+1).continuation;
    Continuation actual;actual.batch_size=1;shards.export_states(actual);same_states(actual,q,false);
  }
  run({},5,0);Continuation actual;actual.batch_size=1;shards.export_states(actual);same_states(actual,q,false);
  p.close();shards.close();
  std::cout<<"state transaction selective="<<selective<<" windows="<<windows<<" refusals="<<refusals<<" passed\n";
}
void packing(at::Device d,at::ScalarType dtype) {
  auto opts=at::TensorOptions().device(d).dtype(dtype),longs=opts.dtype(at::kLong);const int64_t time=(int64_t(1)<<55)+5;
  ReadyBatch ready;ready.fibers=at::tensor(std::vector<int64_t>{0,0,time,0,0,2,time,0,1,2,time+1,1},at::kLong).reshape({3,4});
  auto padded=at::zeros({8,4},at::kLong);padded.narrow(0,0,3).copy_(ready.fibers);ready.fibers=padded.to(d);
  ready.fiber_offsets=at::tensor({0,1,3,4,4,4,4,4,4},at::kLong).to(d);ready.counts=at::tensor({4,3,2},at::kLong).to(d);
  auto coordinates=at::zeros({8,6},at::kLong);coordinates.narrow(0,0,4).copy_(at::tensor(std::vector<int64_t>{0,0,time,0,0,0,0,2,time,1,17,time-1,0,2,time,1,18,time-1,1,2,time+1,0,2,0},at::kLong).reshape({4,6}));
  auto values=at::arange(24,at::kFloat).reshape({8,3}).to(dtype);values.narrow(0,4,4).fill_(std::numeric_limits<float>::quiet_NaN());
  ready.atoms={coordinates.to(d),values.to(d),at::zeros({8},opts.dtype(at::kBool))};
  ContentBatch content{values.to(d),at::zeros({8},opts.dtype(at::kFloat)),values.to(d)};
  auto map=at::tensor({-1,-1,0},at::kLong).to(d),error=at::zeros({1},opts.dtype(at::kInt));
  DeviceProgram p(d);p.limit_workspace(1024*1024);auto packed=append_state_shard_pack(p,ready,content,map,1,4,error);p.finish();
  p.run();require(!error.cpu().item<int>(),"valid whole-fiber packing refused");
  require(at::equal(packed.ready.counts.cpu(),at::tensor({3,2,0},at::kLong)),"packing lost complete fibers");
  require(at::equal(packed.atom_rows.cpu(),at::tensor({1,2,3,8},at::kLong)),"physical parallel-edge order changed");
  auto actual=packed.ready.atoms.coordinates.cpu();require(actual[0][4].item<int64_t>()==17&&actual[1][4].item<int64_t>()==18,"physical edge identity changed");
  require(actual[0][2].item<int64_t>()==time,"packing rounded int64 time");
  require(at::equal(packed.ready.atoms.values.cpu()[3],at::zeros({3},dtype)),"inactive gather read NaN padding");p.close();
  error.zero_();DeviceProgram refused(d);refused.limit_workspace(1024*1024);
  auto small=append_state_shard_pack(refused,ready,content,map,1,1,error);refused.finish();refused.run();
  require(error.cpu().item<int>()==1&&!small.ready.branch.cpu().item<int>()&&!small.ready.counts.cpu().any().item<bool>(),"capacity silently split a fiber");
  require(!small.ready.atoms.valid.cpu().any().item<bool>(),"failed pack exposed partial messages");refused.close();
  std::cout<<"state packing parallel-edges/int64/padding/whole-fiber-refusal passed\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    args.allow_npu_float16=true;auto d=portable_torch::resolve_device(args);
    if(d.type()!=tide::device_online::resident_device_type||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("state shard checks require explicit NPU FP32/FP16");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    packing(d,args.dtype);for(bool selective:{false,true})transaction(d,args.dtype,selective);
    std::cout<<"state-shard: passed; two independent owners; no partial state/KV commit\n";runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
