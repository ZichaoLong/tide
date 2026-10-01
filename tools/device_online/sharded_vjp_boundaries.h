#pragma once
#include "sharded_full_vjp.h"
#include "retained_fixture.h"
#include "precision_graph_fixture.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online::test {
inline void sharded_vjp_boundaries(at::Device device,int devices,at::ScalarType dtype) {
  at::NoGradGuard guard;auto f=graph_vjp_fixture(0,0);fixture_dtype(f,dtype);
  std::vector<at::Device> owner;for(int i=0;i<devices;++i)owner.emplace_back(device.type(),device.index()+i);
  ContentFlow flow(f.graph,f.model,f.initial,device,{},place_full(f.graph,f.model,owner,"memory"));
  flow.advance_device({},f.initial.cut);auto t=retain_sharded_reverse_tape(flow.sharded_reverse_tape(),64*1024*1024);
  auto roots=retained_roots(t.tape.coordinator,3,5);auto error=at::zeros({1},roots.final.options().dtype(at::kInt));
  auto require=[](bool ok){if(!ok)throw std::runtime_error("sharded reverse boundary check failed");};
  auto reject=[&](auto fn){bool refused=false;try{fn();}catch(const std::invalid_argument&){refused=true;}require(refused);};
  reject([&]{retain_sharded_reverse_tape(t.tape,1);});
  reject([&]{CannProgram p(device);append_graph_vjp(p,t.tape.coordinator,roots,error,3,128*1024*1024);});
  reject([&]{CannProgram p(device);append_sharded_graph_vjp(p,t.tape,roots,error,3,1,64*1024*1024);});
  auto duplicate=t.tape;duplicate.shards[0].nodes[0]=duplicate.shards.back().nodes.back();
  reject([&]{CannProgram p(device);append_sharded_graph_vjp(p,duplicate,roots,error,3,1024*1024*1024,64*1024*1024);});
  auto wrong=roots;wrong.final=roots.final.to(at::kHalf);
  reject([&]{CannProgram p(device);append_sharded_graph_vjp(p,t.tape,wrong,error,3,1024*1024*1024,64*1024*1024);});
  auto execute=[&](const ShardedReverseTape& tape,bool malformed) {
    CannProgram p(device);p.limit_workspace(64*1024*1024);
    auto g=append_sharded_graph_vjp(p,tape,roots,error,3,1024*1024*1024,64*1024*1024);p.finish();run_sharded_graph_vjp(p,{g});
    require(error.cpu().item<int>()==(malformed?2:0));require(!g.coordinator.initial.cpu().any().item<bool>());
    require(!g.coordinator.full_connected.cpu().any().item<bool>()&&!g.coordinator.scale_connected.cpu().any().item<bool>());
    for(const auto& s:g.full->gradients())require(!s.values.parameter_connected.cpu().any().item<bool>());
    if(!malformed)require(at::equal(g.coordinator.initial_connected.cpu(),roots.final_connected.cpu()));
    p.close();g.full->close();
  };
  execute(t.tape,false);
  flow.advance_device(f.input,f.initial.cut+11);auto malformed=retain_sharded_reverse_tape(flow.sharded_reverse_tape(),64*1024*1024);
  malformed.tape.coordinator.state.metadata[0][12].fill_(1);execute(malformed.tape,true);flow.close();
}
} // namespace tide::device_online::test
