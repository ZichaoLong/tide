#pragma once
#include "state_reverse_test.h"
#include "precision_graph_fixture.h"
#include "portable_torch/runtime.hpp"
namespace tide::device_online::test {
inline void state_reverse_boundaries(at::Device device,int devices,at::ScalarType dtype) {
  at::NoGradGuard guard;auto f=graph_vjp_fixture(3,1);fixture_dtype(f,dtype);
  std::vector<at::Device> placement;for(int i=0;i<devices;++i)placement.emplace_back(device.type(),device.index()+i);
  auto flow=reverse_candidate(f,device,{},place_full(f.graph,f.model,placement,"memory"),true);
  flow->advance_device({},f.initial.cut);auto saved=retain_sharded_reverse_tape(flow->sharded_reverse_tape(),64*1024*1024);
  auto& tape=saved.tape;auto roots=retained_roots(tape.coordinator,3,5);auto error=at::zeros({1},roots.final.options().dtype(at::kInt));
  auto require=[](bool x){if(!x)throw std::runtime_error("compact state reverse boundary check failed");};
  auto reject=[&](auto fn){bool refused=false;try{fn();}catch(const std::invalid_argument&){refused=true;}require(refused);};
  reject([&]{retain_sharded_reverse_tape(tape,1);});
  auto invalid=tape;invalid.states[0].global_nodes[0]=invalid.states.back().global_nodes.back();
  reject([&]{CannProgram p(device);append_sharded_graph_vjp(p,invalid,roots,error,3,1024*1024*1024,64*1024*1024);});
  auto run=[&](const ShardedReverseTape& t,int expected) {
    CannProgram p(device);p.limit_workspace(64*1024*1024);
    auto g=append_sharded_graph_vjp(p,t,roots,error,3,1024*1024*1024,64*1024*1024);p.finish();run_sharded_graph_vjp(p,{g});
    require(error.cpu().item<int>()==expected);
    if(!expected){require(at::equal(g.coordinator.initial_connected.cpu(),roots.final_connected.cpu()));
      require(!g.coordinator.initial.cpu().any().item<bool>());}
    p.close();close_sharded_graph_vjp({g});
  };
  run(tape,0);
  flow->advance_device(f.input,f.initial.cut+11);auto actual=retain_sharded_reverse_tape(flow->sharded_reverse_tape(),64*1024*1024);
  const auto& t=actual.tape.coordinator;const auto count=t.state.count.cpu().item<int64_t>();require(count>1);
  // Check complete physical records, int64 times/parallel-edge IDs, and actual
  // scale lookup by comparing only after device execution. Candidate input is
  // exclusively its own retained journal.
  for(bool overflow:{false,true}) {
    error.zero_();CannProgram p(device);auto links=append_reverse_links(p,t,error,16*1024*1024);
    auto mapping=at::arange(int64_t(f.graph.nodes.size()),t.state.metadata.options());
    auto packet=append_state_reverse_pack(p,t,links,mapping,f.graph.nodes.size(),overflow?1:t.state.metadata.size(0),t.fiber_meta.size(0),error,32*1024*1024);
    p.finish();portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==(overflow?1:0));
    if(overflow){require(!packet.event_count.cpu().item<int64_t>()&&!packet.fiber_count.cpu().item<int64_t>());
      require(!packet.event_values.cpu().any().item<bool>()&&!packet.fiber_values.cpu().any().item<bool>());}
    else {
      require(packet.event_count.cpu().item<int64_t>()==count);
      require(at::equal(packet.event_meta.cpu().narrow(0,0,count),t.state.metadata.cpu().narrow(0,0,count)));
      const auto atoms=t.fiber_count.cpu().item<int64_t>();require(packet.fiber_count.cpu().item<int64_t>()==atoms);
      require(at::equal(packet.fiber_meta.cpu().narrow(0,0,atoms),t.fiber_meta.cpu().narrow(0,0,atoms)));
      require(at::equal(packet.event_rows.cpu().narrow(0,0,count),at::arange(count,at::kLong)));
      require(at::equal(packet.fiber_rows.cpu().narrow(0,0,atoms),at::arange(atoms,at::kLong)));
    }
    p.close();
  }
  error.zero_();actual.tape.coordinator.state.metadata[0][12].fill_(1);run(actual.tape,2);flow->close();
}
} // namespace tide::device_online::test
