#include "block_settle_boundary.h"
#include "block_read_selection_internal.h"
#include "device_launch_tide_block_flow_boundary.h"
namespace tide::device_online {
BlockSettleBoundary append_block_settle_boundary(DeviceProgram& p,const SettleGraph& spec,const BlockSettleFlowTape& t,
    const at::Tensor& external,const BlockSettleBoundary& previous,int64_t budget){
  using namespace block_selection_detail;const auto& g=spec.graph();const auto b=t.positions.size(0),l=t.positions.size(1);
  const auto nodes=int64_t(g.nodes.size()),regions=int64_t(g.regions.size()),ports=int64_t(g.inputs.size()),stride=spec.stride();const auto error=t.dispatcher->error();
  check(previous.node_present,error,at::kBool,{b,nodes});check(previous.region_times,error,at::kLong,{b,regions});
  check(previous.input_counts,error,at::kLong,{b,ports});check(previous.input_times,error,at::kLong,{b,ports});check(external,error,at::kBool,{b,l,ports});
  admit(b*(nodes+8.L*regions+16.L*ports)+8.L*ports,budget);
  BlockSettleBoundary out{at::empty_like(previous.node_present),at::empty_like(previous.region_times),at::empty_like(previous.input_counts),at::empty_like(previous.input_times)};
  p.copy(out.node_present,previous.node_present);p.copy(out.region_times,previous.region_times);p.copy(out.input_counts,previous.input_counts);p.copy(out.input_times,previous.input_times);
  auto append=[&](const at::Tensor& ids,const at::Tensor& present,const at::Tensor& times,int64_t command,int64_t count,int64_t region){
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_flow_boundary)(1,stream,
      ptr(ids),ptr(present),ptr(times),ptr(out.node_present),ptr(out.region_times),ptr(out.input_counts),ptr(out.input_times),ptr(error),
      command,b,l,count,nodes,region,regions,stride),"actual block complete-cut presence/history/input occurrence ledger");},
      {ids,present,times,out.node_present,out.region_times,out.input_counts,out.input_times,error});};
  std::vector<int64_t> ranks;for(auto v:g.inputs)ranks.push_back(spec.rank(v));const auto ids=at::tensor(ranks,at::kLong).to(error.device());
  append(ids,external,t.positions,0,ports,0);
  for(int64_t r=0;r<regions;++r){const auto& stage=t.regions[r];append(stage.nodes,stage.local->present,
      stage.local->messages.selection.node_times,1,stage.nodes.numel(),r);}
  return out;
}
}
