#include "device_backend.h"
#include "state_reverse_pack.h"
#include "device_launch_tide_state_reverse_pack.h"
#include <stdexcept>
#include <ATen/core/grad_mode.h>

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
StateReversePacket append_state_reverse_pack(DeviceProgram& p,const ReverseTape& tape,const ReverseLinks& links,
    const at::Tensor& mapping,int64_t nodes,int64_t events,int64_t fibers,const at::Tensor& error,int64_t budget,
    const ReverseGatherInput& event_input,const ReverseGatherInput& fiber_input,const ReverseGatherInput& scale_input) {
  if(at::GradMode::is_enabled()||!tape.graph||!tape.state.metadata.defined()||tape.state.metadata.dim()!=2
      ||!tape.fiber_meta.defined()||tape.fiber_meta.dim()!=2||!mapping.defined()||mapping.dim()!=1)
    throw std::invalid_argument("compact reverse pack requires no-grad actual journals and node map");
  const auto rows=tape.state.metadata.size(0),atoms=tape.fiber_meta.size(0),width=tape.full.width;
  const auto global_nodes=mapping.numel();const auto longs=tape.state.metadata.options(),floats=tape.state.values.options();
  const long double own=16.L*(events+rows)*(5.L*width+16)+16.L*(fibers+atoms)*(width+16)+4096;
  if(events<1||fibers<1||events>rows||fibers>atoms||nodes<1||budget<1||own>budget
      ||mapping.device()!=error.device()||mapping.scalar_type()!=at::kLong||!mapping.is_contiguous())
    throw std::invalid_argument("invalid compact reverse packet capacity, mapping or budget");
  const auto device=tape.state.metadata.device();
  auto tensor=[&](const at::Tensor& x,at::ScalarType type,at::IntArrayRef shape) {
    if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
      throw std::invalid_argument("invalid compact reverse pack buffer");
  };
  if(device.type()!=tide::device_online::resident_device_type||global_nodes!=int64_t(tape.graph->nodes.size())||width<1||nodes>global_nodes||links.scales.numel()<1)
    throw std::invalid_argument("invalid compact reverse pack geometry");
  tensor(mapping,at::kLong,{global_nodes});tensor(error,at::kInt,{1});tensor(tape.state.metadata,at::kLong,{rows,13});
  tensor(tape.state.count,at::kLong,{1});tensor(tape.state.values,at::kFloat,{rows,5*width+2});
  tensor(tape.fiber_meta,at::kLong,{atoms,6});tensor(tape.fiber_values,at::kFloat,{atoms,width});tensor(tape.fiber_count,at::kLong,{1});
  tensor(links.messages,at::kLong,{links.fibers+links.pending+links.outputs,4});
  tensor(links.scales,at::kFloat,{links.scales.numel()});
  if(links.fibers!=atoms)throw std::invalid_argument("reverse links disagree with physical atom capacity");
  StateReversePacket out;
  out.event_meta=at::empty({events,13},longs);out.event_values=at::empty({events,5*width+2},floats);out.event_count=at::zeros({1},longs);
  out.fiber_meta=at::empty({fibers,6},longs);out.fiber_values=at::empty({fibers,width},floats);out.fiber_count=at::zeros({1},longs);
  out.event_rows=at::empty({events},longs);out.fiber_rows=at::empty({fibers},longs);
  out.links.messages=at::empty({fibers,4},longs);out.links.valid=at::empty({fibers},floats.dtype(at::kBool));
  out.links.consumer_head=at::empty({events},longs);out.links.consumer_next=at::empty({fibers},longs);
  out.links.scales=at::empty({fibers},floats);out.links.fibers=fibers;out.links.pending=out.links.outputs=0;out.links.parameters=fibers;
  auto inverse=at::empty({rows},longs),tail=at::empty({events},longs),scale_rows=at::empty({fibers},longs);
  const auto scales=links.scales.numel();
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_state_reverse_pack)(1,stream,
    ptr(tape.state.metadata),ptr(tape.state.count),ptr(tape.fiber_meta),ptr(tape.fiber_count),ptr(links.messages),ptr(mapping),
    ptr(out.event_meta),ptr(out.event_count),ptr(out.fiber_meta),ptr(out.fiber_count),ptr(out.event_rows),ptr(out.fiber_rows),ptr(inverse),
    ptr(out.links.messages),ptr(out.links.valid),ptr(out.links.consumer_head),ptr(out.links.consumer_next),ptr(tail),ptr(scale_rows),ptr(error),
    rows,atoms,events,fibers,global_nodes,nodes,scales),"pack actual owner reverse journals and consumer links");},
    {tape.state.metadata,tape.state.count,tape.fiber_meta,tape.fiber_count,links.messages,mapping,out.event_meta,out.event_count,
     out.fiber_meta,out.fiber_count,out.event_rows,out.fiber_rows,inverse,out.links.messages,out.links.valid,out.links.consumer_head,
     out.links.consumer_next,tail,scale_rows,error});
  event_input.select(p,out.event_rows,out.event_values);fiber_input.select(p,out.fiber_rows,out.fiber_values);
  scale_input.select(p,scale_rows,out.links.scales);return out;
}
} // namespace tide::device_online
