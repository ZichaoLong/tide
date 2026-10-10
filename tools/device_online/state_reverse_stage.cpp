#include "state_reverse_stage.h"
#include "device_backend.h"
#include "device_launch_tide_state_reverse_stage.h"
#include <stdexcept>

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
StateReverseStage append_state_reverse_stage(DeviceProgram& p,const StateReversePacket& packet,
    const StateTape& parameters,const at::Tensor& global_range,const StateCotangents& roots,
    const at::Tensor& ids,const at::Tensor& scores,const at::Tensor& read_on,const at::Tensor& error,int64_t budget,
    const ReverseGatherInput& events,const ReverseGatherInput& connections) {
  const int64_t capacity=packet.event_meta.size(0),global=roots.events.size(0),nodes=ids.numel(),width=roots.events.size(2),samples=roots.final.size(0);
  const bool controlled=scores.defined();
  if(nodes<1||capacity<1||global<capacity||controlled!=read_on.defined()||budget<1
      ||64.L*capacity*(width+8)+16.L*samples*nodes*(width+2)+4096>budget)
    throw std::invalid_argument("compact state reverse stage budget exceeded");
  auto longs=packet.event_meta.options(),floats=roots.events.options(),bits=roots.connected.options();
  StateReverseStage out;out.tape=parameters;
  out.tape.metadata=at::empty_like(packet.event_meta);out.tape.values=at::empty_like(packet.event_values);out.tape.count=at::empty_like(packet.event_count);
  out.range=at::empty({2},longs);out.destinations=at::empty({capacity},longs);
  out.cot={at::empty({capacity,5,width},floats),at::empty({capacity,5},bits),
    at::empty({samples,nodes,width},floats),at::empty({samples,nodes},bits)};
  auto rows=at::empty({capacity},longs),source=at::empty_like(rows);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_state_reverse_stage)(1,stream,
    ptr(packet.event_meta),ptr(packet.event_count),ptr(packet.event_rows),ptr(global_range),ptr(out.tape.metadata),ptr(out.tape.count),
    ptr(out.range),ptr(rows),ptr(source),ptr(out.destinations),ptr(error),capacity,global),"pack current compact state reverse stage");},
    {packet.event_meta,packet.event_count,packet.event_rows,global_range,out.tape.metadata,out.tape.count,out.range,rows,source,out.destinations,error});
  auto gather=[&](const at::Tensor& input,const at::Tensor& indices,const at::Tensor& output) {
    auto shape=input.sizes().vec();++shape[0];auto padded=at::zeros(shape,input.options());
    p.copy(padded.narrow(0,0,input.size(0)),input);p.index_select(padded,0,indices,output);
  };
  gather(packet.event_values,rows,out.tape.values);events.select(p,source,out.cot.events);connections.select(p,source,out.cot.connected);
  p.index_select(roots.final,1,ids,out.cot.final);p.index_select(roots.final_connected,1,ids,out.cot.final_connected);
  if(controlled){out.score_gradient=at::empty({capacity},floats);out.read_connected=at::empty({capacity},bits);
    gather(scores,source,out.score_gradient);gather(read_on,source,out.read_connected);}
  return out;
}
} // namespace tide::device_online
