#include "state_reverse_merge.h"
#include "device_backend.h"
#include "device_launch_tide_state_reverse_merge.h"
namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
void append_state_reverse_error(DeviceProgram& p,const at::Tensor& source,const at::Tensor& target) {
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_state_reverse_merge)(1,stream,
    ptr(source),ptr(source),ptr(source),ptr(source),ptr(source),ptr(source),ptr(source),ptr(source),ptr(source),ptr(source),ptr(source),
    ptr(source),ptr(target),0,0,0,0,0,0,4),"merge sticky state reverse status");},{source,target});
}
void append_state_reverse_merge(DeviceProgram& p,const StateReverseStage& stage,const StateOwnerVjp& result,
    const at::Tensor& ids,const StateVjp& out,const at::Tensor& local_error,const at::Tensor& error) {
  const auto x=result.state;const auto capacity=x.content.size(0),global=out.content.size(0),nodes=ids.numel(),total=out.initial.size(1),samples=out.initial.size(0),width=x.content.size(1);
  for(int64_t phase:{0,1})p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_state_reverse_merge)(phase?32:1,stream,
    ptr(stage.tape.count),ptr(stage.destinations),ptr(ids),ptr(x.content),ptr(x.content_connected),ptr(x.initial),ptr(x.initial_connected),
    ptr(out.content),ptr(out.content_connected),ptr(out.initial),ptr(out.initial_connected),ptr(local_error),ptr(error),capacity,global,nodes,total,samples,width,phase),
    "merge compact state adjoints by actual device row maps");},{stage.tape.count,stage.destinations,ids,x.content,x.content_connected,x.initial,x.initial_connected,
    out.content,out.content_connected,out.initial,out.initial_connected,local_error,error});
}
void append_state_reverse_sources(DeviceProgram& p,const StateReversePacket& packet,const StateOwnerVjp& result,
    const at::Tensor& messages,const at::Tensor& on,const at::Tensor& partials,const at::Tensor& error) {
  const auto capacity=result.messages.size(0),global=partials.size(0),width=messages.size(1);
  for(int64_t phase:{2,3})p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_state_reverse_merge)(phase==2?1:32,stream,
    ptr(packet.fiber_count),ptr(packet.fiber_rows),ptr(packet.fiber_rows),ptr(result.messages),ptr(result.connected),ptr(result.scale_partials),ptr(result.connected),
    ptr(messages),ptr(on),ptr(partials),ptr(on),ptr(error),ptr(error),capacity,global,0,0,0,width,phase),"merge owner fiber and physical-scale adjoints");},
    {packet.fiber_count,packet.fiber_rows,result.messages,result.connected,result.scale_partials,messages,on,partials,error});
}
} // namespace tide::device_online
