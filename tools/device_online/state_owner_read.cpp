#include "state_owner_read.h"
#include "device_backend.h"
#include "device_launch_tide_state_read_plan.h"
#include "device_launch_tide_control_payload.h"
#include "device_launch_tide_control_read_reduce.h"
#include <stdexcept>
namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
ControlVjp append_state_owner_read(DeviceProgram& p,const StateOwnerTape& owner,const StateReverseStage& stage,
    const at::Tensor& error,int64_t budget) {
  const auto& t=stage.tape;const auto capacity=t.metadata.size(0),nodes=owner.layout.nodes,width=owner.layout.width;
  if(!stage.score_gradient.defined()||!stage.read_connected.defined()||budget<1
      ||16.L*capacity*(width+4)+16.L*nodes*(width+4)+1024>budget)
    throw std::invalid_argument("compact Read reverse packet or budget unavailable");
  auto longs=t.metadata.options(),floats=t.values.options(),bits=floats.dtype(at::kBool);
  auto config=at::stack({at::zeros_like(owner.read_modes),owner.read_modes,owner.read_kinds},1);
  auto head=at::empty({nodes},longs),next=at::empty({capacity},longs),partials=at::empty({capacity,width},floats);
  auto zero_range=at::zeros({2},longs);ControlVjp out;
  out.read=at::empty({nodes,width},floats);out.read_connected=at::empty({nodes},bits);
  p.zero(out.read);p.zero(out.read_connected);p.zero(partials);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_state_read_plan)(1,stream,
    ptr(t.metadata),ptr(t.count),ptr(owner.read_modes),ptr(owner.read_kinds),ptr(stage.read_connected),ptr(head),ptr(next),
    ptr(out.read_connected),ptr(error),capacity,nodes),"link compact Read reverse rows");},
    {t.metadata,t.count,owner.read_modes,owner.read_kinds,stage.read_connected,head,next,out.read_connected,error});
  // Mode 1 uses only local Read/actual values/scores and adds to the direct
  // Emit/Full cotangent. Unused mode-0 operands are valid owned sentinels.
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_control_payload)(32,stream,
    ptr(t.metadata),ptr(t.values),ptr(partials),ptr(owner.read),ptr(t.count),ptr(zero_range),ptr(config),ptr(partials),
    ptr(stage.read_connected),ptr(stage.read_connected),ptr(stage.score_gradient),ptr(partials),ptr(stage.cot.events),ptr(partials),ptr(partials),
    ptr(error),capacity,width,1,1,int64_t(owner.read.scalar_type()==at::kHalf)),"packed compact Read adjoints");},
    {t.metadata,t.values,partials,owner.read,t.count,zero_range,config,stage.read_connected,stage.score_gradient,stage.cot.events,error});
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_control_read_reduce)(32,stream,
    ptr(head),ptr(next),ptr(stage.read_connected),ptr(partials),ptr(out.read),ptr(error),nodes,width),"reduce compact Read parameter partials");},
    {head,next,stage.read_connected,partials,out.read,error});
  return out;
}
} // namespace tide::device_online
