#include "control_forward.h"
#include "device_backend.h"
#include "device_launch_tide_emit_mix.h"
namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
ActionBatch append_control_forward(DeviceProgram& p,const ContentProfile& profile,const ActionBatch& actions,
    const at::Tensor& content,const at::Tensor& controls,const at::Tensor& error) {
  auto output=at::empty_like(actions.values);p.copy(output,actions.values);
  const auto identities=profile.read_modes;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_emit_mix)(32,stream,
    ptr(actions.coordinates),ptr(actions.valid),ptr(identities),ptr(content),ptr(actions.values),ptr(controls),ptr(output),ptr(error),
    content.size(0),content.size(1),content.scalar_type()==at::kHalf),"packed selected SOFTP emission");},
    {actions.coordinates,actions.valid,identities,content,actions.values,controls,output,error});
  return {actions.coordinates,output,actions.valid};
}
}
