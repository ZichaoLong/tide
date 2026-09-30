#include "retained_tape.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_window_bridge_meta.h"
#include "aclrtlaunch_tide_window_bridge_values.h"
#include <ATen/core/grad_mode.h>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void buffer(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid retained-window bridge buffer");
}
}
GraphCotangents append_window_bridge(CannProgram& p,const ReverseTape& a,const GraphCotangents& local,
    const ReverseTape& b,const GraphVjp& grad,const at::Tensor& error,int64_t budget) {
  if(at::GradMode::is_enabled()||!a.graph||!b.graph||a.graph->identity!=b.graph->identity||a.stop!=b.cut
      ||a.full.width!=b.full.width||a.state.samples!=b.state.samples||!a.pending.valid.defined()||!b.fiber_values.defined()||budget<1)
    throw std::invalid_argument("retained-window bridge requires consecutive matching no-grad tapes");
  const auto device=b.fiber_values.device();const auto rows=a.pending.valid.numel(),fibers=b.fiber_values.size(0),pending=b.pending.valid.numel(),outputs=b.outputs.valid.numel();
  const int64_t width=b.full.width,nodes=b.graph->nodes.size(),samples=b.state.samples,total=fibers+pending+outputs;
  if(device.type()!=c10::DeviceType::PrivateUse1||rows<1||fibers<1||pending<1||width<1||nodes<1||samples<1)
    throw std::invalid_argument("retained-window bridge requires bounded NPU capacities");
  int64_t buckets=1;while(buckets<2.L*(fibers+static_cast<long double>(pending))) {
    if(buckets>std::numeric_limits<int64_t>::max()/2)throw std::invalid_argument("window bridge hash capacity overflow");buckets*=2;
  }
  const long double bytes=8.L*buckets+8.L*rows+(4.L*width+1)*(rows+static_cast<long double>(samples)*nodes);
  if(bytes>budget)throw std::invalid_argument("retained-window bridge tensor budget exceeded");
  buffer(error,device,at::kInt,{1});buffer(a.pending.coordinates,device,at::kLong,{rows,6});buffer(a.pending.valid,device,at::kBool,{rows});
  buffer(b.fiber_meta,device,at::kLong,{fibers,6});buffer(b.pending.coordinates,device,at::kLong,{pending,6});
  buffer(grad.links.messages,device,at::kLong,{total,4});buffer(grad.links.valid,device,at::kBool,{total});
  buffer(grad.messages,device,at::kFloat,{total,width});buffer(grad.message_connected,device,at::kBool,{total});
  buffer(grad.initial,device,at::kFloat,{samples,nodes,width});buffer(grad.initial_connected,device,at::kBool,{samples,nodes});
  buffer(local.pending,device,at::kFloat,{rows,width});buffer(local.pending_connected,device,at::kBool,{rows});
  buffer(local.final,device,at::kFloat,{samples,nodes,width});buffer(local.final_connected,device,at::kBool,{samples,nodes});
  GraphCotangents out{local.outputs,local.outputs_connected,at::empty_like(local.pending),at::empty_like(local.pending_connected),at::empty_like(local.final),at::empty_like(local.final_connected)};
  auto hash=at::empty({buckets},a.pending.coordinates.options()),map=at::empty({rows},a.pending.coordinates.options());
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_window_bridge_meta)(1,stream,
    ptr(a.pending.coordinates),ptr(a.pending.valid),ptr(b.fiber_meta),ptr(b.pending.coordinates),ptr(grad.links.messages),ptr(grad.links.valid),
    ptr(grad.message_connected),ptr(grad.initial_connected),ptr(local.pending_connected),ptr(local.final_connected),
    ptr(hash),ptr(map),ptr(out.pending_connected),ptr(out.final_connected),ptr(error),rows,fibers,pending,samples*nodes,buckets),
    "match actual retained-window boundary messages");},{a.pending.coordinates,a.pending.valid,b.fiber_meta,b.pending.coordinates,grad.links.messages,grad.links.valid,
      grad.message_connected,grad.initial_connected,local.pending_connected,local.final_connected,hash,map,out.pending_connected,out.final_connected,error});
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_window_bridge_values)(32,stream,
    ptr(map),ptr(a.pending.valid),ptr(grad.messages),ptr(grad.message_connected),ptr(grad.initial),ptr(grad.initial_connected),
    ptr(local.pending),ptr(local.pending_connected),ptr(local.final),ptr(local.final_connected),ptr(out.pending),ptr(out.final),ptr(error),rows,samples*nodes,width),
    "pack retained-window state and pending adjoints");},{map,a.pending.valid,grad.messages,grad.message_connected,grad.initial,grad.initial_connected,
      local.pending,local.pending_connected,local.final,local.final_connected,out.pending,out.final,error});
  return out;
}
} // namespace tide::device_online
