#include "state_vjp.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_state_vjp_plan.h"
#include "aclrtlaunch_tide_state_vjp.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void tensor(const at::Tensor& x,at::Device device,at::ScalarType dtype,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=dtype||x.sizes()!=shape
      ||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid state VJP buffer");
}
}
StateVjp append_state_vjp(CannProgram& p,const StateTape& tape,const StateCotangents& cot,
                         const at::Tensor& error,int64_t budget,int64_t repeat_chunk_ticks) {
  if(at::GradMode::is_enabled()||!tape.values.defined()||tape.values.dim()!=2
      ||!tape.config.defined()||tape.config.dim()!=2||!tape.decay.defined()||tape.decay.dim()!=2)
    throw std::invalid_argument("state VJP requires explicit no-grad, packed device buffers");
  const auto device=tape.values.device();
  const auto capacity=tape.values.size(0),nodes=tape.config.size(0),width=tape.decay.size(1),samples=tape.samples;
  // Includes persistent partials, reverse indices and sigmoid coefficients.
  const long double estimate=16.L*capacity*(width+8.L)+32.L*samples*nodes*(width+4.L)+8.L*nodes*width;
  if(device.type()!=c10::DeviceType::PrivateUse1||capacity<1||nodes<1||width<1||samples<1
      ||budget<1||estimate>=budget||repeat_chunk_ticks<1||tape.max_repeat_ticks<1)
    throw std::invalid_argument("state VJP workspace budget exceeded");
  const int64_t repeat_rows=tape.has_repeat?std::min(repeat_chunk_ticks,tape.max_repeat_ticks):1;
  const int64_t scratch_width=((std::min<int64_t>(256,width)+7)/8)*8;
  const long double block_bytes=tape.has_repeat?4.L*repeat_rows*scratch_width:1.L;
  const uint32_t blocks=std::min<long double>({32.L,static_cast<long double>(samples*nodes*((width+255)/256)),
                                              (budget-estimate)/block_bytes});
  if(blocks<1)throw std::invalid_argument("one state VJP repeat tile exceeds workspace budget");
  tensor(tape.metadata,device,at::kLong,{capacity,13});tensor(tape.values,device,at::kFloat,{capacity,5*width+2});
  tensor(tape.count,device,at::kLong,{1});tensor(tape.config,device,at::kLong,{nodes,3});
  tensor(tape.decay,device,at::kFloat,{nodes,width});
  tensor(tape.retention,device,at::kFloat,{nodes});tensor(tape.clock_policy,device,at::kLong,{nodes,3});
  tensor(cot.events,device,at::kFloat,{capacity,5,width});tensor(cot.connected,device,at::kBool,{capacity,5});
  tensor(cot.final,device,at::kFloat,{samples,nodes,width});tensor(cot.final_connected,device,at::kBool,{samples,nodes});
  tensor(error,device,at::kInt,{1});
  auto floats=tape.values.options(),longs=tape.metadata.options(),booleans=cot.connected.options();
  StateVjp out{at::zeros({capacity,width},floats),at::zeros({capacity},booleans),
    at::zeros({samples,nodes,width},floats),at::zeros({samples,nodes},booleans),
    at::zeros({samples,nodes,width},floats),at::zeros({samples,nodes},booleans),
    at::zeros({samples,nodes,width},floats),at::zeros({samples,nodes},booleans)};
  out.proposal=at::zeros({tape.has_attention?capacity:1,tape.has_attention?width:1},floats);
  out.proposal_connected=at::zeros({tape.has_attention?capacity:1},booleans);
  auto previous=at::empty({capacity},longs),tail=at::empty({samples,nodes},longs),ticks=at::empty({capacity},longs);
  auto replay=at::empty({tape.has_repeat?blocks:1,repeat_rows,tape.has_repeat?scratch_width:1},floats);
  auto coefficients=at::empty_like(tape.decay);
  // A program can be invoked again after the forward owner records a shorter
  // window. Clear the entire output capacity, including now-absent rows.
  for(const auto& value:{out.content,out.content_connected,out.initial,out.initial_connected,
                        out.decay,out.decay_connected,out.retention_components,out.retention_connected,out.proposal,out.proposal_connected})
    p.copy(value,at::zeros_like(value));
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_state_vjp_plan)(1,stream,
    ptr(tape.metadata),ptr(tape.count),ptr(tape.config),ptr(previous),ptr(tail),ptr(cot.connected),ptr(cot.final_connected),
    ptr(out.content_connected),ptr(out.initial_connected),ptr(out.decay_connected),ptr(out.retention_connected),
    ptr(tape.clock_policy),ptr(ticks),ptr(out.proposal_connected),ptr(error),capacity,nodes,samples,int64_t(tape.has_repeat),tape.max_repeat_ticks,int64_t(tape.has_attention)),
    "validate and link reverse state chains");},{tape.metadata,tape.count,tape.config,previous,tail,cot.connected,cot.final_connected,
      out.content_connected,out.initial_connected,out.decay_connected,out.retention_connected,tape.clock_policy,ticks,out.proposal_connected,error});
  p.sigmoid(tape.decay,coefficients);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_state_vjp)(blocks,stream,
    ptr(tape.metadata),ptr(tape.values),ptr(tape.config),ptr(coefficients),ptr(previous),ptr(tail),
    ptr(cot.events),ptr(cot.connected),ptr(cot.final),ptr(cot.final_connected),ptr(out.content),ptr(out.content_connected),
    ptr(out.initial),ptr(out.initial_connected),ptr(out.decay),ptr(out.decay_connected),ptr(tape.retention),ptr(ticks),ptr(replay),
    ptr(out.retention_components),ptr(out.proposal),ptr(error),nodes,samples,width,repeat_rows,scratch_width),
    "packed reverse state VJP");},{tape.metadata,tape.values,tape.config,coefficients,previous,tail,cot.events,cot.connected,
      cot.final,cot.final_connected,out.content,out.content_connected,out.initial,out.initial_connected,out.decay,out.decay_connected,
      tape.retention,ticks,replay,out.retention_components,out.proposal,error});
  return out;
}
} // namespace tide::device_online
