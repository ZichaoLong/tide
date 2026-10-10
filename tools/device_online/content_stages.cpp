#include "content_profile.h"
#include "device_backend.h"
#include "packed_sum.h"
#include "packed_aggregate.h"
#include "device_launch_tide_state_read.h"
#include "device_launch_tide_vector_read.h"
#include "device_launch_tide_read_reduce.h"
#include "device_launch_tide_content_state.h"
#include "device_launch_tide_vector_state.h"
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void payloads(const StateKernelProfile& p,std::initializer_list<at::Tensor> values) {
  for(const auto& value:values)if(value.scalar_type()!=p.dtype||value.device()!=p.sources.device()
      ||!value.is_contiguous()||value.requires_grad())
    throw std::invalid_argument("state/Read requires matching colocated contiguous payload dtype");
}
}
ContentBatch append_content(DeviceProgram& p,const ContentProfile& profile,const ReadyBatch& ready,const at::Tensor& error,bool vectorized,
                            const PackedAggregate* aggregate) {
  auto sum=append_packed_sum(p,ready,profile.sources,profile.scales,profile.graph.nodes.size(),
    profile.graph.inputs.size(),profile.graph.edges.size(),error,vectorized,profile.origins);
  if(aggregate)aggregate->append(p,ready,sum,error,vectorized);
  return {sum.content,at::zeros({ready.fibers.size(0)},sum.content.options().dtype(at::kFloat)),sum.weighted};
}
void append_read(DeviceProgram& p,const StateKernelProfile& profile,const ReadyBatch& ready,const ContentBatch& content,
                 const ContentState& old,const at::Tensor& coefficients,const at::Tensor& error,int64_t max_repeat_ticks,bool vectorized,const at::Tensor& attention_proposals) {
  const auto capacity=ready.fibers.size(0),width=profile.width,nodes=int64_t(profile.nodes.size()),samples=old.values.size(0);
  const auto proposals=attention_proposals.defined()?attention_proposals:content.content;
  payloads(profile,{content.content,old.values,coefficients,proposals});
  const int64_t fp16=profile.dtype==at::kHalf;
  if(content.scores.scalar_type()!=at::kFloat)throw std::invalid_argument("Read scores require FP32");
  auto scratch=profile.all_content||vectorized?old.values:at::empty_like(old.values);
  auto clocks=profile.all_content?old.clocks:at::empty_like(old.clocks);
  if(!profile.all_content){if(!vectorized)p.copy(scratch,old.values);p.copy(clocks,old.clocks);}
  auto steps=at::empty({vectorized?capacity:1},ready.fibers.options());
  const auto reads=profile.read,modes=profile.read_modes,kinds=profile.read_kinds,config=profile.config,retention=profile.retention,policy=profile.clock_policy;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_state_read)(1,stream,ptr(ready.fibers),ptr(ready.counts),
    ptr(content.content),ptr(reads),ptr(modes),ptr(kinds),ptr(config),ptr(coefficients),ptr(retention),ptr(policy),ptr(scratch),ptr(clocks),ptr(content.scores),ptr(steps),ptr(proposals),ptr(error),
    capacity,width,nodes,samples,max_repeat_ticks,int64_t(vectorized),fp16),"packed contract-relative Read");},
    {ready.fibers,ready.counts,content.content,reads,modes,kinds,config,coefficients,retention,policy,scratch,clocks,content.scores,steps,proposals,error});
  if(vectorized) {
    const int64_t tiles=(width+255)/256;const uint32_t blocks=std::min<int64_t>(32,capacity*tiles);
    auto partials=at::empty({capacity,tiles},content.scores.options());
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_vector_read)(blocks,stream,
      ptr(ready.fibers),ptr(ready.counts),ptr(content.content),ptr(reads),ptr(modes),ptr(kinds),ptr(config),
      ptr(coefficients),ptr(retention),ptr(old.values),ptr(steps),ptr(partials),ptr(proposals),ptr(error),width,nodes,fp16),"vector Read tiles");},
      {ready.fibers,ready.counts,content.content,reads,modes,kinds,config,coefficients,retention,old.values,steps,partials,proposals,error});
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_read_reduce)(1,stream,
      ptr(ready.fibers),ptr(ready.counts),ptr(kinds),ptr(partials),ptr(content.scores),ptr(error),tiles),"finish Read tiles");},
      {ready.fibers,ready.counts,kinds,partials,content.scores,error});
  }
}
ContentUpdate append_content_state(DeviceProgram& p,const StateKernelProfile& profile,const ReadyBatch& ready,
    const ContentBatch& content,const SelectionProposal& selection,const ContentState& old,
    const at::Tensor& coefficients,const at::Tensor& stages,const at::Tensor& event_count,const at::Tensor& error,const ContentLimits& limits,const at::Tensor& attention_proposals) {
  const bool diagnostics=limits.diagnostics,vectorized=limits.vectorized_state;
  const auto max_ticks=limits.max_repeat_ticks;
  const auto capacity=ready.fibers.size(0),width=profile.width,nodes=int64_t(profile.nodes.size()),samples=old.values.size(0);
  const auto proposals=attention_proposals.defined()?attention_proposals:content.content;
  payloads(profile,{content.content,old.values,coefficients,proposals});
  const int64_t fp16=profile.dtype==at::kHalf;
  if(content.scores.scalar_type()!=at::kFloat||selection.controls.scalar_type()!=at::kFloat)
    throw std::invalid_argument("state score/control journals require FP32");
  ContentUpdate out{{at::empty_like(old.values),at::empty_like(old.clocks),at::empty_like(old.present)},
    {at::zeros({capacity,4},ready.fibers.options()),content.content,selection.active},
    at::zeros_like(content.content),
    at::zeros({diagnostics||vectorized?capacity:1,13},ready.fibers.options()),
    at::zeros({diagnostics?capacity:1,diagnostics?5*width+2:1},content.content.options().dtype(at::kFloat))};
  p.copy(out.state.values,old.values);p.copy(out.state.clocks,old.clocks);p.copy(out.state.present,old.present);
  auto config=profile.config,retention=profile.retention,policy=profile.clock_policy;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_content_state)(1,stream,ptr(ready.fibers),ptr(ready.counts),
    ptr(content.content),ptr(content.scores),ptr(selection.controls),ptr(selection.active),ptr(config),ptr(coefficients),ptr(retention),ptr(policy),
    ptr(out.state.values),ptr(out.state.clocks),ptr(out.state.present),ptr(out.actions.coordinates),ptr(out.comparison),ptr(out.event_meta),
    ptr(out.event_values),ptr(stages),ptr(event_count),ptr(proposals),ptr(error),capacity,width,nodes,samples,int64_t(diagnostics),int64_t(vectorized),max_ticks,fp16),"content state proposal");},
    {ready.fibers,ready.counts,content.content,content.scores,selection.controls,selection.active,config,coefficients,retention,policy,
      out.state.values,out.state.clocks,out.state.present,out.actions.coordinates,out.comparison,out.event_meta,out.event_values,stages,event_count,proposals,error});
  if(vectorized) {
    const uint32_t blocks=std::min<int64_t>(32,capacity);
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_vector_state)(blocks,stream,
      ptr(ready.counts),ptr(content.content),ptr(config),ptr(coefficients),ptr(retention),ptr(policy),ptr(out.state.values),
      ptr(out.comparison),ptr(out.event_meta),ptr(out.event_values),ptr(proposals),ptr(error),width,nodes,int64_t(diagnostics),fp16),"vector state update");},
      {ready.counts,content.content,config,coefficients,retention,policy,out.state.values,out.comparison,out.event_meta,out.event_values,proposals,error});
  }
  return out;
}
void commit_content_state(DeviceProgram& p,const ContentState& old,const ContentUpdate& out,const at::Tensor& error) {
  auto zero=at::zeros_like(error),ok=at::zeros({1},old.present.options()),index=at::zeros_like(error);
  auto commit=p.label(),done=p.label();p.equal(error,zero,ok);p.cast_index(ok,index);p.branch(index,{done,commit});p.mark(commit);
  p.copy(old.values,out.state.values);p.copy(old.clocks,out.state.clocks);p.copy(old.present,out.state.present);p.mark(done);
}
} // namespace tide::device_online
