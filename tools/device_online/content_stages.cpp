#include "content_profile.h"
#include "cann_api.h"
#include "packed_sum.h"
#include "aclrtlaunch_tide_state_read.h"
#include "aclrtlaunch_tide_content_state.h"
#include "aclrtlaunch_tide_content_outputs.h"

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
ContentBatch append_content(CannProgram& p,const ContentProfile& profile,const ReadyBatch& ready,const at::Tensor& error,bool vectorized) {
  auto sum=append_packed_sum(p,ready,profile.sources,profile.scales,profile.graph.nodes.size(),
    profile.graph.inputs.size(),profile.graph.edges.size(),error,vectorized);
  return {sum.content,at::zeros({ready.fibers.size(0)},sum.content.options()),sum.weighted};
}
void append_read(CannProgram& p,const ContentProfile& profile,const ReadyBatch& ready,const ContentBatch& content,
                 const ContentState& old,const at::Tensor& coefficients,const at::Tensor& error) {
  const auto capacity=ready.fibers.size(0),width=profile.width,nodes=int64_t(profile.graph.nodes.size()),samples=old.values.size(0);
  auto scratch=profile.all_content?old.values:at::empty_like(old.values);
  if(!profile.all_content)p.copy(scratch,old.values);
  const auto reads=profile.read,modes=profile.read_modes,config=profile.config;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_state_read)(1,stream,ptr(ready.fibers),ptr(ready.counts),
    ptr(content.content),ptr(reads),ptr(modes),ptr(config),ptr(coefficients),ptr(scratch),ptr(content.scores),ptr(error),
    capacity,width,nodes,samples),"packed contract-relative Read");},
    {ready.fibers,ready.counts,content.content,reads,modes,config,coefficients,scratch,content.scores,error});
}
ContentUpdate append_content_state(CannProgram& p,const ContentProfile& profile,const ReadyBatch& ready,
    const ContentBatch& content,const SelectionProposal& selection,const ContentState& old,
    const at::Tensor& coefficients,const at::Tensor& stages,const at::Tensor& event_count,const at::Tensor& error,bool diagnostics) {
  const auto capacity=ready.fibers.size(0),width=profile.width,nodes=int64_t(profile.graph.nodes.size()),samples=old.values.size(0);
  ContentUpdate out{{at::empty_like(old.values),at::empty_like(old.clocks),at::empty_like(old.present)},
    {at::zeros({capacity,4},ready.fibers.options()),content.content,selection.active},
    at::zeros_like(content.content),
    at::zeros({diagnostics?capacity:1,13},ready.fibers.options()),
    at::zeros({diagnostics?capacity:1,diagnostics?5*width+2:1},content.content.options())};
  p.copy(out.state.values,old.values);p.copy(out.state.clocks,old.clocks);p.copy(out.state.present,old.present);
  auto config=profile.config;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_content_state)(1,stream,ptr(ready.fibers),ptr(ready.counts),
    ptr(content.content),ptr(content.scores),ptr(selection.controls),ptr(selection.active),ptr(config),ptr(coefficients),
    ptr(out.state.values),ptr(out.state.clocks),ptr(out.state.present),ptr(out.actions.coordinates),ptr(out.comparison),ptr(out.event_meta),
    ptr(out.event_values),ptr(stages),ptr(event_count),ptr(error),capacity,width,nodes,samples,int64_t(diagnostics)),"content state proposal");},
    {ready.fibers,ready.counts,content.content,content.scores,selection.controls,selection.active,config,coefficients,
      out.state.values,out.state.clocks,out.state.present,out.actions.coordinates,out.comparison,out.event_meta,out.event_values,stages,event_count,error});
  return out;
}
AtomBatch append_outputs(CannProgram& p,const ContentProfile& profile,const ActionBatch& actions,int64_t capacity,const at::Tensor& error) {
  AtomBatch out{at::zeros({capacity,6},actions.coordinates.options()),at::zeros({capacity,profile.width},actions.values.options()),
    at::zeros({capacity},actions.valid.options())};
  const auto nodes=profile.output_nodes,scales=profile.output_scales;const auto ports=int64_t(profile.graph.outputs.size()),width=profile.width;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_content_outputs)(1,stream,ptr(actions.coordinates),ptr(actions.values),
    ptr(actions.valid),ptr(nodes),ptr(scales),ptr(out.coordinates),ptr(out.values),ptr(out.valid),ptr(error),actions.valid.numel(),
    ports,capacity,width),"content output delivery");},{actions.coordinates,actions.values,actions.valid,nodes,scales,out.coordinates,out.values,out.valid,error});
  return out;
}
void commit_content_state(CannProgram& p,const ContentState& old,const ContentUpdate& out,const at::Tensor& error) {
  auto zero=at::zeros_like(error),ok=at::zeros({1},old.present.options()),index=at::zeros_like(error);
  auto commit=p.label(),done=p.label();p.equal(error,zero,ok);p.cast_index(ok,index);p.branch(index,{done,commit});p.mark(commit);
  p.copy(old.values,out.state.values);p.copy(old.clocks,out.state.clocks);p.copy(old.present,out.state.present);p.mark(done);
}
} // namespace tide::device_online
