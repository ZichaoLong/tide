#include "graph_vjp.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_graph_reverse_meta.h"
#include "aclrtlaunch_tide_graph_reverse_payload.h"
#include "aclrtlaunch_tide_graph_reverse_scales.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>
#include <algorithm>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void tensor(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid graph cotangent buffer");
}
}
GraphVjp append_graph_vjp(CannProgram& p,const ReverseTape& t,const GraphCotangents& roots,
                         const at::Tensor& error,int64_t chunk,int64_t budget) {
  if(at::GradMode::is_enabled()||!t.graph||!t.state.metadata.defined()||t.state.metadata.dim()!=2
      ||!t.fiber_values.defined()||t.fiber_values.dim()!=2)
    throw std::invalid_argument("graph VJP requires no-grad actual device tape");
  const auto device=t.state.metadata.device();const int64_t capacity=t.state.metadata.size(0),width=t.full.width;
  const int64_t nodes=t.graph->nodes.size(),samples=t.state.samples,fibers=t.fiber_values.size(0);
  const auto pending=t.pending.valid.numel(),outputs=t.outputs.valid.numel(),total=fibers+pending+outputs;
  const int64_t parameters=t.graph->inputs.size()+2*t.graph->edges.size()+t.graph->outputs.size(),physical=std::max<int64_t>(1,parameters);
  const long double own=4.L*(total+fibers+physical)*width+4.L*capacity*(11.L*width+15)
    +16.L*samples*nodes*width+16.L*nodes*width+(t.full.has_tanh?4.L*nodes*(width*static_cast<long double>(width)+width):0.L)
    +8.L*total+16.L*capacity+32.L*samples*nodes+32.L*nodes+physical+1024;
  if(device.type()!=c10::DeviceType::PrivateUse1||capacity<1||width<1||nodes<1||samples<1||chunk<1||budget<1||own>budget/2.L)
    throw std::invalid_argument("graph VJP tensor budget exceeded");
  tensor(roots.outputs,device,at::kFloat,{outputs,width});tensor(roots.outputs_connected,device,at::kBool,{outputs});
  tensor(roots.pending,device,at::kFloat,{pending,width});tensor(roots.pending_connected,device,at::kBool,{pending});
  tensor(roots.final,device,at::kFloat,{samples,nodes,width});tensor(roots.final_connected,device,at::kBool,{samples,nodes});
  tensor(t.full_values,device,at::kFloat,{capacity,width});tensor(t.fiber_values,device,at::kFloat,{fibers,width});
  tensor(t.state.values,device,at::kFloat,{capacity,5*width+2});
  // Reserve half for the three bounded components. Each rejects before its
  // allocations; no nested component can consume another component's reserve.
  auto links=append_reverse_links(p,t,error,budget/8);
  auto floats=t.fiber_values.options(),longs=t.state.metadata.options(),booleans=roots.final_connected.options();
  auto messages=at::empty({total,width},floats),connected=at::empty({total},booleans);
  auto carry=at::empty_like(roots.final),carry_on=at::empty_like(roots.final_connected);
  GraphVjp out{links,messages,connected,carry,carry_on,{}, {},at::empty({nodes},booleans),
    at::empty({nodes,width},floats),at::empty({nodes},booleans),at::empty({nodes},floats),at::empty({nodes},booleans),
    at::empty({physical},floats),at::empty({physical},booleans),at::empty({1},longs)};
  if(t.full.has_tanh){out.weights=at::empty({nodes,width,width},floats);out.biases=at::empty({nodes,width},floats);p.zero(out.weights);p.zero(out.biases);}
  auto stage_meta=at::empty_like(t.state.metadata),stage_values=at::empty_like(t.state.values),stage_count=at::empty_like(t.state.count);
  auto full_grad=at::empty({capacity,width},floats),full_on=at::empty({capacity},booleans);
  auto aggregate_partials=at::empty({fibers,width},floats),scalar_partials=at::empty({physical,width},floats);
  auto dc=at::empty_like(carry),rc=at::empty_like(carry),dcon=at::empty_like(carry_on),rcon=at::empty_like(carry_on);
  auto cursor=at::empty({1},longs),range=at::empty({2},longs),branch=at::empty_like(error);
  StateCotangents cot{at::empty({capacity,5,width},floats),at::empty({capacity,5},booleans),carry,carry_on};
  for(const auto& x:{messages,connected,carry,carry_on,stage_values,full_grad,full_on,aggregate_partials,scalar_partials,
    dc,rc,dcon,rcon,out.full_connected,out.decay_connected,out.retention_connected,out.scale_connected,out.reverse_stages})p.zero(x);
  auto meta=[&](int64_t mode,const at::Tensor& fh,const at::Tensor& fc,const at::Tensor& fp,const StateVjp& s) {
    // Unused arguments point at owned buffers of the right scalar type; mode
    // dispatch occurs on device before they are read.
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_graph_reverse_meta)(1,stream,
      ptr(t.state.metadata),ptr(links.stage_offsets),ptr(links.stages),ptr(links.messages),ptr(links.valid),
      ptr(links.producer_head),ptr(links.producer_next),ptr(links.consumer_head),ptr(links.consumer_next),
      ptr(links.scale_head),ptr(links.scale_next),ptr(roots.pending_connected),ptr(roots.outputs_connected),ptr(roots.final_connected),
      ptr(connected),ptr(carry_on),ptr(stage_meta),ptr(stage_count),ptr(full_on),ptr(cot.connected),ptr(cursor),ptr(range),ptr(branch),
      ptr(fh),ptr(fc),ptr(fp),ptr(s.content_connected),ptr(s.decay_connected),ptr(s.retention_connected),ptr(dcon),ptr(rcon),
      ptr(out.full_connected),ptr(out.decay_connected),ptr(out.retention_connected),ptr(out.scale_connected),ptr(out.reverse_stages),ptr(error),
      capacity,fibers,pending,outputs,nodes,samples,parameters,mode),"advance graph reverse stage metadata");},
      {t.state.metadata,links.stage_offsets,links.stages,links.messages,links.valid,links.producer_head,links.producer_next,
       links.consumer_head,links.consumer_next,links.scale_head,links.scale_next,roots.pending_connected,roots.outputs_connected,
       roots.final_connected,connected,carry_on,stage_meta,stage_count,full_on,cot.connected,cursor,range,branch,
       fh,fc,fp,s.content_connected,s.decay_connected,s.retention_connected,dcon,rcon,out.full_connected,out.decay_connected,
       out.retention_connected,out.scale_connected,out.reverse_stages,error});
  };
  StateVjp dummy;dummy.content_connected=full_on;dummy.decay_connected=dcon;dummy.retention_connected=rcon;
  auto payload=[&](int64_t mode,const at::Tensor& fh,const at::Tensor& fc,const at::Tensor& sh,const at::Tensor& shc) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_graph_reverse_payload)(32,stream,
      ptr(t.state.values),ptr(t.fiber_values),ptr(links.messages),ptr(links.scales),ptr(links.producer_head),ptr(links.producer_next),
      ptr(links.consumer_head),ptr(links.consumer_next),ptr(roots.pending),ptr(roots.outputs),ptr(roots.final),
      ptr(messages),ptr(connected),ptr(carry),ptr(carry_on),ptr(stage_values),ptr(full_grad),ptr(range),ptr(stage_count),
      ptr(cot.events),ptr(cot.connected),ptr(fh),ptr(fc),ptr(sh),ptr(shc),ptr(aggregate_partials),ptr(error),
      width,fibers,pending,outputs,nodes,samples,mode),"packed graph reverse payload phase");},
      {t.state.values,t.fiber_values,links.messages,links.scales,links.producer_head,links.producer_next,links.consumer_head,links.consumer_next,
       roots.pending,roots.outputs,roots.final,messages,connected,carry,carry_on,stage_values,full_grad,range,stage_count,
       cot.events,cot.connected,fh,fc,sh,shc,aggregate_partials,error});
  };
  meta(0,full_on,full_on,out.full_connected,dummy);payload(0,full_grad,full_grad,full_grad,full_on);
  auto head=p.label(),body=p.label(),done=p.label();p.mark(head);meta(1,full_on,full_on,out.full_connected,dummy);
  p.branch(branch,{done,body});p.mark(body);payload(1,full_grad,full_grad,full_grad,full_on);
  auto full_tape=t.full;full_tape.metadata=stage_meta;full_tape.values=stage_values;full_tape.count=stage_count;
  auto full=append_full_vjp(p,full_tape,full_grad,full_on,error,chunk,budget/8);
  meta(2,full.content_connected,full.comparison_connected,full.parameter_connected,dummy);
  payload(2,full.content,full.comparison,full_grad,full_on);
  auto state_tape=t.state;state_tape.metadata=stage_meta;state_tape.values=stage_values;state_tape.count=stage_count;
  auto state=append_state_vjp(p,state_tape,cot,error,budget/4);
  payload(3,full.content,full.comparison,state.content,state.content_connected);
  p.copy(carry,state.initial);p.copy(carry_on,state.initial_connected);
  p.add(dc,state.decay);p.add(rc,state.retention_components);
  if(t.full.has_tanh){p.add(out.weights,full.weights);p.add(out.biases,full.biases);}
  meta(3,full.content_connected,full.comparison_connected,full.parameter_connected,state);p.branch(branch,{head});p.mark(done);
  meta(4,full_on,full_on,out.full_connected,dummy);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_graph_reverse_scales)(32,stream,
    ptr(links.messages),ptr(links.scale_head),ptr(links.scale_next),ptr(messages),ptr(connected),ptr(aggregate_partials),
    ptr(t.full_values),ptr(scalar_partials),ptr(error),physical,width),"ordered physical scale adjoints");},
    {links.messages,links.scale_head,links.scale_next,messages,connected,aggregate_partials,t.full_values,scalar_partials,error});
  p.sum(scalar_partials,1,false,out.scales);p.sum(dc,0,false,out.decay);
  auto retention_features=at::empty_like(out.decay);p.sum(rc,0,false,retention_features);p.sum(retention_features,1,false,out.retention);
  return out;
}
} // namespace tide::device_online
