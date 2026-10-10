#include "device_backend.h"
#include "graph_vjp.h"
#include "emission_vjp.h"
#include "aggregate_vjp.h"
#include "event_reverse.h"
#include "fiber_reverse.h"
#include "device_launch_tide_graph_reverse_meta.h"
#include "device_launch_tide_graph_reverse_payload.h"
#include "device_launch_tide_graph_reverse_scales.h"
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
GraphVjp append_graph_vjp(DeviceProgram& p,const ReverseTape& t,const GraphCotangents& roots,
                         const at::Tensor& error,int64_t chunk,int64_t budget) {
  return append_graph_vjp(p,t,roots,error,chunk,budget,{});
}
GraphVjp append_graph_vjp(DeviceProgram& p,const ReverseTape& t,const GraphCotangents& roots,
                         const at::Tensor& error,int64_t chunk,int64_t budget,const FullStageVjp& full_stage) {
  return append_graph_vjp(p,t,roots,error,chunk,budget,full_stage,{});
}
GraphVjp append_graph_vjp(DeviceProgram& p,const ReverseTape& t,const GraphCotangents& roots,
    const at::Tensor& error,int64_t chunk,int64_t budget,const FullStageVjp& full_stage,const GraphStateVjp& state_owner,int64_t projection_workspace) {
  return append_graph_vjp(p,t,roots,error,chunk,budget,full_stage,state_owner,projection_workspace,{});
}
GraphVjp append_graph_vjp(DeviceProgram& p,const ReverseTape& t,const GraphCotangents& roots,
    const at::Tensor& error,int64_t chunk,int64_t budget,const FullStageVjp& full_stage,const GraphStateVjp& state_owner,
    int64_t projection_workspace,const std::vector<ProjectionGradient>& reuse) {
  const bool sharded=bool(state_owner.stage);
  if(sharded&&(!state_owner.prepare||!state_owner.sources))throw std::invalid_argument("incomplete compact state reverse executor");
  if(!sharded&&!t.state.decay.defined())throw std::invalid_argument("compact state tape requires its reverse executor");
  if(at::GradMode::is_enabled()||!t.graph||!t.state.metadata.defined()||t.state.metadata.dim()!=2
      ||!t.fiber_values.defined()||t.fiber_values.dim()!=2)
    throw std::invalid_argument("graph VJP requires no-grad actual device tape");
  if(!t.full.kinds.defined()&&!full_stage)
    throw std::invalid_argument("sharded tape requires its Full reverse executor");
  // Identity/LH/SwiGLU-only graphs need no tanh weight bank. Physical source
  // scales always exist (including the source-free sentinel) and own dtype.
  if(t.source_scales.scalar_type()!=at::kFloat&&t.source_scales.scalar_type()!=at::kHalf)
    throw std::invalid_argument("graph VJP requires FP32/FP16 forward parameters and FP32 cotangents");
  const auto device=t.state.metadata.device();const int64_t capacity=t.state.metadata.size(0),width=t.full.width;
  const int64_t nodes=t.graph->nodes.size(),samples=t.state.samples,fibers=t.fiber_values.size(0);
  const auto pending=t.pending.valid.numel(),outputs=t.outputs.valid.numel(),total=fibers+pending+outputs;
  const int64_t parameters=t.graph->inputs.size()+2*t.graph->edges.size()+t.graph->outputs.size(),physical=std::max<int64_t>(1,parameters);
  long double extra_bytes=0;
  for(const auto& x:{t.full.extra.lh_weights,t.full.extra.lh_biases,t.full.extra.gate,t.full.extra.up,t.full.extra.down})
    if(x.defined())extra_bytes+=4.L*x.numel();
  const bool attention=!t.attention.empty();
  const auto attention_offsets=event_parameter_offsets(*t.graph,width);
  if(attention)extra_bytes+=4.L*attention_offsets.back()+4.L*nodes;
  const bool fiber=!t.fiber.empty();const auto fiber_offsets=fiber_parameter_offsets(*t.graph,width);
  if(fiber)extra_bytes+=4.L*fiber_offsets.back()+6.L*nodes;
  const bool normalized=t.aggregate.kinds.defined();
  const bool controlled=t.control.mode!=0,affine=t.emission.weights.defined()||!t.emission.shards.empty();
  bool needs_emission=false;
  for(size_t n=0;n<t.graph->nodes.size();++n)needs_emission|=!t.graph->nodes[n].identity&&t.graph->nodes[n].emission=="slot_affine"
    &&t.graph->outgoing_ports.offsets[n+1]>t.graph->outgoing_ports.offsets[n];
  if(needs_emission&&!affine)throw std::invalid_argument("slot-affine graph reverse requires its emission journal");
  if(!reuse.empty()&&!affine)throw std::invalid_argument("reusable projection gradients require an emission journal");
  if(affine&&controlled)throw std::invalid_argument("controlled slot-affine reverse is not implemented");
  if(controlled)extra_bytes+=5.L*nodes*width;
  if(normalized)extra_bytes+=5.L*nodes*t.aggregate.slots+8;
  const long double own=extra_bytes+4.L*(total+fibers+physical)*width+4.L*capacity*(11.L*width+15)
    +16.L*samples*nodes*width+16.L*nodes*width+(t.full.has_tanh?4.L*nodes*(width*static_cast<long double>(width)+width):0.L)
    +8.L*total+16.L*capacity+32.L*samples*nodes+32.L*nodes+physical+1024;
  if(device.type()!=tide::device_online::resident_device_type||capacity<1||width<1||nodes<1||samples<1||chunk<1||budget<1||own>budget/2.L)
    throw std::invalid_argument("graph VJP tensor budget exceeded");
  tensor(roots.outputs,device,at::kFloat,{outputs,width});tensor(roots.outputs_connected,device,at::kBool,{outputs});
  tensor(roots.pending,device,at::kFloat,{pending,width});tensor(roots.pending_connected,device,at::kBool,{pending});
  tensor(roots.final,device,at::kFloat,{samples,nodes,width});tensor(roots.final_connected,device,at::kBool,{samples,nodes});
  tensor(t.full_values,device,at::kFloat,{capacity,width});tensor(t.fiber_values,device,at::kFloat,{fibers,width});
  tensor(t.state.values,device,at::kFloat,{capacity,5*width+2});
  // Reserve half for the bounded reverse components. Each rejects before its
  // allocations; no nested component can consume another component's reserve.
  const int divisor=(attention||fiber?24:controlled?12:normalized?10:8)+(affine?4:0);
  auto links=append_reverse_links(p,t,error,budget/divisor);
  if(sharded)state_owner.prepare(p,links);
  EmissionReverse emission;
  if(affine)emission=prepare_emission_reverse(p,t,links,error,budget/divisor,chunk,projection_workspace,reuse);
  auto floats=t.fiber_values.options(),longs=t.state.metadata.options(),booleans=roots.final_connected.options();
  auto messages=at::empty({total,width},floats),connected=at::empty({total},booleans);
  auto carry=at::empty_like(roots.final),carry_on=at::empty_like(roots.final_connected);
  GraphVjp out{links,messages,connected,carry,carry_on,{}, {},at::empty({nodes},booleans),
    at::empty({nodes,width},floats),at::empty({nodes},booleans),at::empty({nodes},floats),at::empty({nodes},booleans),
    at::empty({physical},floats),at::empty({physical},booleans),at::empty({1},longs)};
  if(affine)out.emission=emission.gradient;
  if(t.full.has_tanh){out.weights=at::empty({nodes,width,width},floats);out.biases=at::empty({nodes,width},floats);p.zero(out.weights);p.zero(out.biases);}
  auto extra_output=[&](const at::Tensor& bank) {
    if(!bank.defined())return at::Tensor();
    auto shape=bank.sizes().vec();if(shape.empty()||shape[0]<2)throw std::invalid_argument("invalid Full extra bank extent");
    --shape[0];auto x=at::empty(shape,floats);p.zero(x);return x;
  };
  out.extra={extra_output(t.full.extra.lh_weights),extra_output(t.full.extra.lh_biases),
    extra_output(t.full.extra.gate),extra_output(t.full.extra.up),extra_output(t.full.extra.down)};
  if(normalized) {
    out.aggregate={at::empty_like(t.aggregate.weights),at::empty(t.aggregate.weights.sizes(),booleans),at::empty({1},longs)};
    p.zero(out.aggregate.values);p.zero(out.aggregate.connected);p.zero(out.aggregate.chunks);
  }
  if(controlled) {
    out.read=at::empty({nodes,width},floats);out.read_connected=at::empty({nodes},booleans);
    p.zero(out.read);p.zero(out.read_connected);
  }
  if(!roots.cache.empty()&&roots.cache.size()!=t.attention.size()+t.fiber.size())throw std::invalid_argument("cache root group count mismatch");
  std::vector<EventReverse> attention_reverse;
  if(attention) {
    out.attention=at::empty({attention_offsets.back()},floats);out.attention_connected=at::empty({nodes,4},booleans);
    p.zero(out.attention);p.zero(out.attention_connected);
    for(size_t i=0;i<t.attention.size();++i) {
      attention_reverse.push_back(prepare_event_reverse(p,t,t.attention[i],roots.cache.empty()?CacheCotangents{}:roots.cache[i],error,budget/divisor/t.attention.size()));
      out.cache.push_back(attention_reverse.back().cache);
    }
  }
  std::vector<FiberReverse> fiber_reverse;
  if(fiber) {
    out.fiber=at::empty({fiber_offsets.back()},floats);out.fiber_connected=at::empty({nodes,6},booleans);
    p.zero(out.fiber);p.zero(out.fiber_connected);
    for(size_t i=0;i<t.fiber.size();++i) {
      fiber_reverse.push_back(prepare_fiber_reverse(p,t,links,t.fiber[i],roots.cache.empty()?CacheCotangents{}:roots.cache[t.attention.size()+i],error,budget/divisor/t.fiber.size()));
      out.cache.push_back(fiber_reverse.back().cache);
    }
  }
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
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_graph_reverse_meta)(1,stream,
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
    const auto aggregate_kinds=normalized?t.aggregate.kinds:t.state.count;
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_graph_reverse_payload)(32,stream,
      ptr(t.state.values),ptr(t.fiber_values),ptr(links.messages),ptr(links.scales),ptr(links.producer_head),ptr(links.producer_next),
      ptr(links.consumer_head),ptr(links.consumer_next),ptr(roots.pending),ptr(roots.outputs),ptr(roots.final),
      ptr(messages),ptr(connected),ptr(carry),ptr(carry_on),ptr(stage_values),ptr(full_grad),ptr(range),ptr(stage_count),
      ptr(cot.events),ptr(cot.connected),ptr(fh),ptr(fc),ptr(sh),ptr(shc),ptr(aggregate_partials),ptr(error),ptr(t.state.metadata),ptr(aggregate_kinds),
      width,fibers,pending,outputs,nodes,samples,mode,int64_t(normalized)),"packed graph reverse payload phase");},
      {t.state.values,t.fiber_values,links.messages,links.scales,links.producer_head,links.producer_next,links.consumer_head,links.consumer_next,
       roots.pending,roots.outputs,roots.final,messages,connected,carry,carry_on,stage_values,full_grad,range,stage_count,
       cot.events,cot.connected,fh,fc,sh,shc,aggregate_partials,error,t.state.metadata,aggregate_kinds});
  };
  meta(0,full_on,full_on,out.full_connected,dummy);payload(0,full_grad,full_grad,full_grad,full_on);
  auto head=p.label(),body=p.label(),done=p.label();p.mark(head);meta(1,full_on,full_on,out.full_connected,dummy);
  p.branch(branch,{done,body});p.mark(body);payload(1,full_grad,full_grad,full_grad,full_on);
  if(affine)append_emission_reverse(p,t,links,emission,messages,connected,range,full_grad,error,chunk,budget/divisor);
  auto full_tape=t.full;full_tape.metadata=stage_meta;full_tape.values=stage_values;full_tape.count=stage_count;
  ControlVjp control;ControlScores scores;
  if(controlled) {
    if(sharded){scores=append_control_scores(p,*t.graph,t.state,t.control,t.source_scales.scalar_type(),stage_count,range,full_grad,full_on,error,budget/divisor);control=scores.emit;}
    else control=append_control_vjp(p,*t.graph,t.state,t.control,stage_count,range,full_grad,full_on,error,budget/divisor);
  }
  auto full=full_stage?full_stage(p,full_tape,controlled?control.fresh:full_grad,full_on,error):
    append_full_vjp(p,full_tape,controlled?control.fresh:full_grad,full_on,error,chunk,budget/divisor);
  meta(2,full.content_connected,full.comparison_connected,full.parameter_connected,dummy);
  payload(2,full.content,full.comparison,full_grad,full_on);
  if(controlled)append_control_merge(p,control,cot,error);
  auto state_tape=t.state;state_tape.metadata=stage_meta;state_tape.values=stage_values;state_tape.count=stage_count;
  auto state=sharded?state_owner.stage(p,range,cot,scores):append_state_vjp(p,state_tape,cot,error,budget/(divisor/2));
  for(size_t i=0;i<t.attention.size();++i)
    record_reverse_plan(out.statistics,append_event_reverse(p,t,t.attention[i],attention_reverse[i],range,state,out.attention,out.attention_connected,error,chunk,budget/(divisor/2)/t.attention.size()),"event");
  payload(3,full.content,full.comparison,state.content,state.content_connected);
  if(normalized)append_aggregate_vjp(p,t,links,stage_count,range,state.content,state.content_connected,
    messages,aggregate_partials,out.aggregate,error,chunk,budget/divisor);
  for(size_t i=0;i<t.fiber.size();++i)
    record_reverse_plan(out.statistics,append_fiber_reverse(p,t,links,t.fiber[i],fiber_reverse[i],range,state,messages,connected,aggregate_partials,
      out.fiber,out.fiber_connected,error,chunk,budget/(divisor/2)/t.fiber.size()),"fiber");
  if(sharded)state_owner.sources(p,messages,connected,aggregate_partials);
  p.copy(carry,state.initial);p.copy(carry_on,state.initial_connected);
  p.add(dc,state.decay);p.add(rc,state.retention_components);
  if(controlled&&!sharded) {
    p.add(out.read,control.read);
    append_connection_union(p,control.read_connected,out.read_connected,error);
  }
  if(t.full.has_tanh){p.add(out.weights,full.weights);p.add(out.biases,full.biases);}
  for(const auto& pair:{std::make_pair(out.extra.lh_weights,full.extra.lh_weights),std::make_pair(out.extra.lh_biases,full.extra.lh_biases),
    std::make_pair(out.extra.gate,full.extra.gate),std::make_pair(out.extra.up,full.extra.up),std::make_pair(out.extra.down,full.extra.down)})
    if(pair.first.defined())p.add(pair.first,pair.second);
  meta(3,full.content_connected,full.comparison_connected,full.parameter_connected,state);p.branch(branch,{head});p.mark(done);
  if(emission.gradient.program)emission.gradient.program->append_stop(p);
  meta(4,full_on,full_on,out.full_connected,dummy);
  const auto emission_rows=affine?emission.rows:links.messages,emission_values=affine?t.emission.values:t.full_values;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_graph_reverse_scales)(32,stream,
    ptr(links.messages),ptr(links.scale_head),ptr(links.scale_next),ptr(messages),ptr(connected),ptr(aggregate_partials),
    ptr(t.full_values),ptr(scalar_partials),ptr(error),ptr(emission_rows),ptr(emission_values),physical,width,int64_t(affine)),"ordered physical scale adjoints");},
    {links.messages,links.scale_head,links.scale_next,messages,connected,aggregate_partials,t.full_values,scalar_partials,error,emission_rows,emission_values});
  p.sum(scalar_partials,1,false,out.scales);p.sum(dc,0,false,out.decay);
  auto retention_features=at::empty_like(out.decay);p.sum(rc,0,false,retention_features);p.sum(retention_features,1,false,out.retention);
  return out;
}
} // namespace tide::device_online
