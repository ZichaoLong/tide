#include "content_flow_internal.h"
#include "content_budget.h"
#include "control_forward.h"
#include "portable_torch/runtime.hpp"
#include "tide/ops.h"
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
void validate_reverse_modules(const Graph& graph) {
  for(const auto& n:graph.nodes)
    if((!n.identity&&n.emission!="broadcast")
        ||(!n.identity&&n.memory!="identity"&&n.memory!="ema"&&n.memory!="lh-add-repeat-v1"&&n.memory!="attention"&&!is_fiber_attention_profile(n.memory)))
      throw std::invalid_argument("graph reverse module contract unavailable");
}
}
ContentFlow::Impl::Impl(Graph g,Model m,const Continuation& q,at::Device d,ContentLimits l,at::Device fd)
    :profile(std::move(g),std::move(m),d,true),limits(l),device(d),full_device(fd),boundary(q),window_start(q.cut) {
  if(fd.type()!=d.type()||fd.index()<0)throw std::invalid_argument("Full peer requires an explicit NPU");
  validate_window(profile.graph,profile.model,boundary,{},q.cut,q.cut);
  if((l.mode!="hard"&&l.mode!="hst"&&l.mode!="softp")||!std::isfinite(l.zeta))
    throw std::invalid_argument("invalid resident Emit mode/zeta");
  if(l.mode!="hard")for(const auto& n:profile.graph.nodes)if(n.emission!="broadcast")
    throw std::invalid_argument("resident control modes require broadcast emission");
  const auto nodes=int64_t(profile.graph.nodes.size()),regions=int64_t(profile.graph.regions.size()),width=profile.width,samples=q.batch_size;
  // Static CPU validation/planning precedes any profile/payload device upload.
  // These conservative tensor bounds exclude caller CPU storage/vendor internals.
  long double estimate=64.L*(l.queue+static_cast<long double>(l.arrivals)+l.outputs+(l.diagnostics?l.trace:0))*(width*5.L+32)
    +64.L*samples*(nodes*(width+4.L)+regions*(regions+4.L))
    +160.L*(profile.graph.edges.size()+profile.graph.inputs.size()+profile.graph.outputs.size()+nodes);
  if(l.mode=="softp")estimate+=4.L*l.queue*width+(l.diagnostics?l.trace*(4.L*width+104.L):0.L);
  // Both endpoint packet storage and the returned payload are charged to the
  // same total bound before allocation. Operator arenas share the remainder.
  if(fd!=d)estimate+=8.L*l.queue*(4.L*width+64)+4096;
  if(l.queue<1||l.arrivals<1||l.outputs<1||l.trace<0||(l.diagnostics&&l.trace<1)||l.stages<1||l.full_chunk_rows<1||l.emission_chunk_rows<1||l.aggregate_chunk_rows<1||l.max_repeat_ticks<1||l.workspace_bytes<1||estimate>l.workspace_bytes
      ||(l.chunk_policy!=ChunkPolicy::conservative&&l.chunk_policy!=ChunkPolicy::aggressive))
    throw std::invalid_argument("content flow buffer budget exceeded or invalid limits");
  std::vector<int64_t> kinds,lh_kinds;std::vector<Tensor> weights,biases,norm_weights,norm_biases;bool has_lh=false;
  for(size_t n=0;n<profile.graph.nodes.size();++n){const auto& node=profile.graph.nodes[n];
    const auto& w=profile.model.nodes[n];const auto kind=node.identity?0:lh_full_kind(node.full);has_lh|=kind!=0;lh_kinds.push_back(kind);
    kinds.push_back(!node.identity&&node.full=="tanh");weights.push_back(w.weight);biases.push_back(w.bias);
    norm_weights.push_back(kind&&(kind-1)%3? w.extra.at("lh_norm_weight"):at::ones_like(w.bias));
    norm_biases.push_back(kind&&(kind-1)%3==2? w.extra.at("lh_norm_bias"):at::zeros_like(w.bias));}
  const std::array<long double,7> minimum={
    PackedEmission::minimum_bytes(profile,l.arrivals,l.outputs),
    PackedSwiGluFull::minimum_bytes(profile,l.queue),PackedLhFull::minimum_bytes(lh_kinds,width,l.queue,profile.dtype),
    PackedFiberAttention::minimum_bytes(profile,q,l),PackedEventAttention::minimum_bytes(profile,q,l),
    PackedFull::minimum_bytes(kinds,width,profile.dtype),PackedAggregate::minimum_bytes(profile,l.queue)};
  ContentBudget budget(l.workspace_bytes,l.chunk_policy==ChunkPolicy::aggressive,estimate,minimum);
  profile.upload(device);
  const auto opts=at::TensorOptions().device(device).dtype(profile.dtype);
  error=at::zeros({1},opts.dtype(at::kInt));stop=at::full({1},q.cut,opts.dtype(at::kLong));stages=at::zeros_like(stop);
  event_count=at::zeros_like(stop);
  pending=std::make_unique<QueueTransaction>(l.queue,width,nodes,samples,opts,error);
  outputs=std::make_unique<QueueTransaction>(l.outputs,width,nodes,samples,opts,error);
  if(l.diagnostics)messages=std::make_unique<QueueTransaction>(l.trace,width,nodes,samples,opts,error);
  selector=std::make_unique<FrameSelector>(profile.owners,profile.policies,samples,device,l.workspace_bytes);
  history=selector->initial();
  if(l.diagnostics)history_before=selector->initial();
  state={at::zeros({samples,nodes,width},opts),at::zeros({samples,nodes,2},opts.dtype(at::kLong)),
         at::zeros({samples,nodes},opts.dtype(at::kBool))};
  auto values=at::zeros({samples,nodes,width},at::TensorOptions().dtype(profile.dtype)),clocks=at::zeros({samples,nodes,2},at::kLong),present=at::zeros({samples,nodes},at::kBool);
  clocks.select(2,0).fill_(-1);
  for(const auto& [owner,s]:q.states){auto [b,n]=owner;values[b][n].copy_(s.value);clocks[b][n][0].fill_(s.last_time);
    clocks[b][n][1].fill_(s.observations);present[b][n].fill_(true);}
  state.values.copy_(values);state.clocks.copy_(clocks);state.present.copy_(present);
  auto counts=at::zeros({samples,nodes},at::kLong),seen=at::zeros({samples,nodes},at::kBool);
  auto last=at::full({samples,regions},-1,at::kLong),has=at::zeros({samples,regions},at::kBool);
  for(const auto& [owner,h]:q.history){auto [b,r]=owner;has[b][r].fill_(true);last[b][r].fill_(h.last_time);
    for(const auto& [n,c]:h.node_maps.at("selected")){counts[b][n].fill_(c);seen[b][n].fill_(true);}}
  history.counts.copy_(counts);history.seen.copy_(seen);history.last_time.copy_(last);history.present.copy_(has);
  upload_atoms(q.pending,pending->atoms());pending->stats().fill_(q.pending.size());
  external={at::zeros({l.queue,6},opts.dtype(at::kLong)),at::zeros({l.queue,width},opts),at::zeros({l.queue},opts.dtype(at::kBool))};
  if(l.diagnostics) {
    events=std::make_unique<DeviceJournal>(l.trace,13,5*width+2,device);
    fibers=std::make_unique<DeviceJournal>(l.trace,6,width,device);
    contributions=std::make_unique<DeviceJournal>(l.trace,6,width,device);
    full_trace=std::make_unique<DeviceJournal>(l.trace,13,width,device);
    if(l.mode=="softp")raw_full_trace=std::make_unique<DeviceJournal>(l.trace,13,width,device);
    emission_trace=std::make_unique<DeviceJournal>(l.trace,6,width,device);
  }
  emission=std::make_unique<PackedEmission>(profile,device,samples,l.arrivals,l.outputs,
    std::min(l.emission_chunk_rows,l.arrivals+l.outputs),budget.available(0));
  budget.reserve(0,emission->reserved_bytes());
  const auto max_full=std::min(l.full_chunk_rows,l.queue);
  if(minimum[1]>0) {
    swiglu_full=std::make_unique<PackedSwiGluFull>(profile,full_device,l.queue,max_full,budget.available(1));
    budget.reserve(1,swiglu_full->reserved_bytes());
  }
  if(has_lh) {
    lh_full=std::make_unique<PackedLhFull>(lh_kinds,at::stack(norm_weights),at::stack(norm_biases),full_device,l.queue,max_full,budget.available(2));
    budget.reserve(2,lh_full->reserved_bytes());
  }
  if(minimum[3]>0) {
    attention=std::make_unique<PackedFiberAttention>(profile,q,device,l,budget.available(3));
    budget.reserve(3,attention->reserved_bytes());
  }
  if(minimum[4]>0) {
    event_attention=std::make_unique<PackedEventAttention>(profile,q,device,l,budget.available(4));
    budget.reserve(4,event_attention->reserved_bytes());
  }
  full=std::make_unique<PackedFull>(kinds,at::stack(weights),at::stack(biases),full_device,max_full,budget.available(5));
  budget.reserve(5,full->reserved_bytes());
  if(minimum[6]>0) {
    aggregate=std::make_unique<PackedAggregate>(profile,device,l.queue,l.aggregate_chunk_rows,budget.available(6));
    budget.reserve(6,aggregate->reserved_bytes());
  }
  planned_buffer_bytes=budget.planned_bytes();operator_workspace_budget=budget.operator_budget();usable_memory_budget=budget.usable_bytes();
  full_chunks=full->chunks();
  if(fd!=d) {
    operator_workspace_budget/=2;
    if(operator_workspace_budget<1)throw std::invalid_argument("insufficient peer operator workspace");
    full_chunks=at::zeros({1},stop.options());
    remote_full=std::make_unique<RemoteFull>(*full,lh_full.get(),swiglu_full.get(),operator_workspace_budget);
  }
  construct();
  // Only input-seal/ledger metadata belongs on the host between windows.
  // The authoritative state, history and pending payloads are device owners.
  boundary.states.clear();boundary.history.clear();boundary.pending.clear();
}
void ContentFlow::Impl::construct() {
  const auto& g=profile.graph;const auto opts=state.values.options();
  DeviceReady planner(profile.owners,g.regions.size(),profile.wires,boundary.batch_size,device,limits.prefill,profile.causal_regions);
  program=std::make_unique<CannProgram>(device);auto& p=*program;p.limit_workspace(operator_workspace_budget);
  auto zeros=at::zeros({limits.queue},error.options()),out_zeros=at::zeros({limits.outputs},error.options());
  if(limits.diagnostics) {
    // Preserve the parameters that generated this window's recorded sources;
    // a later optimizer publish must not rewrite diagnostic provenance.
    source_scales_before=at::zeros_like(profile.scales);p.copy(source_scales_before,profile.scales);
    p.copy(history_before.counts,history.counts);p.copy(history_before.seen,history.seen);
    p.copy(history_before.last_time,history.last_time);p.copy(history_before.present,history.present);
  }
  // Inputs are boundary uploads; all recursive messages below are generated on device.
  pending->append_stage(p,zeros,external);
  auto coefficients=at::empty_like(profile.decay);p.sigmoid(profile.decay,coefficients);
  auto budget=at::full({1},limits.stages,stop.options()),one=at::ones_like(stop),budget_error=at::full_like(error,4);
  auto predicate=at::zeros({1},opts.dtype(at::kBool)),index=at::zeros_like(error);
  auto head=p.label(),test=p.label(),body=p.label(),exhausted=p.label(),end=p.label();
  p.mark(head);auto ready=planner.append_stage(p,pending->atoms(),stop,error);p.branch(ready.branch,{end,test});
  p.mark(test);p.less(stages,budget,predicate);p.cast_index(predicate,index);p.branch(index,{exhausted,body});p.mark(body);
  auto content=append_content(p,profile,ready,error,limits.vectorized_aggregate,aggregate.get());
  FiberStage attended;
  if(attention)attended=attention->propose(p,profile,ready,content,state,error);
  EventAttentionStage event_attended;auto proposals=attended.values;
  if(event_attention){event_attended=event_attention->propose(p,ready,content,error,proposals);proposals=event_attended.values;}
  append_read(p,profile,ready,content,state,coefficients,error,limits.max_repeat_ticks,limits.vectorized_read,proposals);
  auto selection=selector->append_stage(p,ready,content.scores,history,error);
  auto update=append_content_state(p,profile,ready,content,selection,state,coefficients,stages,event_count,error,limits,proposals);
  auto actions=update.actions;
  if(remote_full)actions=remote_full->append_stage(p,actions,content.content,update.comparison,error,full_chunks);
  else {
    actions=full->append_stage(p,actions,update.comparison,error);
    if(lh_full)actions=lh_full->append_stage(p,actions,update.comparison,error,full_chunks);
    if(swiglu_full)actions=swiglu_full->append_stage(p,actions,content.content,update.comparison,error,full_chunks);
  }
  const auto raw_full=actions.values;
  if(limits.mode=="softp")actions=append_control_forward(p,profile,actions,content.content,selection.controls,error);
  auto emitted=emission->append_stage(p,actions,error);
  auto arrivals=emitted.arrivals;
  // Every capacity/error preflight precedes every live state/history/queue/log commit.
  auto pending_proposal=pending->propose_stage(p,ready.consumed,arrivals);
  auto output_proposal=outputs->propose_stage(p,out_zeros,emitted.outputs);
  if(limits.diagnostics) {
    auto msg_zeros=at::zeros({limits.trace},error.options());
    auto message_proposal=messages->propose_stage(p,msg_zeros,arrivals);
    auto event_proposal=events->propose(p,update.event_meta,update.event_values,ready.counts.narrow(0,1,1),error);
    auto fiber_proposal=fibers->propose(p,ready.atoms.coordinates,ready.atoms.values,ready.counts.narrow(0,0,1),error);
    auto contribution_proposal=contributions->propose(p,ready.atoms.coordinates,content.weighted,ready.counts.narrow(0,0,1),error);
    auto full_proposal=full_trace->propose(p,update.event_meta,actions.values,ready.counts.narrow(0,1,1),error);
    JournalProposal raw_proposal;
    if(raw_full_trace)raw_proposal=raw_full_trace->propose(p,update.event_meta,raw_full,ready.counts.narrow(0,1,1),error);
    auto emission_proposal=emission_trace->propose(p,emitted.meta,emitted.values,emitted.count,error);
    messages->commit_stage(p,message_proposal);
    events->commit(p,event_proposal,error);fibers->commit(p,fiber_proposal,error);contributions->commit(p,contribution_proposal,error);
    full_trace->commit(p,full_proposal,error);
    if(raw_full_trace)raw_full_trace->commit(p,raw_proposal,error);
    emission_trace->commit(p,emission_proposal,error);
  }
  pending->commit_stage(p,pending_proposal);outputs->commit_stage(p,output_proposal);
  selector->append_commit(p,history,selection,error);commit_content_state(p,state,update,error);
  if(attention)attention->commit(p,attended,selection,error);
  if(event_attention)event_attention->commit(p,event_attended,selection,error);
  p.add(stages,one);p.branch(index,{head});p.mark(exhausted);p.copy(error,budget_error);p.mark(end);
  if(remote_full)remote_full->append_stop(p);
  p.finish();
}
ContentFlow::ContentFlow(Graph g,Model m,const Continuation& q,at::Device d,ContentLimits l)
    :ContentFlow(std::move(g),std::move(m),q,d,l,d) {}
ContentFlow::ContentFlow(Graph g,Model m,const Continuation& q,at::Device d,ContentLimits l,at::Device fd)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(m),q,d,l,fd)) {}
ContentFlow::~ContentFlow()=default;
void ContentFlow::close() {
  if(!impl_)return;
  impl_->program->close();if(impl_->remote_full)impl_->remote_full->close();impl_.reset();
}
Result ContentFlow::advance(const std::vector<External>& input,Index until) {
  advance_device(input,until);
  try {return result();}catch(...){impl_->failed=true;throw;}
}
ContentWindow ContentFlow::advance_device(const std::vector<External>& input,Index until) {
  if(!impl_)throw std::logic_error("content flow is closed");
  auto& s=*impl_;
  if(s.failed)throw std::logic_error("content flow failed; restore an earlier complete cut into a new owner");
  if(at::GradMode::is_enabled())throw std::invalid_argument("content flow has no autograd contract");
  auto validated=prepare_external(s.profile.graph,s.profile.model,s.boundary,input,until,s.device,s.external);
  try {
    s.outputs->atoms().valid.zero_();s.outputs->stats().zero_();
    if(s.limits.diagnostics) {
      s.messages->atoms().valid.zero_();s.messages->stats().zero_();s.events->count.zero_();s.fibers->count.zero_();s.contributions->count.zero_();
      s.full_trace->count.zero_();s.emission_trace->count.zero_();
      if(s.raw_full_trace)s.raw_full_trace->count.zero_();
    }
    if(s.attention)s.attention->reset_window();
    if(s.event_attention)s.event_attention->reset_window();
    if(s.aggregate)s.aggregate->chunks().zero_();
    s.event_count.zero_();s.full->chunks().zero_();if(s.remote_full)s.full_chunks.zero_();
    s.emission->chunks().zero_();s.stages.zero_();s.stop.fill_(until);
    // Dispatch through the process's registered owner. This works with either
    // the standalone SDK or the Python wheel, never linking both together.
    c10::impl::VirtualGuardImpl(s.device.type()).synchronizeDevice(s.device.index());
    if(s.remote_full) {
      s.remote_full->synchronize_inputs();s.program->submit();s.remote_full->submit();
      std::exception_ptr failure;
      try{s.program->wait();}catch(...){failure=std::current_exception();}
      try{s.remote_full->wait();}catch(...){if(!failure)failure=std::current_exception();}
      if(failure)std::rethrow_exception(failure);
    } else s.program->run();
    const auto error=s.error.cpu().item<int>();
    if(error)throw std::runtime_error("content flow device refusal code="+std::to_string(error));
    s.window_start=s.boundary.cut;s.boundary.cut=until;for(const auto& [owner,last]:validated.ledger_updates)s.boundary.ledger[owner]=last;
    return {s.outputs->atoms(),s.outputs->stats(),s.pending->stats(),s.stages,s.event_count,s.full_chunks,s.emission->chunks()};
  } catch(...) {s.failed=true;throw;}
}
Continuation ContentFlow::snapshot() const {
  if(!impl_)throw std::logic_error("content flow is closed");
  if(impl_->failed)throw std::logic_error("content flow failed; complete-cut snapshot unavailable");
  return impl_->export_continuation();
}
Result ContentFlow::result() const {
  if(!impl_)throw std::logic_error("content flow is closed");
  if(impl_->failed)throw std::logic_error("content flow failed; result unavailable");
  return impl_->export_result();
}
StateTape ContentFlow::state_tape() const {
  if(!impl_||impl_->failed)throw std::logic_error("state tape unavailable on closed/failed content flow");
  const auto& s=*impl_;
  if(!s.limits.diagnostics)throw std::logic_error("state tape requires recorded forward values");
  const bool repeat=std::any_of(s.profile.graph.nodes.begin(),s.profile.graph.nodes.end(),
    [](const auto& n){return !n.identity&&n.memory=="lh-add-repeat-v1";});
  return {s.events->meta,s.events->values,s.events->count,s.profile.config,s.profile.decay,
          s.profile.retention,s.profile.clock_policy,s.boundary.batch_size,repeat,s.limits.max_repeat_ticks,bool(s.event_attention)||bool(s.attention)};
}
FullTape ContentFlow::full_tape() const {
  if(!impl_||impl_->failed)throw std::logic_error("Full tape unavailable on closed/failed content flow");
  const auto& s=*impl_;
  if(s.remote_full)throw std::invalid_argument("remote Full adjoint is not implemented");
  if(!s.limits.diagnostics)throw std::logic_error("Full tape requires recorded forward values");
  FullTape out{s.events->meta,s.events->values,s.events->count,s.full->kinds(),s.full->weights(),s.full->biases(),
          s.boundary.batch_size,s.profile.width,s.full->has_tanh()};
  if(s.lh_full)s.lh_full->tape(out.extra);if(s.swiglu_full)s.swiglu_full->tape(out.extra);
  return out;
}
ReverseTape ContentFlow::reverse_tape() const {
  auto state=state_tape();auto full=full_tape();const auto& s=*impl_;
  // Journal access is distinct from complete graph reverse capability. Local
  // half components consume their own actual tapes; append_graph_vjp retains
  // the complete-graph dtype gate until every integration is qualified.
  validate_reverse_modules(s.profile.graph);
  ReverseTape tape{&s.profile.graph,state,full,s.full_trace->values,s.fibers->meta,s.fibers->values,s.fibers->count,
          s.profile.sources,s.profile.scales,s.emission->scales(),s.pending->atoms(),s.outputs->atoms(),
          s.pending->stats().narrow(0,0,1),s.outputs->stats().narrow(0,0,1),s.window_start,s.boundary.cut};
  if(s.aggregate)tape.aggregate=s.aggregate->tape();
  if(s.event_attention)tape.attention=s.event_attention->tape();
  if(s.attention)tape.fiber=s.attention->tape();
  if(s.limits.mode!="hard")tape.control={s.profile.read,s.raw_full_trace?s.raw_full_trace->values:s.full_trace->values,
    s.limits.mode=="hst"?1:2,s.limits.zeta};
  return tape;
}
ParameterBanks ContentFlow::parameter_banks() const {
  // Publication is independently testable before a dtype's VJP is available.
  // This internal view does not expose mutation through public inference.
  const auto full=full_tape();const auto& s=*impl_;validate_reverse_modules(s.profile.graph);
  return {&s.profile.graph,full.weights,full.biases,s.profile.decay,s.profile.retention,
          s.profile.read,s.profile.scales,s.emission->scales(),full.extra,
          s.aggregate?s.aggregate->tape():AggregateTape{},
          s.event_attention?s.event_attention->tape():std::vector<EventAttentionTape>{},
          s.attention?s.attention->banks():FiberParameterBanks{}};
}
std::pair<Tensor,Tensor> ContentFlow::state_device() const {
  if(!impl_||impl_->failed)throw std::logic_error("state view unavailable on closed/failed content flow");
  return {impl_->state.values,impl_->state.present};
}
} // namespace tide::device_online
