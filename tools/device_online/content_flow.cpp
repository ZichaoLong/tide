#include "content_flow_internal.h"
#include "portable_torch/runtime.hpp"
#include "tide/ops.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
ContentFlow::Impl::Impl(Graph g,Model m,const Continuation& q,at::Device d,ContentLimits l)
    :profile(std::move(g),std::move(m),d),limits(l),device(d),boundary(q) {
  validate_window(profile.graph,profile.model,boundary,{},q.cut,q.cut);
  const auto nodes=int64_t(profile.graph.nodes.size()),regions=int64_t(profile.graph.regions.size()),width=profile.width,samples=q.batch_size;
  // Conservative arithmetic-only preflight of persistent and packed scratch.
  // It is deliberately bounded and is not yet the model/KV/training chunker.
  long double estimate=64.L*(l.queue+static_cast<long double>(l.arrivals)+l.outputs+(l.diagnostics?l.trace:0))*(width*5.L+32)
    +64.L*samples*(nodes*(width+4.L)+regions*(regions+4.L));
  if(l.queue<1||l.arrivals<1||l.outputs<1||l.trace<0||(l.diagnostics&&l.trace<1)||l.stages<1||l.full_chunk_rows<1||l.emission_chunk_rows<1||l.max_repeat_ticks<1||l.workspace_bytes<1||estimate>l.workspace_bytes)
    throw std::invalid_argument("content flow buffer budget exceeded or invalid limits");
  const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
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
  auto values=at::zeros({samples,nodes,width},at::kFloat),clocks=at::zeros({samples,nodes,2},at::kLong),present=at::zeros({samples,nodes},at::kBool);
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
    emission_trace=std::make_unique<DeviceJournal>(l.trace,6,width,device);
  }
  std::vector<int64_t> kinds,lh_kinds;std::vector<Tensor> weights,biases,norm_weights,norm_biases;bool has_lh=false;
  for(size_t n=0;n<profile.graph.nodes.size();++n){const auto& node=profile.graph.nodes[n];
    const auto& w=profile.model.nodes[n];const auto kind=node.identity?0:lh_full_kind(node.full);has_lh|=kind!=0;lh_kinds.push_back(kind);
    kinds.push_back(!node.identity&&node.full=="tanh");weights.push_back(w.weight);biases.push_back(w.bias);
    norm_weights.push_back(kind&&(kind-1)%3? w.extra.at("lh_norm_weight"):at::ones_like(w.bias));
    norm_biases.push_back(kind&&(kind-1)%3==2? w.extra.at("lh_norm_bias"):at::zeros_like(w.bias));}
  const auto remaining=l.workspace_bytes-static_cast<int64_t>(estimate);
  emission=std::make_unique<PackedEmission>(profile,device,samples,l.arrivals,l.outputs,l.emission_chunk_rows,remaining);
  auto full_budget=remaining-emission->reserved_bytes();
  if(std::any_of(profile.graph.nodes.begin(),profile.graph.nodes.end(),[](const Node& n){return !n.identity&&n.full=="swiglu";})) {
    swiglu_full=std::make_unique<PackedSwiGluFull>(profile,device,l.queue,l.full_chunk_rows,full_budget);
    full_budget-=swiglu_full->reserved_bytes();
  }
  if(has_lh)lh_full=std::make_unique<PackedLhFull>(lh_kinds,at::stack(norm_weights),at::stack(norm_biases),device,l.queue,l.full_chunk_rows,full_budget);
  full_budget-=lh_full?lh_full->reserved_bytes():0;
  const auto minimum_event=PackedEventAttention::minimum_bytes(profile,q,l);
  if(std::any_of(profile.graph.nodes.begin(),profile.graph.nodes.end(),[](const Node& n){return !n.identity&&is_fiber_attention_profile(n.memory);})) {
    // Leave the minimum remaining Full work before growing an attention chunk.
    // Otherwise a legal smaller query chunk could be rejected during Full setup.
    const auto minimum_full=PackedFull::minimum_bytes(kinds,width)+minimum_event;
    if(minimum_full>=full_budget)throw std::invalid_argument("fiber cache leaves no Full workspace");
    const auto attention_budget=static_cast<int64_t>(full_budget-minimum_full);
    attention=std::make_unique<PackedFiberAttention>(profile,q,device,l,attention_budget);
    full_budget-=attention->reserved_bytes();
  }
  if(minimum_event>0) {
    const auto minimum_full=PackedFull::minimum_bytes(kinds,width);
    if(minimum_full>=full_budget)throw std::invalid_argument("event cache leaves no Full workspace");
    event_attention=std::make_unique<PackedEventAttention>(profile,q,device,l,static_cast<int64_t>(full_budget-minimum_full));
    full_budget-=event_attention->reserved_bytes();
  }
  full=std::make_unique<PackedFull>(kinds,at::stack(weights),at::stack(biases),device,l.full_chunk_rows,full_budget);
  construct();
  // Only input-seal/ledger metadata belongs on the host between windows.
  // The authoritative state, history and pending payloads are device owners.
  boundary.states.clear();boundary.history.clear();boundary.pending.clear();
}
void ContentFlow::Impl::construct() {
  const auto& g=profile.graph;const auto opts=state.values.options();
  DeviceReady planner(profile.owners,g.regions.size(),profile.wires,boundary.batch_size,device,limits.prefill,profile.causal_regions);
  program=std::make_unique<CannProgram>(device);auto& p=*program;
  auto zeros=at::zeros({limits.queue},error.options()),out_zeros=at::zeros({limits.outputs},error.options());
  if(limits.diagnostics) {
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
  auto content=append_content(p,profile,ready,error,limits.vectorized_aggregate);
  FiberStage attended;
  if(attention)attended=attention->propose(p,profile,ready,content,state,error);
  EventAttentionStage event_attended;auto proposals=attended.values;
  if(event_attention){event_attended=event_attention->propose(p,ready,content,error,proposals);proposals=event_attended.values;}
  append_read(p,profile,ready,content,state,coefficients,error,limits.max_repeat_ticks,limits.vectorized_read,proposals);
  auto selection=selector->append_stage(p,ready,content.scores,history,error);
  auto update=append_content_state(p,profile,ready,content,selection,state,coefficients,stages,event_count,error,limits,proposals);
  auto actions=full->append_stage(p,update.actions,update.comparison,error);
  if(lh_full)actions=lh_full->append_stage(p,actions,update.comparison,error,full->chunks());
  if(swiglu_full)actions=swiglu_full->append_stage(p,actions,content.content,update.comparison,error,full->chunks());
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
    auto emission_proposal=emission_trace->propose(p,emitted.meta,emitted.values,emitted.count,error);
    messages->commit_stage(p,message_proposal);
    events->commit(p,event_proposal,error);fibers->commit(p,fiber_proposal,error);contributions->commit(p,contribution_proposal,error);
    full_trace->commit(p,full_proposal,error);
    emission_trace->commit(p,emission_proposal,error);
  }
  pending->commit_stage(p,pending_proposal);outputs->commit_stage(p,output_proposal);
  selector->append_commit(p,history,selection,error);commit_content_state(p,state,update,error);
  if(attention)attention->commit(p,attended,selection,error);
  if(event_attention)event_attention->commit(p,event_attended,selection,error);
  p.add(stages,one);p.branch(index,{head});p.mark(exhausted);p.copy(error,budget_error);p.mark(end);p.finish();
  if(p.workspace_bytes()>limits.workspace_bytes)throw std::invalid_argument("CANN numerical workspace exceeds content budget");
}
ContentFlow::ContentFlow(Graph g,Model m,const Continuation& q,at::Device d,ContentLimits l)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(m),q,d,l)) {}
ContentFlow::~ContentFlow()=default;
Result ContentFlow::advance(const std::vector<External>& input,Index until) {
  advance_device(input,until);
  try {return result();}catch(...){impl_->failed=true;throw;}
}
ContentWindow ContentFlow::advance_device(const std::vector<External>& input,Index until) {
  auto& s=*impl_;
  if(s.failed)throw std::logic_error("content flow failed; restore an earlier complete cut into a new owner");
  if(at::GradMode::is_enabled())throw std::invalid_argument("content flow has no autograd contract");
  auto validated=validate_external(s.profile.graph,s.profile.model,s.boundary,input,until,until);
  if(validated.atoms.size()>size_t(s.limits.queue))throw std::invalid_argument("external input buffer capacity exceeded");
  try {
    upload_atoms(validated.atoms,s.external);s.outputs->atoms().valid.zero_();s.outputs->stats().zero_();
    if(s.limits.diagnostics) {
      s.messages->atoms().valid.zero_();s.messages->stats().zero_();s.events->count.zero_();s.fibers->count.zero_();s.contributions->count.zero_();
      s.full_trace->count.zero_();s.emission_trace->count.zero_();
    }
    if(s.attention)s.attention->reset_window();
    if(s.event_attention)s.event_attention->reset_window();
    s.event_count.zero_();s.full->chunks().zero_();s.emission->chunks().zero_();s.stages.zero_();s.stop.fill_(until);portable_torch::synchronize(s.device);s.program->run();
    const auto error=s.error.cpu().item<int>();
    if(error)throw std::runtime_error("content flow device refusal code="+std::to_string(error));
    s.boundary.cut=until;for(const auto& [owner,last]:validated.ledger_updates)s.boundary.ledger[owner]=last;
    return {s.outputs->atoms(),s.outputs->stats(),s.pending->stats(),s.stages,s.event_count,s.full->chunks(),s.emission->chunks()};
  } catch(...) {s.failed=true;throw;}
}
Continuation ContentFlow::snapshot() const {
  if(impl_->failed)throw std::logic_error("content flow failed; complete-cut snapshot unavailable");
  return impl_->export_continuation();
}
Result ContentFlow::result() const {
  if(impl_->failed)throw std::logic_error("content flow failed; result unavailable");
  return impl_->export_result();
}
} // namespace tide::device_online
