#include "bounded.h"
#include <algorithm>

namespace accelerator_scale::bounded {
namespace {
bool yes(const Tensor& value,Index row) { return value[row].item<bool>(); }
Index integer(const Tensor& value,Index row) { return value[row].item<Index>(); }
tide::State state(const State& s,Index b) {
  tide::State result{s.value.data[b].detach(),integer(s.last,b),integer(s.observations,b)};
  if(s.valid.defined()) {
    auto index=at::nonzero(s.valid[b]).reshape({-1});
    result.slots={{"key",s.key[b].index_select(0,index).detach()},
      {"value",s.cache_value[b].index_select(0,index).detach()},{"log_bias",s.bias[b].index_select(0,index).detach()}};
  }
  return result;
}
tide::History history(const Program& p,const History& h,Index r,Index b) {
  tide::History result;result.last_time=integer(h.last,b);result.node_maps["selected"]={};
  const bool lh=p.graph().regions[r].selector=="lh-count-affect-v1";
  if(lh)result.node_maps["affected"]={};
  for(size_t i=0;i<p.members()[r].size();++i) {
    auto selected=h.selected[b][i].item<Index>(),affected=h.affected[b][i].item<Index>();
    if(selected)result.node_maps["selected"][p.members()[r][i]]=selected;
    if(lh && affected)result.node_maps["affected"][p.members()[r][i]]=affected;
  }
  return result;
}
Atom atom(const Message& m,Index b) { return {b,m.node,m.time,m.kind,m.source,m.position,m.value.data[b].detach()}; }
std::vector<Atom> atoms(const std::vector<Message>& messages,Index batch) {
  std::vector<Atom> result;
  for(const auto& m:messages)for(Index b=0;b<batch;++b)if(yes(m.present,b))result.push_back(atom(m,b));
  return result;
}
}  // namespace
Result export_result(const Program& p,const Window& w) {
  Result out;auto& q=out.continuation;const auto batch=p.limits().batch;
  q.identity=p.graph().identity;q.batch_size=batch;
  const auto period=p.period();
  q.cut=p.limits().tokens*period;
  for(Index b=0;b<batch;++b)q.ledger[{b,0}]={p.limits().tokens-1,(p.limits().tokens-1)*period};
  for(size_t n=0;n<w.states.size();++n)for(Index b=0;b<batch;++b)
    if(yes(w.states[n].seen,b))q.states[{b,n}]=state(w.states[n],b);
  for(size_t r=0;r<w.histories.size();++r)for(Index b=0;b<batch;++b)
    if(yes(w.histories[r].seen,b))q.history[{b,r}]=history(p,w.histories[r],r,b);
  q.pending=atoms(w.pending,batch);
  std::sort(q.pending.begin(),q.pending.end(),[](const auto& a,const auto& b){return a.key()<b.key();});
  out.messages=atoms(w.messages,batch);
  std::sort(out.messages.begin(),out.messages.end(),[&](const auto& a,const auto& b){
    return std::tie(a.position,a.batch,p.graph().edges[a.source].source,a.source)<std::tie(b.position,b.batch,p.graph().edges[b.source].source,b.source);});
  for(const auto& event:w.events)for(Index b=0;b<batch;++b)if(yes(event.candidate,b)) {
    tide::Event e;e.node=event.node;e.time=event.time;e.batch=b;e.active=yes(event.active,b);
    const auto region=p.graph().nodes[e.node].region;
    e.old=state(event.old,b);e.proposed_state=state(event.proposal,b);e.comparison_state=e.proposed_state;e.next_state=state(event.next,b);
    e.content=event.content.data[b].detach();e.proposal=e.proposed_state.value;e.comparison=e.comparison_state.value;e.next=e.next_state.value;
    e.descriptor=event.descriptor.data[b].detach();e.control=event.control.data[b].detach();e.history=history(p,event.history,region,b);
    for(size_t i=0;i<event.fiber.size();++i)if(yes(event.fiber[i].present,b)) {
      const auto& m=event.fiber[i];auto a=atom(m,b);e.fiber.push_back(a);
      auto visible=a;const auto& graph=p.graph();
      if(a.kind==1 && !graph.origins.empty() && graph.origin_index[a.source]!=-1) {
        const auto& origin=graph.origins[graph.origin_index[a.source]];
        if(a.position%origin.stride)throw std::logic_error("bounded input origin clock mismatch");
        visible.kind=0;visible.source=origin.port;visible.position/=origin.stride;
      }
      e.sources.push_back({m.slot,visible,at::ones({},a.value.options())});
      e.contributions.push_back({m.slot,event.contributions[i].data[b].detach()});
    }
    std::sort(e.contributions.begin(),e.contributions.end(),[](const auto& a,const auto& b){return a.slot<b.slot;});
    if(!p.graph().origins.empty())std::sort(e.sources.begin(),e.sources.end(),[](const auto& a,const auto& b){return a.atom.key()<b.atom.key();});
    if(e.active) {
      e.full=event.fresh.data[b].detach();
      for(size_t i=0;i<event.emitted.size();++i) {
        e.emitted.push_back({event.emit_slots[i],event.emitted[i].data[b].detach()});
        const auto binding=p.graph().outgoing_ports.bindings[p.graph().outgoing_ports.offsets[e.node]+event.emit_slots[i]];
        if(binding.kind==0)out.outputs.push_back({b,e.time,binding.id,event.emitted[i].data[b].detach()});
      }
    }
    out.trace.push_back(std::move(e));
  }
  std::sort(out.trace.begin(),out.trace.end(),[](const auto& a,const auto& b){return std::tie(a.time,a.batch,a.node)<std::tie(b.time,b.batch,b.node);});
  std::sort(out.outputs.begin(),out.outputs.end(),[](const auto& a,const auto& b){return std::tie(a.time,a.batch,a.port)<std::tie(b.time,b.batch,b.port);});
  return out;
}
}  // namespace accelerator_scale::bounded
