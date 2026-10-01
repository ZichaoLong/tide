#include "bounded.h"
#include <tide/lh_full.h>
#include <limits>

namespace accelerator_scale::bounded {
void Program::select(Index region,Index time,std::map<Index,Event>& events,History& history) const {
  const auto& members=members_[region];const auto& spec=f_.graph.regions[region];
  const auto d=history.selected.device();const auto opts=f_.model.nodes[members[0]].bias.options().device(d);
  bool any=false;std::vector<Tensor> scores,presences;auto deps=empty_dependencies(d);
  for(auto node:members) {
    auto it=events.find(node);
    if(it==events.end()) {
      scores.push_back(at::zeros({limits_.batch},opts.dtype(at::kFloat)));
      presences.push_back(at::zeros({limits_.batch},opts.dtype(at::kBool)));
    } else {
      any=true;auto& e=it->second;auto p=move(e.candidate,d);
      scores.push_back(move(e.descriptor.data,d));presences.push_back(p);
      deps=deps|(move(e.descriptor.dependencies,d)&p.unsqueeze(1));
    }
  }
  if(!any)return; // compile-time reachability, never a tensor decision
  auto score=at::stack(scores,1),present=at::stack(presences,1);Tensor active;
  {
    at::NoGradGuard guard;
    auto order=at::argsort(score,true,-1,true);
    if(spec.selector=="lh-count-affect-v1")order=order.gather(1,at::argsort(history.affected.gather(1,order),true,-1,true));
    if(spec.count_priority)order=order.gather(1,at::argsort(history.selected.gather(1,order),true,-1,false));
    order=order.gather(1,at::argsort(present.to(at::kLong).gather(1,order),true,-1,true));
    auto rank=at::argsort(order,-1,false);
    active=(rank<spec.budget)&present;
    history.selected=history.selected+active.to(at::kLong);
    if(spec.selector=="lh-count-affect-v1")history.affected=history.affected+present.to(at::kLong);
    auto seen=present.any(1);history.last=at::where(seen,at::full_like(history.last,time),history.last);
    history.seen=history.seen|seen;
  }
  auto masked=score.masked_fill(~present,-std::numeric_limits<float>::infinity());
  masked=at::where(present.any(1).unsqueeze(1),masked,at::zeros_like(masked));
  auto controls=masked_rows(at::softmax(masked,1).to(at::kFloat),present);
  for(size_t i=0;i<members.size();++i) {
    auto it=events.find(members[i]);if(it==events.end())continue;
    auto& e=it->second;const auto dest=e.content.data.device();
    e.active=move(active.select(1,i),dest);e.control={move(controls.select(1,i),dest),move(deps,dest)};
    e.history=history;
  }
}
void Program::finish(Event& e) const {
  const auto& w=f_.model.nodes[e.node];const auto& node=f_.graph.nodes[e.node];const auto d=w.bias.device();
  e.next=e.proposal;
  if(node.clear) {
    // A selected zero state remains connected; empty KV retains its structural
    // VJP history even though no cache row remains visible.
    e.next.value.data=at::where(e.active.unsqueeze(1),e.proposal.value.data*0,e.proposal.value.data);
    if(e.next.valid.defined())e.next.valid=e.next.valid & ~e.active.unsqueeze(1);
  }
  // Mask absent rows before nonlinear arithmetic; masking overflowed results
  // afterwards can still poison a connected VJP with NaNs.
  auto fresh=lh_full_fresh(w,masked_rows(e.proposal.value.data,e.active));auto dep=e.proposal.value.dependencies;
  auto norm=w.extra.find("lh_norm_weight");if(norm!=w.extra.end())dep=dep|dependency(norm->second,d);
  e.fresh={fresh,dep};Tensor projected;
  auto weight=w.extra.find("row_emit_weight");
  if(weight!=w.extra.end())projected=at::linear(fresh,weight->second);
  const auto first=f_.graph.outgoing_ports.offsets[e.node];
  for(Index j=first;j<f_.graph.outgoing_ports.offsets[e.node+1];++j) {
    const auto slot=j-first;const auto binding=f_.graph.outgoing_ports.bindings[j];
    if(binding.kind==1 && e.time%node.emit_period!=node.emit_phases[slot])continue;
    auto row=binding.kind==1?edge_rows_[binding.id]:-1;
    e.emit_slots.push_back(slot);
    if(row<0)e.emitted.push_back(e.fresh);
    else e.emitted.push_back({projected.slice(1,row*f_.model.width(),(row+1)*f_.model.width()),dep|dependency(weight->second,d)});
  }
}
}  // namespace accelerator_scale::bounded
