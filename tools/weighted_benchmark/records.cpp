#include "benchmark.h"
namespace weighted_benchmark {
namespace {
void tensor(Tensor& x){if(x.defined())x=x.detach().cpu().clone();}
void state(State& s){tensor(s.value);for(auto& [_,x]:s.slots)tensor(x);}
void history(History& h){for(auto& [_,x]:h.tensors)tensor(x);}
}
Continuation detached(Continuation q){
  for(auto& [_,s]:q.states)state(s);for(auto& [_,h]:q.history)history(h);for(auto& a:q.pending)tensor(a.value);return q;
}
Result detached(Result r){
  r.continuation=detached(r.continuation);for(auto& x:r.outputs)tensor(x.value);for(auto& a:r.messages)tensor(a.value);
  for(auto& e:r.trace){for(auto x:{&e.content,&e.proposal,&e.descriptor,&e.control,&e.comparison,&e.next,&e.full})tensor(*x);
    for(auto& [_,x]:e.descriptor_fields)tensor(x);for(auto& [_,x]:e.state_prepared)tensor(x);
    for(auto x:{&e.old,&e.proposed_state,&e.comparison_state,&e.next_state})state(*x);history(e.history);
    for(auto& a:e.fiber)tensor(a.value);for(auto& x:e.emitted)tensor(x.value);for(auto& x:e.contributions)tensor(x.value);
    for(auto& x:e.sources){tensor(x.atom.value);tensor(x.scale);}
    if(e.source_batch){auto b=std::make_shared<SourceBatch>(*e.source_batch);tensor(b->values);e.source_batch=std::move(b);}}
  return r;
}
Tensor loss(const Result& r){
  std::vector<Tensor> parts;for(const auto& o:r.outputs)parts.push_back((o.value[0]*o.value.slice(0,1)).square().sum()*.001);
  if(parts.empty())throw std::runtime_error("benchmark produced no outputs");return at::stack(parts).sum();
}
Json work(const Result& r){
  Index full=0;for(const auto& e:r.trace)full+=e.active;
  return { {"candidate_updates",r.trace.size()},{"full_calls",full},{"messages",r.messages.size()},
    {"outputs",r.outputs.size()},{"pending",r.continuation.pending.size()},{"actual_stats",r.stats} };
}
}
