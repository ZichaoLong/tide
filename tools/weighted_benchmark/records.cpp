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
Tensor loss(const Result& r,bool allow_empty){
  std::vector<Tensor> parts;for(const auto& o:r.outputs){auto x=o.value.scalar_type()==at::kHalf?o.value.to(at::kFloat):o.value;
    parts.push_back((x[0]*x.slice(0,1)).square().sum()*.001);}
  if(parts.empty()){
    if(allow_empty)return at::zeros({});
    throw std::runtime_error("benchmark produced no outputs");
  }
  return at::stack(parts).sum();
}
Json work(const Result& r){
  Index full=0;for(const auto& e:r.trace)full+=e.active;
  return { {"candidate_updates",r.trace.size()},{"full_calls",full},{"messages",r.messages.size()},
    {"outputs",r.outputs.size()},{"pending",r.continuation.pending.size()},{"actual_stats",r.stats} };
}
void compare_tensors(const std::map<std::string,Tensor>& actual,const std::map<std::string,Tensor>& expected,
                     at::ScalarType dtype,const std::string& label,bool exact){
  if(actual.size()!=expected.size())throw std::runtime_error(label+" owner count differs");
  for(const auto& [name,value]:expected){
    const auto found=actual.find(name);
    if(found==actual.end())throw std::runtime_error(label+" owner missing: "+name);
    const auto& got=found->second;
    if(got.defined()!=value.defined())throw std::runtime_error(label+" None/zero differs: "+name);
    if(value.defined()&&(got.sizes()!=value.sizes()||got.scalar_type()!=value.scalar_type()||
        !at::allclose(got,value,exact?0.:dtype==at::kDouble?1e-8:dtype==at::kHalf?2e-2:1e-5,exact?0.:dtype==at::kDouble?1e-10:dtype==at::kHalf?1e-3:1e-6)))
      throw std::runtime_error(label+" differs: "+name);
  }
}
}
