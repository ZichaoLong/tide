#include "benchmark.h"
#include "return_packing.h"
#include "../../cpp/bench/comparison.h"
#include "tide/communication.h"
namespace weighted_benchmark {
namespace {
void scalar(Tensor& x){if(x.defined())x=communication_copy(x.detach(),at::Device(at::kCPU),x.scalar_type()).clone();}
template<class Copy> void state(State& s,Copy copy){copy(s.value);for(auto& [_,x]:s.slots)copy(x);}
template<class Copy> void history(History& h,Copy copy){for(auto& [_,x]:h.tensors)copy(x);}
template<class Copy> void continuation(Continuation& q,Copy copy){
  for(auto& [_,s]:q.states)state(s,copy);for(auto& [_,h]:q.history)history(h,copy);
  for(auto& a:q.pending)copy(a.value);
}
template<class Copy> void result_fields(Result& r,Copy copy){
  continuation(r.continuation,copy);
  for(auto& x:r.outputs)copy(x.value);for(auto& a:r.messages)copy(a.value);
  for(auto& e:r.trace){for(auto x:{&e.content,&e.proposal,&e.descriptor,&e.control,&e.comparison,&e.next,&e.full})copy(*x);
    for(auto& [_,x]:e.descriptor_fields)copy(x);for(auto& [_,x]:e.state_prepared)copy(x);
    for(auto x:{&e.old,&e.proposed_state,&e.comparison_state,&e.next_state})state(*x,copy);history(e.history,copy);
    for(auto& a:e.fiber)copy(a.value);for(auto& x:e.emitted)copy(x.value);for(auto& x:e.contributions)copy(x.value);
    for(auto& x:e.sources){copy(x.atom.value);copy(x.scale);}
    if(e.source_batch){auto b=std::make_shared<SourceBatch>(*e.source_batch);copy(b->values);e.source_batch=std::move(b);}}
}
Json activity(const Result& r){
  struct Sample {Index candidates=0,full=0,received=0,messages=0,outputs=0,pending=0,owners=0;std::map<Index,Index> cache;};
  std::vector<Sample> rows(r.continuation.batch_size);std::map<Index,Index> total;
  for(const auto& e:r.trace){auto& row=rows.at(e.batch);++row.candidates;row.full+=e.active;row.received+=e.fiber.size();}
  for(const auto& a:r.messages)++rows.at(a.batch).messages;
  for(const auto& a:r.continuation.pending)++rows.at(a.batch).pending;
  for(const auto& o:r.outputs)++rows.at(o.batch).outputs;
  for(const auto& [owner,state]:r.continuation.states){auto found=state.slots.find("times");if(found==state.slots.end())continue;
    const auto& times=found->second;
    if(times.scalar_type()!=at::kLong||times.dim()!=1)
      throw std::invalid_argument("activity observation requires int64 KV timestamps");
    auto& row=rows.at(owner.first);++row.owners;++row.cache[times.numel()];++total[times.numel()];}
  auto histogram=[](const std::map<Index,Index>& counts){Json out=Json::object();for(const auto& [length,n]:counts)out[std::to_string(length)]=n;return out;};
  Json samples=Json::array();
  for(size_t b=0;b<rows.size();++b){const auto& row=rows[b];samples.push_back({{"sample",b},{"candidates",row.candidates},
    {"full",row.full},{"received_atoms",row.received},{"messages",row.messages},{"outputs",row.outputs},{"pending",row.pending},
    {"stored_kv_owners",row.owners},{"raw_kv_cache_lengths",histogram(row.cache)}});}
  return {{"scope","actual record metadata; raw stored KV lengths, not visible query buckets; no timestamp value copies"},
    {"samples",samples},{"raw_kv_cache_lengths",histogram(total)}};
}
}
Continuation detached(Continuation q){
  continuation(q,scalar);return q;
}
Result detached(Result r,bool packed){
  if(!packed){result_fields(r,scalar);return r;}
  std::vector<Tensor*> fields;result_fields(r,[&](Tensor& x){fields.push_back(&x);});
  return_copy::tensors(fields,packed);
  return r;
}
Result projected(Result r,const std::string& mode,bool packed){
  if(mode=="complete")return detached(std::move(r),packed);
  Result out;out.continuation.identity=r.continuation.identity;out.continuation.cut=r.continuation.cut;
  out.continuation.batch_size=r.continuation.batch_size;
  if(mode=="outputs"){
    out.outputs=std::move(r.outputs);std::vector<Tensor*> fields;
    for(auto& o:out.outputs)fields.push_back(&o.value);return_copy::tensors(fields,packed);
  }else if(mode!="metrics")throw std::invalid_argument("unknown return mode");
  return out;
}
Tensor loss(const Result& r,bool allow_empty,double scale){
  std::vector<Tensor> parts;for(const auto& o:r.outputs){auto x=o.value.scalar_type()==at::kHalf?o.value.to(at::kFloat):o.value;
    parts.push_back((x[0]*x.slice(0,1)).square().sum()*scale);}
  if(parts.empty()){
    if(allow_empty)return at::zeros({});
    throw std::runtime_error("benchmark produced no outputs");
  }
  return at::stack(parts).sum();
}
Json work(const Result& r){
  Index full=0;for(const auto& e:r.trace)full+=e.active;
  return { {"candidate_updates",r.trace.size()},{"full_calls",full},{"messages",r.messages.size()},
    {"outputs",r.outputs.size()},{"pending",r.continuation.pending.size()},{"actual_stats",r.stats},
    {"boundary_activity",activity(r)} };
}
void compare_tensors(const std::map<std::string,Tensor>& actual,const std::map<std::string,Tensor>& expected,
                     at::ScalarType dtype,const std::string& label,bool exact){
  const auto report=tide_bench::collect_tensors(actual,expected,dtype,label,
    exact?0.:dtype==at::kDouble?1e-8:dtype==at::kHalf?2e-2:1e-5,
    exact?0.:dtype==at::kDouble?1e-10:dtype==at::kHalf?1e-3:1e-6);
  if(!report.passed())throw tide_bench::ComparisonFailure(report.differences.front());
}
}
