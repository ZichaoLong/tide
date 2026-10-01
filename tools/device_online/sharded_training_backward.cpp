#include "sharded_training_internal.h"
#include <stdexcept>
namespace tide::training_detail {
using namespace device_online;
namespace {
void root_pair(const Tensor& value,const Tensor& on,const Tensor& shape,const Tensor& present) {
  if(value.defined()!=on.defined())throw std::invalid_argument("cotangent value/connection pair is incomplete");
  if(!value.defined())return;
  for(const auto& pair:std::vector<std::pair<Tensor,Tensor>>{{value,shape},{on,present}}) {
    const auto& x=pair.first;const auto& ref=pair.second;
    if(x.device()!=ref.device()||x.scalar_type()!=(ref.scalar_type()==at::kBool?at::kBool:at::kFloat)
        ||x.sizes()!=ref.sizes()||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid resident cotangent layout");
  }
  if((on&present.logical_not()).any().item<bool>())throw std::invalid_argument("cotangent connects absent output/state/message");
}
void cache_roots(const std::vector<ResidentCacheCotangents>& roots,const std::vector<ResidentCacheWindow>& cache) {
  if(!roots.empty()&&roots.size()!=cache.size())throw std::invalid_argument("resident cache root group count mismatch");
  for(size_t i=0;i<roots.size();++i) {
    const auto& r=roots[i];const auto& c=cache[i];
    root_pair(r.key,r.key_connected,c.key,c.present);root_pair(r.value,r.value_connected,c.value,c.present);
    if(c.log_bias.defined())root_pair(r.log_bias,r.log_bias_connected,c.log_bias,c.present);
    else if(r.log_bias.defined()||r.log_bias_connected.defined())throw std::invalid_argument("event cache has no log-bias root");
  }
}
std::vector<std::vector<CacheCotangents>> cache_inputs(const ResidentCotangents& root,size_t owners) {
  std::vector<std::vector<CacheCotangents>> out(owners);
  for(size_t i=0;i<root.states.size();++i)for(const auto& c:root.states[i].cache)
    out[i].push_back({c.key,c.value,c.key_connected,c.value_connected,c.log_bias,c.log_bias_connected});
  return out;
}
GraphCotangents roots(const ResidentCotangents& r,const ReverseTape& t,const std::vector<ResidentStateWindow>& states) {
  auto f=t.outputs.values.options().dtype(at::kFloat),b=t.outputs.valid.options();
  GraphCotangents out{r.outputs.defined()?r.outputs:at::zeros(t.outputs.values.sizes(),f),
    r.outputs_connected.defined()?r.outputs_connected:at::zeros_like(t.outputs.valid),
    r.pending.defined()?r.pending:at::zeros(t.pending.values.sizes(),f),
    r.pending_connected.defined()?r.pending_connected:at::zeros_like(t.pending.valid),
    at::zeros({t.state.samples,int64_t(t.graph->nodes.size()),t.full.width},f),
    at::zeros({t.state.samples,int64_t(t.graph->nodes.size())},b)};
  // Fixed owner-level transfers and scatters, never an event/message host loop.
  // Only cotangents are assembled; forward state/cache values stay on owners.
  // CANN's Bool ScatterUpdate uses AiCPU. Integer flags preserve exactly the
  // same connection mask and use the vector-core scatter/cast path.
  auto flags=at::zeros(out.final_connected.sizes(),f.dtype(at::kInt));
  for(size_t i=0;i<r.states.size();++i)if(r.states[i].final.defined()) {
    auto ids=at::tensor(states[i].nodes,at::kLong).to(out.final.device());
    out.final.index_copy_(1,ids,r.states[i].final.to(out.final.device()));
    flags.index_copy_(1,ids,r.states[i].final_connected.to(flags.options()));
  }
  out.final_connected=flags.to(at::kBool);
  return out;
}
ResidentParameterGradient parameter_view(const ParameterVjp& p) {
  ResidentParameterGradient out;
  for(const auto& owner:p.owners){out.names.push_back(owner.canonical);out.aliases.push_back(owner.aliases);}
  out.offsets=p.offsets;out.values=p.values;out.connected=p.connected;return out;
}
}
ResidentGradients ShardedTrainingOwner::backward(const std::vector<ResidentCotangents>& roots) {
  auto& s=*impl_;s.check();
  if(s.gradients_ready||s.saved.empty())throw std::logic_error("backward requires unconsumed retained windows");
  try{return s.reverse(roots);}catch(const std::invalid_argument&){throw;}catch(...){s.failed=true;throw;}
}
ResidentGradients ShardedTrainingOwner::Impl::reverse(const std::vector<ResidentCotangents>& input) {
  auto& s=*this;
  if(input.size()!=s.saved.size())throw std::invalid_argument("provide roots for all retained windows in forward order");
  const Index per=s.limits.backward_bytes/Index(s.saved.size());
  for(size_t i=0;i<input.size();++i) {
    const auto& r=input[i];const auto& w=s.saved[i];const auto& t=w.tape.tape.coordinator;
    if(r.token.session!=w.token.session||r.token.index!=w.token.index||r.token.generation!=w.token.generation)
      throw std::invalid_argument("stale, foreign or out-of-order resident token");
    if(r.final.defined()||r.final_connected.defined()||!r.cache.empty())
      throw std::invalid_argument("sharded sessions require owner-local state/cache roots");
    if(!r.states.empty()&&r.states.size()!=w.states.size())throw std::invalid_argument("state root owner count mismatch");
    long double own=5.L*(t.outputs.values.numel()+static_cast<long double>(t.pending.values.numel()))
      +64.L*(t.fiber_values.size(0)+static_cast<long double>(t.pending.valid.numel()))+4096;
    for(const auto& state:w.states) {
      own+=24.L*state.values.numel()+8.L*state.present.numel();
      for(const auto& c:state.cache)own+=16.L*c.key.numel()+40.L*c.lengths.numel()+(c.log_bias.defined()?8.L*c.log_bias.numel():0.L);
    }
    if(per<16||own>per/8)throw std::invalid_argument("sharded backward root/export budget exceeded");
    root_pair(r.outputs,r.outputs_connected,t.outputs.values,t.outputs.valid);
    root_pair(r.pending,r.pending_connected,t.pending.values,t.pending.valid);
    for(size_t j=0;j<r.states.size();++j) {
      root_pair(r.states[j].final,r.states[j].final_connected,w.states[j].values,w.states[j].present);
      cache_roots(r.states[j].cache,w.states[j].cache);
    }
  }
  auto error=at::zeros({1},at::TensorOptions().device(s.device).dtype(at::kInt));
  CannSequence sequence(s.device,s.saved.size(),s.limits.program_workspace_bytes);
  std::vector<ShardedGraphVjp> gradients(s.saved.size());
  for(size_t i=s.saved.size();i>0;) {--i;auto& p=sequence.append();const auto& t=s.saved[i].tape.tape;
    auto cot=roots(input[i],t.coordinator,s.saved[i].states);
    if(i+1<s.saved.size())cot=append_window_bridge(p,t.coordinator,cot,s.saved[i+1].tape.tape.coordinator,gradients[i+1].coordinator,error,per/8);
    gradients[i]=append_sharded_graph_vjp(p,t,cot,error,s.limits.reverse_chunk_rows,per/2,s.limits.program_workspace_bytes,
      i+1<s.saved.size()?gradients[i+1].state:nullptr,cache_inputs(input[i],t.states.size()));
  }
  ShardedParameterReduce reduction(sharded_parameter_sources(s.graph,s.registry,gradients,s.limits.backward_bytes/16),
    s.placement.devices,error,s.limits.backward_bytes/8,s.limits.program_workspace_bytes);
  const auto& parts=reduction.gradients();
  if(parts.size()!=s.layout.size())throw std::logic_error("canonical owner count changed after forward");
  for(size_t d=0;d<parts.size();++d) {
    if(parts[d].offsets!=s.layout[d].offsets||parts[d].owners.size()!=s.layout[d].owners.size())throw std::logic_error("canonical layout changed after forward");
    for(size_t i=0;i<parts[d].owners.size();++i)if(parts[d].owners[i].canonical!=s.layout[d].owners[i].canonical)
      throw std::logic_error("canonical parameter order changed after forward");
  }
  sequence.finish();reduction.finish();
  try {
    run_sharded_graph_vjp(sequence,gradients);reduction.run();
    const auto code=reduction.errors()[0].cpu().item<int>();
    if(code)throw std::runtime_error("resident sharded backward refusal code="+std::to_string(code));
    ResidentGradients out;for(const auto& p:parts)out.parameter_shards.push_back(parameter_view(p));
    const auto& first=gradients.front().coordinator;const auto local=gradients.front().state->gradients();
    for(size_t d=0;d<s.saved.front().states.size();++d) {
      const auto& state=s.saved.front().states[d];const auto& present=s.initial_present[d];auto ids=at::tensor(state.nodes,at::kLong).to(s.device);
      auto values=first.initial.index_select(1,ids).to(present.device()),connected=first.initial_connected.index_select(1,ids).to(present.device());
      ResidentStateGradient shard{state.nodes,at::where(present.unsqueeze(-1),values,at::zeros_like(values)),connected&present};
      const auto& tape=s.saved.front().tape.tape.states[d];
      for(size_t i=0;i<local[d].cache.size();++i) {
        const auto& a=i<tape.attention.size()?tape.attention[i]:tape.fiber[i-tape.attention.size()].cache;const auto& g=local[d].cache[i];
        auto local_ids=at::tensor(a.nodes,at::kLong).to(present.device()),on=present.index_select(1,local_ids).reshape({-1}),mask=on.reshape({-1,1,1,1});
        shard.cache.push_back({state.cache[i].nodes,at::where(mask,g.key,at::zeros_like(g.key)),at::where(mask,g.value,at::zeros_like(g.value)),
          g.lengths,g.key_connected&on,g.value_connected&on,g.bias.defined()?at::where(on.unsqueeze(1),g.bias,at::zeros_like(g.bias)):Tensor{},
          g.bias_connected.defined()?g.bias_connected&on:Tensor{}});
      }
      out.initial_shards.push_back(std::move(shard));
    }
    for(size_t i=0;i<s.saved.size();++i) {
      const auto& t=s.saved[i].tape.tape.coordinator;const auto& g=gradients[i].coordinator;const auto n=g.links.fibers+g.links.pending;
      merge_reverse_statistics(out.statistics,g.statistics);
      for(const auto& state:gradients[i].state->gradients())merge_reverse_statistics(out.statistics,state.statistics);
      auto valid=g.links.valid.narrow(0,0,n)&g.links.messages.narrow(0,0,n).select(1,1).lt(0);
      out.boundaries.push_back({s.saved[i].token,at::cat({t.fiber_meta,t.pending.coordinates}),g.messages.narrow(0,0,n),valid,g.message_connected.narrow(0,0,n)&valid});
    }
    s.gradient=parts;reduction.close();sequence.close();close_sharded_graph_vjp(gradients);
    out.statistics["retained_projection_bytes"]=s.projection_bytes;
    out.statistics["retained_window_bytes"]=s.bytes_per_window;
    out.statistics["retained_windows"]=s.saved.size();
    out.statistics["retained_bytes"]=s.saved_bytes;
    s.gradients_ready=true;s.saved.clear();s.projection_snapshot={};s.saved_bytes=0;return out;
  }catch(...){s.failed=true;throw;}
}
} // namespace tide::training_detail
