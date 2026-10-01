#include "training_internal.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <stdexcept>

namespace tide {
using namespace device_online;
namespace {
void root_pair(const Tensor& value,const Tensor& on,const Tensor& shape,const Tensor& present) {
  if(value.defined()!=on.defined())throw std::invalid_argument("cotangent value/connection pair is incomplete");
  if(!value.defined())return;
  for(const auto& pair:std::vector<std::pair<Tensor,Tensor>>{{value,shape},{on,present}}) {
    const auto& x=pair.first;const auto& ref=pair.second;
    const auto dtype=ref.scalar_type()==at::kBool?at::kBool:at::kFloat;
    if(x.device()!=ref.device()||x.scalar_type()!=dtype||x.sizes()!=ref.sizes()
        ||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid resident cotangent layout");
  }
  if((on&present.logical_not()).any().item<bool>())throw std::invalid_argument("cotangent connects an absent output/state/message");
}
GraphCotangents roots(const ResidentCotangents& r,const ReverseTape& t,const Tensor& state,const Tensor& present) {
  GraphCotangents out{r.outputs.defined()?r.outputs:at::zeros_like(t.outputs.values,t.outputs.values.options().dtype(at::kFloat)),
    r.outputs_connected.defined()?r.outputs_connected:at::zeros_like(t.outputs.valid),
    r.pending.defined()?r.pending:at::zeros_like(t.pending.values,t.pending.values.options().dtype(at::kFloat)),
    r.pending_connected.defined()?r.pending_connected:at::zeros_like(t.pending.valid),
    r.final.defined()?r.final:at::zeros_like(state,state.options().dtype(at::kFloat)),r.final_connected.defined()?r.final_connected:at::zeros_like(present)};
  for(const auto& cache:r.cache)out.cache.push_back({cache.key,cache.value,cache.key_connected,cache.value_connected,cache.log_bias,cache.log_bias_connected});
  return out;
}
}
ResidentGradients ResidentTrainingSession::backward(const std::vector<ResidentCotangents>& input) {
  auto& s=*impl_;s.check();
  if(s.gradients_ready||s.saved.empty())throw std::logic_error("backward requires unconsumed retained windows");
  try{return s.reverse(input);}
  catch(const std::invalid_argument&){throw;} // Preflight/construction has not submitted a reverse program.
  catch(...){s.failed=true;throw;}
}
ResidentGradients ResidentTrainingSession::Impl::reverse(const std::vector<ResidentCotangents>& input) {
  auto& s=*this;
  if(input.size()!=s.saved.size())throw std::invalid_argument("provide cotangents for all retained windows in forward order");
  const auto per=s.limits.backward_bytes/static_cast<Index>(s.saved.size());
  for(size_t i=0;i<input.size();++i) {
    const auto& r=input[i];const auto& w=s.saved[i];const auto& t=w.tape.tape;
    if(r.token.session!=w.token.session||r.token.index!=w.token.index||r.token.generation!=w.token.generation)
      throw std::invalid_argument("stale, foreign or out-of-order resident window token");
    // Admission covers zero roots, exported boundary coordinates/masks and
    // initial connection masking. Component allocations use disjoint budgets.
    long double cache_bytes=0;for(const auto& a:t.attention)cache_bytes+=16.L*a.key.numel()+32.L*a.lengths.numel();
    for(const auto& f:t.fiber)cache_bytes+=16.L*f.cache.key.numel()+8.L*f.bias.numel()+40.L*f.cache.lengths.numel();
    const long double own=cache_bytes+5.L*(t.outputs.values.numel()+static_cast<long double>(t.pending.values.numel()))+13.L*w.final.numel()
      +64.L*(t.fiber_values.size(0)+static_cast<long double>(t.pending.valid.numel()))+4.L*w.present.numel()+1024;
    if(per<8||own>per/8)throw std::invalid_argument("resident backward root/export budget exceeded");
    root_pair(r.outputs,r.outputs_connected,t.outputs.values,t.outputs.valid);
    root_pair(r.pending,r.pending_connected,t.pending.values,t.pending.valid);
    root_pair(r.final,r.final_connected,w.final,w.present);
    if(!r.cache.empty()&&r.cache.size()!=t.attention.size()+t.fiber.size())throw std::invalid_argument("resident cache root group count mismatch");
    for(size_t j=0;j<r.cache.size();++j) {
      const bool fiber=j>=t.attention.size();
      const auto& a=fiber?t.fiber[j-t.attention.size()].cache:t.attention[j];const auto& c=r.cache[j];
      auto ids=at::tensor(a.nodes,at::kLong).to(s.device),present=w.present.index_select(1,ids).reshape({-1});
      root_pair(c.key,c.key_connected,a.key,present);root_pair(c.value,c.value_connected,a.value,present);
      if(fiber)root_pair(c.log_bias,c.log_bias_connected,t.fiber[j-t.attention.size()].bias,present);
      else if(c.log_bias.defined()||c.log_bias_connected.defined())throw std::invalid_argument("event attention has no log-bias cache roots");
    }
  }
  auto error=at::zeros({1},s.layout.values.options().dtype(at::kInt));
  CannProgram p(s.device);p.limit_workspace(s.limits.program_workspace_bytes);
  std::vector<GraphVjp> gradients(s.saved.size());ParameterVjp total;
  for(size_t i=s.saved.size();i>0;) {--i;const auto& w=s.saved[i];auto cot=roots(input[i],w.tape.tape,w.final,w.present);
    if(i+1<s.saved.size())cot=append_window_bridge(p,w.tape.tape,cot,s.saved[i+1].tape.tape,gradients[i+1],error,per/8);
    gradients[i]=append_graph_vjp(p,w.tape.tape,cot,error,s.limits.reverse_chunk_rows,per/2);
    auto partial=append_parameter_vjp(p,s.graph,s.registry,gradients[i],error,per/8);
    total=total.values.defined()?append_parameter_accumulate(p,total,partial,error,per/8):partial;
  }
  p.finish();
  try {
    c10::impl::VirtualGuardImpl(s.device.type()).synchronizeDevice(s.device.index());p.run();
    const auto code=error.cpu().item<int>();p.close();
    if(code)throw std::runtime_error("resident backward refusal code="+std::to_string(code));
    ResidentGradients out;
    for(const auto& owner:total.owners){out.names.push_back(owner.canonical);out.aliases.push_back(owner.aliases);}
    out.offsets=total.offsets;out.values=total.values;out.connected=total.connected;
    out.initial=at::where(s.initial_present.unsqueeze(-1),gradients.front().initial,at::zeros_like(gradients.front().initial));
    out.initial_connected=gradients.front().initial_connected&s.initial_present;
    for(size_t i=0;i<gradients.front().cache.size();++i) {
      const auto& t=s.saved.front().tape.tape;
      const auto& a=i<t.attention.size()?t.attention[i]:t.fiber[i-t.attention.size()].cache;const auto& g=gradients.front().cache[i];
      auto ids=at::tensor(a.nodes,at::kLong).to(s.device),present=s.initial_present.index_select(1,ids).reshape({-1});
      auto mask=present.reshape({-1,1,1,1});
      out.initial_cache.push_back({a.nodes,at::where(mask,g.key,at::zeros_like(g.key)),at::where(mask,g.value,at::zeros_like(g.value)),
        g.lengths,g.key_connected&present,g.value_connected&present,
        g.bias.defined()?at::where(present.unsqueeze(1),g.bias,at::zeros_like(g.bias)):Tensor{},
        g.bias_connected.defined()?g.bias_connected&present:Tensor{}});
    }
    for(size_t i=0;i<s.saved.size();++i) {
      const auto& t=s.saved[i].tape.tape;const auto& g=gradients[i];const auto n=g.links.fibers+g.links.pending;
      auto valid=g.links.valid.narrow(0,0,n)&g.links.messages.narrow(0,0,n).select(1,1).lt(0);
      out.boundaries.push_back({s.saved[i].token,at::cat({t.fiber_meta,t.pending.coordinates}),g.messages.narrow(0,0,n),
        valid,g.message_connected.narrow(0,0,n)&valid});
    }
    s.gradient=std::move(total);s.gradients_ready=true;s.saved.clear();s.saved_bytes=0;return out;
  }catch(...){s.failed=true;throw;}
}
} // namespace tide
