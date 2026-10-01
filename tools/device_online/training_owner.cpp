#include "training_internal.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <atomic>
#include <limits>
#include <stdexcept>

namespace tide {
using namespace device_online;
namespace {std::atomic<uint64_t> next_session{1};}
ResidentTrainingSession::Impl::Impl(Graph g,Model m,const Continuation& q,at::Device d,ResidentOptimizerKind k,
    std::vector<OptimizerGroup> groups,ResidentTrainingLimits l,const ResidentTrainingCheckpoint* checkpoint)
    :graph(std::move(g)),device(d),limits(l),kind(k),session(next_session.fetch_add(1)),cut(q.cut) {
  training_detail::no_grad();
  if(d.type()!=c10::DeviceType::PrivateUse1||d.index()<0)
    throw std::invalid_argument("resident training requires an explicit logical NPU");
  if(!l.forward.diagnostics||l.windows<1||l.retained_bytes<1||l.backward_bytes<1||l.optimizer_bytes<1
      ||l.program_workspace_bytes<1||l.reverse_chunk_rows<1)
    throw std::invalid_argument("resident training requires recorded journals and positive memory/window limits");
  if(k!=ResidentOptimizerKind::sgd&&k!=ResidentOptimizerKind::adamw)throw std::invalid_argument("unknown resident optimizer");
  model=training_detail::freeze_model(std::move(m),d,versions);
  if(checkpoint) {
    if(checkpoint->mode!=l.forward.mode||checkpoint->zeta!=l.forward.zeta)
      throw std::invalid_argument("resident checkpoint Emit mode/zeta mismatch");
    training_detail::restore_parameters(model,*checkpoint);
  }
  // Filtering by TensorImpl requires_grad retains every alias of each selected
  // owner, including HARD Read aliases that need publication but have no VJP.
  registry=model.parameters(true);
  layout=parameter_layout(graph,registry,model.width(),device,l.optimizer_bytes/4,l.forward.mode!="hard");
  optimizer=std::make_unique<DeviceOptimizer>(layout,k==ResidentOptimizerKind::sgd?DeviceOptimizerKind::sgd:DeviceOptimizerKind::adamw,
                                             std::move(groups),l.optimizer_bytes/2);
  if(checkpoint) {
    if(checkpoint->offsets!=layout.offsets)throw std::invalid_argument("checkpoint parameter layout mismatch");
    const auto& packed=checkpoint->state.values;
    if(!packed.defined()||!packed.device().is_cpu()||packed.scalar_type()!=at::kFloat||packed.sizes()!=layout.values.sizes())
      throw std::invalid_argument("checkpoint packed parameter layout mismatch");
    for(size_t i=0;i<layout.owners.size();++i)if(layout.offsets[i]>=0) {
      const auto& owner=layout.owners[i];
      if(!at::equal(packed.narrow(0,layout.offsets[i],owner.value.numel()).reshape(owner.value.sizes()),owner.value))
        throw std::invalid_argument("checkpoint named and packed parameter values disagree");
    }
    optimizer->restore(checkpoint->state);generation=checkpoint->generation;next_token=checkpoint->next_token;
  }
  flow=std::make_unique<ContentFlow>(graph,model,training_detail::freeze_continuation(q),device,l.forward);
  // Profile preflight happens before any input window can be executed.
  const auto tape=flow->reverse_tape();const auto state=flow->state_device();
  const long double bytes=reverse_tape_bytes(tape)+static_cast<long double>(state.first.numel())*4+state.second.numel()+256;
  if(bytes>l.retained_bytes)throw std::invalid_argument("resident training cannot retain one window within budget");
  bytes_per_window=static_cast<Index>(bytes);initial_present=state.second.clone();
}
void ResidentTrainingSession::Impl::check() const {
  training_detail::no_grad();
  if(!flow)throw std::logic_error("resident training session is closed");
  if(failed)throw std::logic_error("resident training failed; restore a prior checkpoint into a new owner");
  for(const auto& p:versions)if(p.value._version()!=p.version||p.value.const_data_ptr()!=p.data)
    throw std::logic_error("caller parameters changed after resident training construction");
}
void ResidentTrainingSession::Impl::discard() {
  saved.clear();saved_bytes=0;gradient={};gradients_ready=false;
  initial_present=flow->state_device().second.clone();
}
ResidentTrainingSession::ResidentTrainingSession(Graph g,Model m,const Continuation& q,at::Device d,
    ResidentOptimizerKind k,std::vector<OptimizerGroup> groups,ResidentTrainingLimits l)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(m),q,d,k,std::move(groups),l)) {}
ResidentTrainingSession::ResidentTrainingSession(Graph g,Model m,const ResidentTrainingCheckpoint& c,at::Device d,ResidentTrainingLimits l)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(m),c.continuation,d,c.optimizer,c.groups,l,&c)) {}
ResidentTrainingSession::~ResidentTrainingSession()=default;
ResidentTrainingWindow ResidentTrainingSession::advance(const std::vector<External>& input,Index stop,Index seal) {
  auto& s=*impl_;s.check();
  if(s.gradients_ready)throw std::logic_error("consume gradients with step or detach before advance");
  if(seal<stop)throw std::invalid_argument("resident training window is unsealed");
  if(s.saved.size()>=size_t(s.limits.windows)||s.bytes_per_window>s.limits.retained_bytes-s.saved_bytes)
    throw std::invalid_argument("resident retained-window capacity exceeded; backward or explicitly detach first");
  if(s.next_token==std::numeric_limits<Index>::max())throw std::overflow_error("resident window token exhausted");
  ContentWindow window;
  try {window=s.flow->advance_device(input,stop);}
  catch(const std::invalid_argument&){throw;} // Complete input preflight is retryable.
  catch(...){s.failed=true;throw;}
  try {
    auto tape=retain_reverse_tape(s.flow->reverse_tape(),s.limits.retained_bytes-s.saved_bytes);
    const auto state=s.flow->state_device();auto final=state.first.clone(),present=state.second.clone();
    ResidentToken token{s.session,s.next_token++,s.generation};
    auto outputs=ResidentWindow{tape.tape.outputs.coordinates,tape.tape.outputs.values,tape.tape.outputs.valid,
      window.output_stats.clone(),window.pending_stats.clone(),window.stages.clone(),window.events.clone(),
      window.full_chunks.clone(),window.emission_chunks.clone()};
    ResidentTrainingWindow result{token,s.cut,stop,outputs,tape.tape.pending.coordinates,tape.tape.pending.values,
      tape.tape.pending.valid,final,present};
    for(const auto& a:tape.tape.attention) {
      auto ids=at::tensor(a.nodes,at::kLong).to(s.device);
      result.cache.push_back({a.nodes,a.key,a.value,a.lengths,present.index_select(1,ids).reshape({-1})});
    }
    for(const auto& f:tape.tape.fiber) {
      const auto& a=f.cache;auto ids=at::tensor(a.nodes,at::kLong).to(s.device);
      result.cache.push_back({a.nodes,a.key,a.value,a.lengths,present.index_select(1,ids).reshape({-1}),f.bias});
    }
    s.saved.push_back({token,std::move(tape),final,present});s.saved_bytes+=s.bytes_per_window;s.cut=stop;return result;
  }catch(...){s.failed=true;throw;}
}
ResidentStep ResidentTrainingSession::step() {
  auto& s=*impl_;s.check();
  if(!s.gradients_ready)throw std::logic_error("resident step requires completed backward");
  if(s.generation==std::numeric_limits<Index>::max())throw std::overflow_error("resident parameter generation exhausted");
  try {
    auto error=at::zeros({1},s.layout.values.options().dtype(at::kInt));
    CannProgram p(s.device);p.limit_workspace(s.limits.program_workspace_bytes);
    s.optimizer->append_step(p,s.gradient,error);
    append_parameter_publish(p,s.flow->parameter_banks(),s.layout,s.optimizer->values(),error,s.limits.optimizer_bytes/4);
    p.finish();
    c10::impl::VirtualGuardImpl(s.device.type()).synchronizeDevice(s.device.index());p.run();
    const auto code=error.cpu().item<int>();p.close();
    if(code==20||code==21)return {false,code,s.generation}; // Transaction rejected, live values unchanged.
    if(code)throw std::runtime_error("resident optimizer refusal code="+std::to_string(code));
    ++s.generation;s.discard();return {true,0,s.generation};
  }catch(const std::invalid_argument&){throw;}
  catch(...){s.failed=true;throw;}
}
void ResidentTrainingSession::detach() {auto& s=*impl_;s.check();try{s.discard();}catch(...){s.failed=true;throw;}}
Result ResidentTrainingSession::result() const {impl_->check();return impl_->flow->result();}
Index ResidentTrainingSession::cut() const {impl_->check();return impl_->cut;}
Index ResidentTrainingSession::generation() const {impl_->check();return impl_->generation;}
Index ResidentTrainingSession::retained_windows() const {impl_->check();return impl_->saved.size();}
void ResidentTrainingSession::close() {
  if(impl_->flow){impl_->flow->close();impl_->flow.reset();impl_->saved.clear();impl_->optimizer.reset();impl_->gradient={};}
}
} // namespace tide
