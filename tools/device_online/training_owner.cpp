#include "device_backend.h"
#include "training_internal.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <limits>
#include <stdexcept>

namespace tide {
using namespace device_online;
ResidentTrainingSession::Impl::Impl(Graph g,Model m,const Continuation& q,at::Device d,ResidentOptimizerKind k,
    std::vector<OptimizerGroup> groups,ResidentTrainingLimits l,const ResidentTrainingCheckpoint* checkpoint)
    :graph(std::move(g)),device(d),limits(l),kind(k),session(training_detail::session_id()),cut(q.cut) {
  training_detail::no_grad();
  if(d.type()!=tide::device_online::resident_device_type||d.index()<0)
    throw std::invalid_argument("resident training requires an explicit logical resident device");
  if(l.forward.trace<1||l.windows<1||l.retained_bytes<1||l.backward_bytes<1||l.optimizer_bytes<1
      ||l.program_workspace_bytes<1||l.reverse_chunk_rows<1)
    throw std::invalid_argument("resident training requires recorded journals and positive memory/window limits");
  if(k!=ResidentOptimizerKind::sgd&&k!=ResidentOptimizerKind::adamw)throw std::invalid_argument("unknown resident optimizer");
  if(!l.placement.devices.empty()) {
    sharded=std::make_unique<training_detail::ShardedTrainingOwner>(graph,std::move(m),q,d,k,std::move(groups),l,checkpoint);return;
  }
  if(!l.placement.full_owners.empty()||!l.placement.state_owners.empty())
    throw std::invalid_argument("explicit owner maps require a logical device list");
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
  layout.values=optimizer->values(); // Shape/device metadata; no dummy gradient bank.
  if(checkpoint) {
    if(checkpoint->offsets!=layout.offsets)throw std::invalid_argument("checkpoint parameter layout mismatch");
    const auto& packed=checkpoint->state.values;
    if(!packed.defined()||!packed.device().is_cpu()||packed.scalar_type()!=at::kFloat||packed.sizes()!=layout.values.sizes())
      throw std::invalid_argument("checkpoint packed parameter layout mismatch");
    for(size_t i=0;i<layout.owners.size();++i)if(layout.offsets[i]>=0) {
      const auto& owner=layout.owners[i];
      if(!at::equal(packed.narrow(0,layout.offsets[i],owner.value.numel()).reshape(owner.value.sizes()).to(owner.value.scalar_type()),owner.value))
        throw std::invalid_argument("checkpoint named and packed parameter values disagree");
    }
    optimizer->restore(checkpoint->state);generation=checkpoint->generation;next_token=checkpoint->next_token;
  }
  flow=std::make_unique<ContentFlow>(graph,model,training_detail::freeze_continuation(q),device,l.forward,true);
  // Profile preflight happens before any input window can be executed.
  const auto tape=flow->reverse_tape();const auto state=flow->state_device();
  const long double bytes=reverse_tape_bytes(tape)+static_cast<long double>(state.first.numel())*state.first.element_size()+state.second.numel()+256;
  if(bytes>l.retained_bytes)throw std::invalid_argument("resident training cannot retain one window within budget");
  projection_bytes=RetainedProjection::bytes(tape.emission.weights,tape.emission.biases);
  attention_bytes=RetainedAttention::bytes(tape.attention,tape.fiber);
  full_bytes=RetainedFull::bytes(tape.full);
  bytes_per_window=static_cast<Index>(bytes)-projection_bytes-attention_bytes-full_bytes;initial_present=state.second.clone();
}
void ResidentTrainingSession::Impl::check() const {
  if(sharded){sharded->check();return;}
  training_detail::no_grad();
  if(!flow)throw std::logic_error("resident training session is closed");
  if(failed)throw std::logic_error("resident training failed; restore a prior checkpoint into a new owner");
  for(const auto& p:versions)if(p.value._version()!=p.version||p.value.const_data_ptr()!=p.data)
    throw std::logic_error("caller parameters changed after resident training construction");
}
void ResidentTrainingSession::Impl::discard() {
  saved.clear();projection_snapshot={};attention_snapshot={};full_snapshot={};saved_bytes=0;gradient={};gradients_ready=false;
  accumulated={};accumulated_batches=0;
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
  if(s.sharded)return s.sharded->advance(input,stop,seal);
  if(s.gradients_ready)throw std::logic_error("consume gradients with step or detach before advance");
  if(seal<stop)throw std::invalid_argument("resident training window is unsealed");
  const auto required=s.bytes_per_window+(s.saved.empty()?s.projection_bytes+s.attention_bytes+s.full_bytes:0);
  if((s.saved.size()+1.L)*s.bytes_per_window+s.projection_bytes+s.attention_bytes+s.full_bytes>std::numeric_limits<Index>::max())
    throw std::invalid_argument("retained dense envelope extent overflow");
  if(s.saved.size()>=size_t(s.limits.windows)||required>s.limits.retained_bytes-s.saved_bytes)
    throw std::invalid_argument("resident retained-window capacity exceeded; backward or explicitly detach first");
  if(s.next_token==std::numeric_limits<Index>::max())throw std::overflow_error("resident window token exhausted");
  const auto banks=s.flow->parameter_banks();s.attention_snapshot.bind(banks.attention,banks.fiber);
  s.full_snapshot.validate(s.flow->full_tape());
  ContentWindow window;
  try {window=s.flow->advance_device(input,stop);}
  catch(const std::invalid_argument&){throw;} // Complete input preflight is retryable.
  catch(...){s.failed=true;throw;}
  try {
    const bool compact=s.limits.forward.chunk_policy==ResidentChunkPolicy::aggressive;
    auto tape=retain_reverse_tape(s.flow->reverse_tape(),s.limits.retained_bytes-s.saved_bytes,&s.projection_snapshot,compact,&s.attention_snapshot,&s.full_snapshot);
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
    const Index retained=tape.tensor_bytes+Index(final.nbytes())+Index(present.nbytes())+256;
    if(retained>required)throw std::logic_error("retained journal packing exceeded dense admission");
    s.saved.push_back({token,std::move(tape),final,present});s.saved_bytes+=retained;s.cut=stop;return result;
  }catch(...){s.failed=true;throw;}
}
ResidentStep ResidentTrainingSession::step() {
  auto& s=*impl_;s.check();
  if(s.sharded)return s.sharded->step();
  if(!s.saved.empty()||(!s.gradients_ready&&!s.accumulated_batches))throw std::logic_error("resident step requires completed backward and no outstanding windows");
  if(s.gradients_ready&&s.accumulated_batches)throw std::logic_error("accumulate the final backward before step");
  if(s.generation==std::numeric_limits<Index>::max())throw std::overflow_error("resident parameter generation exhausted");
  try {
    auto error=at::zeros({1},s.layout.values.options().dtype(at::kInt));
    DeviceProgram p(s.device);p.limit_workspace(s.limits.program_workspace_bytes);
    s.optimizer->append_step(p,s.accumulated_batches?s.accumulated:s.gradient,error);
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
void ResidentTrainingSession::detach() {auto& s=*impl_;s.check();if(s.sharded){s.sharded->detach();return;}try{s.discard();}catch(...){s.failed=true;throw;}}
Result ResidentTrainingSession::result() const {impl_->check();return impl_->sharded?impl_->sharded->result():impl_->flow->result();}
Index ResidentTrainingSession::cut() const {impl_->check();return impl_->sharded?impl_->sharded->cut():impl_->cut;}
Index ResidentTrainingSession::generation() const {impl_->check();return impl_->sharded?impl_->sharded->generation():impl_->generation;}
Index ResidentTrainingSession::retained_windows() const {impl_->check();return impl_->sharded?impl_->sharded->retained_windows():impl_->saved.size();}
Index ResidentTrainingSession::accumulated_batches() const {impl_->check();return impl_->sharded?impl_->sharded->accumulated_batches():impl_->accumulated_batches;}
ResidentPlacement ResidentTrainingSession::placement() const {
  impl_->check();if(impl_->sharded)return impl_->sharded->placement();
  return {{impl_->device},"locality",std::vector<Index>(impl_->graph.nodes.size(),0),std::vector<Index>(impl_->graph.nodes.size(),0)};
}
void ResidentTrainingSession::close() {
  if(impl_->sharded){impl_->sharded->close();return;}
  if(impl_->flow){impl_->flow->close();impl_->flow.reset();impl_->saved.clear();impl_->projection_snapshot={};impl_->attention_snapshot={};impl_->full_snapshot={};impl_->optimizer.reset();impl_->gradient={};impl_->accumulated={};}
}
} // namespace tide
