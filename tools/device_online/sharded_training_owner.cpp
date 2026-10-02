#include "sharded_training_internal.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace tide::training_detail {
using namespace device_online;
namespace {
std::vector<ResidentStateWindow> state_windows(const std::vector<StateOwnerValues>& values,const ShardedReverseTape& tape) {
  if(values.size()!=tape.states.size())throw std::logic_error("state tape/view ownership mismatch");
  std::vector<ResidentStateWindow> out;
  for(size_t i=0;i<values.size();++i) {
    const auto& v=values[i];const auto& t=tape.states[i];
    if(v.nodes!=t.global_nodes)throw std::logic_error("state tape/view node mapping mismatch");
    ResidentStateWindow w{v.nodes,v.values.clone(),v.present.clone()};
    auto cache=[&](const EventAttentionTape& a,const Tensor& bias) {
      std::vector<Index> nodes;for(auto n:a.nodes)nodes.push_back(v.nodes.at(n));
      auto ids=at::tensor(a.nodes,at::kLong).to(v.values.device());
      w.cache.push_back({nodes,a.key,a.value,a.lengths,w.present.index_select(1,ids).reshape({-1}),bias});
    };
    for(const auto& a:t.attention)cache(a,{});
    for(const auto& f:t.fiber)cache(f.cache,f.bias);
    out.push_back(std::move(w));
  }
  return out;
}
}
ShardedTrainingOwner::Impl::Impl(Graph g,Model m,const Continuation& q,at::Device d,ResidentOptimizerKind k,
    std::vector<OptimizerGroup> optimizer_groups,ResidentTrainingLimits l,const ResidentTrainingCheckpoint* checkpoint)
    :graph(std::move(g)),device(d),limits(l),placement(l.placement),kind(k),session(session_id()),cut(q.cut) {
  no_grad();
  if(placement.devices.empty()||placement.devices[0]!=d||placement.devices.size()>16)
    throw std::invalid_argument("sharded training devices must start with the explicit coordinator");
  if(l.forward.trace<1||l.windows<1||l.retained_bytes<1||l.backward_bytes<1||l.optimizer_bytes<1
      ||l.program_workspace_bytes<1||l.reverse_chunk_rows<1)
    throw std::invalid_argument("sharded training requires journals and positive finite limits");
  if(k!=ResidentOptimizerKind::sgd&&k!=ResidentOptimizerKind::adamw)throw std::invalid_argument("unknown resident optimizer");
  model=freeze_model(std::move(m),d,versions);
  if(checkpoint) {
    if(checkpoint->mode!=l.forward.mode||checkpoint->zeta!=l.forward.zeta)
      throw std::invalid_argument("resident checkpoint Emit mode/zeta mismatch");
    restore_parameters(model,*checkpoint);
  }
  auto full=place_full(graph,model,placement.devices,placement.policy),state=full;
  if(!placement.full_owners.empty())full.owners=placement.full_owners;
  if(!placement.state_owners.empty())state.owners=placement.state_owners;
  validate_full_placement(full,graph.nodes.size(),d);validate_full_placement(state,graph.nodes.size(),d);
  placement.full_owners=full.owners;placement.state_owners=state.owners;
  registry=model.parameters(true);
  global_layout=plan_parameters(graph,registry,model.width(),std::numeric_limits<Index>::max(),l.forward.mode!="hard");
  layout=sharded_parameter_layout(graph,registry,model.width(),placement.devices,l.optimizer_bytes/4,l.forward.mode!="hard");
  optimizers=make_sharded_optimizers(layout,k==ResidentOptimizerKind::sgd?DeviceOptimizerKind::sgd:DeviceOptimizerKind::adamw,
    std::move(optimizer_groups),l.optimizer_bytes/2/placement.devices.size());
  for(size_t i=0;i<layout.size();++i)layout[i].values=optimizers[i]->values();
  groups=optimizers.front()->groups();for(auto& group:groups)group.parameters.clear();
  for(const auto& optimizer:optimizers)for(size_t i=0;i<groups.size();++i) {
    const auto& names=optimizer->groups().at(i).parameters;groups[i].parameters.insert(groups[i].parameters.end(),names.begin(),names.end());
  }
  for(auto& group:groups)std::sort(group.parameters.begin(),group.parameters.end());
  if(checkpoint){restore_optimizer(*checkpoint);generation=checkpoint->generation;next_token=checkpoint->next_token;}
  flow=std::make_unique<ContentFlow>(graph,model,freeze_continuation(q),d,l.forward,ModelPlacement{full,state},true);
  const auto tape=flow->sharded_reverse_tape();const auto values=flow->state_shards_device();
  long double bytes=sharded_reverse_tape_bytes(tape)+256;
  for(const auto& v:values)bytes+=static_cast<long double>(v.values.nbytes())+v.present.nbytes();
  if(bytes>l.retained_bytes)throw std::invalid_argument("sharded training cannot retain one window within budget");
  projection_bytes=RetainedProjection::bytes(tape.coordinator.emission.weights,tape.coordinator.emission.biases);
  for(const auto& bank:tape.coordinator.emission.shards)projection_bytes+=RetainedProjection::bytes(bank.weights,bank.biases);
  bytes_per_window=static_cast<Index>(bytes)-projection_bytes;discard();
}
void ShardedTrainingOwner::Impl::check() const {
  no_grad();if(!flow)throw std::logic_error("sharded resident training is closed");
  if(failed)throw std::logic_error("sharded resident training failed; restore a prior checkpoint into a new owner");
  for(const auto& p:versions)if(p.value._version()!=p.version||p.value.const_data_ptr()!=p.data)
    throw std::logic_error("caller parameters changed after resident training construction");
}
void ShardedTrainingOwner::Impl::discard() {
  saved.clear();projection_snapshot={};gradient.clear();saved_bytes=0;gradients_ready=false;initial_present.clear();
  accumulated.clear();accumulated_batches=0;
  for(const auto& s:flow->state_shards_device())initial_present.push_back(s.present.clone());
}
ShardedTrainingOwner::ShardedTrainingOwner(Graph g,Model m,const Continuation& q,at::Device d,ResidentOptimizerKind k,
    std::vector<OptimizerGroup> groups,ResidentTrainingLimits l,const ResidentTrainingCheckpoint* c)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(m),q,d,k,std::move(groups),l,c)){}
ShardedTrainingOwner::~ShardedTrainingOwner()=default;
void ShardedTrainingOwner::check() const {impl_->check();}
ResidentTrainingWindow ShardedTrainingOwner::advance(const std::vector<External>& input,Index stop,Index seal) {
  auto& s=*impl_;s.check();
  if(s.gradients_ready)throw std::logic_error("consume gradients with step or detach before advance");
  if(seal<stop)throw std::invalid_argument("resident training window is unsealed");
  const auto required=s.bytes_per_window+(s.saved.empty()?s.projection_bytes:0);
  if((s.saved.size()+1.L)*s.bytes_per_window+s.projection_bytes>std::numeric_limits<Index>::max())
    throw std::invalid_argument("retained dense envelope extent overflow");
  if(s.saved.size()>=size_t(s.limits.windows)||required>s.limits.retained_bytes-s.saved_bytes)
    throw std::invalid_argument("resident retained-window capacity exceeded; backward or explicitly detach first");
  if(s.next_token==std::numeric_limits<Index>::max())throw std::overflow_error("resident window token exhausted");
  ContentWindow w;
  try{w=s.flow->advance_device(input,stop);}catch(const std::invalid_argument&){throw;}catch(...){s.failed=true;throw;}
  try {
    const bool compact=s.limits.forward.chunk_policy==ResidentChunkPolicy::aggressive;
    auto tape=retain_sharded_reverse_tape(s.flow->sharded_reverse_tape(),s.limits.retained_bytes-s.saved_bytes,&s.projection_snapshot,compact);
    auto states=state_windows(s.flow->state_shards_device(),tape.tape);const auto& t=tape.tape.coordinator;
    ResidentToken token{s.session,s.next_token++,s.generation};
    ResidentTrainingWindow out{token,s.cut,stop,{t.outputs.coordinates,t.outputs.values,t.outputs.valid,
      w.output_stats.clone(),w.pending_stats.clone(),w.stages.clone(),w.events.clone(),w.full_chunks.clone(),w.emission_chunks.clone()},
      t.pending.coordinates,t.pending.values,t.pending.valid};
    Index retained=tape.tensor_bytes+256;
    for(const auto& state:states)retained+=state.values.nbytes()+state.present.nbytes();
    if(retained>required)throw std::logic_error("retained journal packing exceeded dense admission");
    out.states=states;s.saved.push_back({token,std::move(tape),std::move(states)});s.saved_bytes+=retained;s.cut=stop;return out;
  }catch(...){s.failed=true;throw;}
}
ResidentStep ShardedTrainingOwner::step() {
  auto& s=*impl_;s.check();
  if(!s.saved.empty()||(!s.gradients_ready&&!s.accumulated_batches))throw std::logic_error("resident step requires completed backward and no outstanding windows");
  if(s.gradients_ready&&s.accumulated_batches)throw std::logic_error("accumulate the final backward before step");
  if(s.generation==std::numeric_limits<Index>::max())throw std::overflow_error("resident parameter generation exhausted");
  try {
    auto error=at::zeros({1},at::TensorOptions().device(s.device).dtype(at::kInt));
    ShardedParameterReduce update(s.accumulated_batches?s.accumulated:s.gradient,error,s.limits.optimizer_bytes/4,s.limits.program_workspace_bytes);
    std::vector<DeviceOptimizer*> owners;for(const auto& p:s.optimizers)owners.push_back(p.get());
    update.append_step(owners);update.append_publish(s.flow->sharded_parameter_banks(),owners,s.limits.optimizer_bytes/4);
    update.finish();update.run();const auto code=update.errors()[0].cpu().item<int>();update.close();
    if(code==20||code==21)return {false,code,s.generation};
    if(code)throw std::runtime_error("sharded optimizer refusal code="+std::to_string(code));
    ++s.generation;s.discard();return {true,0,s.generation};
  }catch(const std::invalid_argument&){throw;}catch(...){s.failed=true;throw;}
}
void ShardedTrainingOwner::detach(){auto& s=*impl_;s.check();try{s.discard();}catch(...){s.failed=true;throw;}}
Result ShardedTrainingOwner::result() const {impl_->check();return impl_->flow->result();}
Index ShardedTrainingOwner::cut() const {impl_->check();return impl_->cut;}
Index ShardedTrainingOwner::generation() const {impl_->check();return impl_->generation;}
Index ShardedTrainingOwner::retained_windows() const {impl_->check();return impl_->saved.size();}
Index ShardedTrainingOwner::accumulated_batches() const {impl_->check();return impl_->accumulated_batches;}
ResidentPlacement ShardedTrainingOwner::placement() const {impl_->check();return impl_->placement;}
void ShardedTrainingOwner::close() {
  auto& s=*impl_;
  if(s.flow){s.flow->close();s.flow.reset();s.saved.clear();s.projection_snapshot={};s.optimizers.clear();s.gradient.clear();s.accumulated.clear();s.layout.clear();}
}
} // namespace tide::training_detail
