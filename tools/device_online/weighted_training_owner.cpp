// Complete-cut owner. Numerical scheduling and VJP stay in the independent
// device programs; this layer controls only externally declared windows.
#include "weighted_training_internal.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <limits>
namespace tide::device_online {
void guard_weighted_training(std::vector<WeightedTrainingVersion>& out,const at::Tensor& x){
  if(x.defined())out.push_back({x,x._version(),x.const_data_ptr(),x.requires_grad()});
}
void validate_weighted_training(const std::vector<WeightedTrainingVersion>& values){
  for(const auto& x:values)if(x.value._version()!=x.version||x.value.const_data_ptr()!=x.data||x.value.requires_grad()!=x.trainable)
    throw std::logic_error("weighted training parameter, window or gradient was modified");
}
int64_t weighted_training_view_bytes(const WeightedTrainingWindow& w){
  // Deliberately counts views as separate storage. Admission stays conservative
  // when a reshape must copy or a backend materializes a field.
  int64_t bytes=0;for(const auto& [name,f]:w.fields)for(const auto& x:{f.values,f.present,f.valid})if(x.defined())bytes+=x.nbytes();
  for(const auto& [name,x]:w.metadata)bytes+=x.nbytes();return bytes;
}
WeightedTrainingSession::Impl::Impl(Graph graph,Model model,const Continuation& q,at::Device d,
    ResidentOptimizerKind k,std::vector<OptimizerGroup> groups,WeightedTrainingLimits l,const ResidentTrainingCheckpoint* c)
    :limits(l),kind(k),device(d),session(training_detail::session_id()){
  training_detail::no_grad();
  if(d.type()!=resident_device_type||d.index()<0||l.windows<1||l.retained_bytes<1||l.backward_bytes<1
      ||l.optimizer_bytes<16||l.forward.tensor_bytes<1||l.total_bytes<1
      ||static_cast<long double>(l.forward.tensor_bytes)+l.retained_bytes+l.backward_bytes+l.optimizer_bytes>l.total_bytes)
    throw std::invalid_argument("weighted training requires explicit device and positive jointly bounded tensor reservations");
  if(k!=ResidentOptimizerKind::sgd&&k!=ResidentOptimizerKind::adamw)throw std::invalid_argument("unknown weighted optimizer");
  for(const auto& p:model.parameters(false).owners()){
    if(!p.value.device().is_cpu()||p.value.scalar_type()!=at::kFloat)
      throw std::invalid_argument("weighted training requires CPU FP32 boundary parameters");
    guard_weighted_training(originals,p.value);}
  std::vector<training_detail::Version> unused;model=training_detail::freeze_model(std::move(model),d,unused);
  if(c){if(c->mode!="weighted"||c->zeta!=1.)throw std::invalid_argument("checkpoint is not a weighted training cut");
    training_detail::restore_parameters(model,*c);}
  flow=std::make_unique<WeightedFlow>(std::move(graph),std::move(model),training_detail::freeze_continuation(q),d,l.forward);
  auto& f=*flow->impl_;registry=f.model.parameters();
  layout=weighted_training_layout(registry,d,l.optimizer_bytes/16);
  optimizer=std::make_unique<DeviceOptimizer>(layout,k==ResidentOptimizerKind::sgd?DeviceOptimizerKind::sgd:DeviceOptimizerKind::adamw,
    std::move(groups),l.optimizer_bytes/2);
  layout.values=optimizer->values();
  if(c){
    if(c->offsets!=layout.offsets||!c->state.values.defined()||!c->state.values.device().is_cpu()
        ||c->state.values.scalar_type()!=at::kFloat||c->state.values.sizes()!=layout.values.sizes())
      throw std::invalid_argument("weighted checkpoint packed parameter layout differs");
    for(size_t i=0;i<layout.owners.size();++i){const auto& p=layout.owners[i];
      if(!at::equal(c->state.values.narrow(0,layout.offsets[i],p.value.numel()).reshape(p.value.sizes()),p.value))
        throw std::invalid_argument("weighted checkpoint named and master parameters disagree");}
    optimizer->restore(c->state);generation=c->generation;next_token=c->next_token;
  }
  // Fixed-capacity journals make retention admission possible before advancing
  // any trajectory. Three dense copies cover the snapshot and derived views;
  // outputs additionally duplicate each physical port, including aliases.
  long double bytes=8;
  for(const auto& x:f.journal->tensors())bytes+=x.nbytes();
  for(const auto& a:{f.external,f.pending->atoms()})for(const auto& x:{a.coordinates,a.values,a.valid})bytes+=x.nbytes();
  for(const auto& n:f.nodes)bytes+=n->state.floating.nbytes()+n->state.integer.nbytes()+n->state_present.nbytes();
  for(const auto& r:f.regions)bytes+=r->history.nbytes();
  bytes=3*bytes+static_cast<long double>(f.limits.stages)*q.batch_size*f.graph.outputs.size()*(4.L*f.model.width()+25);
  if(bytes>l.retained_bytes)throw std::invalid_argument("weighted training cannot retain one complete window within budget");
  window_bytes=static_cast<int64_t>(bytes);
}
void WeightedTrainingSession::Impl::check()const{
  training_detail::no_grad();
  if(!flow||failed||flow->impl_->failed)throw std::logic_error("weighted training is closed/failed; restore a prior cut into a new owner");
  validate_weighted_training(originals);validate_weighted_training(gradient_versions);
  for(const auto& s:saved){s.snapshot->validate();validate_weighted_training(s.versions);}
}
void WeightedTrainingSession::Impl::run(DeviceProgram& p,const at::Tensor& error){
  p.finish();c10::impl::VirtualGuardImpl(device.type()).synchronizeDevice(device.index());p.run(limits.forward.timeout_ms);
  const auto code=error.cpu().item<int>();p.close();
  if(code)throw std::runtime_error("weighted training device program refused, error="+std::to_string(code));
}
void WeightedTrainingSession::Impl::discard(bool accumulator){
  saved.clear();saved_bytes=0;gradient={};gradient_versions.clear();gradients_ready=false;
  if(accumulator){accumulated={};accumulated_batches=0;}
}
WeightedTrainingSession::WeightedTrainingSession(Graph g,Model m,const Continuation& q,at::Device d,
    ResidentOptimizerKind k,std::vector<OptimizerGroup> groups,WeightedTrainingLimits l)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(m),q,d,k,std::move(groups),l)){}
WeightedTrainingSession::WeightedTrainingSession(Graph g,Model m,const ResidentTrainingCheckpoint& c,at::Device d,WeightedTrainingLimits l)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(m),c.continuation,d,c.optimizer,c.groups,l,&c)){}
WeightedTrainingSession::~WeightedTrainingSession()=default;
WeightedTrainingWindow WeightedTrainingSession::advance(const std::vector<External>& xs,Index stop){
  auto& s=*impl_;s.check();
  if(s.gradients_ready)throw std::logic_error("consume weighted gradients with step, accumulate or detach before advance");
  if(s.saved.size()>=size_t(s.limits.windows)||s.window_bytes>s.limits.retained_bytes-s.saved_bytes)
    throw std::invalid_argument("weighted retained window capacity exceeded before advance");
  if(s.next_token==std::numeric_limits<Index>::max())throw std::overflow_error("weighted window tokens exhausted");
  s.flow->impl_->execute(xs,stop); // Input preflight rejects without poisoning the owner.
  try{
    auto snapshot=WeightedWindowSnapshot::retain(*s.flow,s.window_bytes);
    auto view=weighted_training_view(*snapshot,{s.session,s.next_token++,s.generation});
    if(snapshot->bytes+weighted_training_view_bytes(view)>s.window_bytes)throw std::logic_error("weighted retained view exceeded admission");
    Impl::Saved saved{view,snapshot,{}};
    for(const auto& [name,f]:view.fields)for(const auto& x:{f.values,f.present,f.valid})guard_weighted_training(saved.versions,x);
    for(const auto& [name,x]:view.metadata)guard_weighted_training(saved.versions,x);
    s.saved.push_back(std::move(saved));s.saved_bytes+=s.window_bytes;return view;
  }catch(...){s.failed=true;throw;}
}
void WeightedTrainingSession::detach(){auto& s=*impl_;s.check();s.discard();}
Result WeightedTrainingSession::result()const{impl_->check();if(!impl_->flow->impl_->ran)throw std::logic_error("weighted session has no latest window");return impl_->flow->impl_->decode();}
Index WeightedTrainingSession::cut()const{impl_->check();return impl_->flow->impl_->boundary.cut;}
Index WeightedTrainingSession::generation()const{impl_->check();return impl_->generation;}
void WeightedTrainingSession::close(){if(impl_&&impl_->flow){impl_->flow->close();impl_->flow.reset();impl_->discard();impl_->optimizer.reset();impl_->layout={};}}
}
