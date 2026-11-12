#include "block_resident_internal.h"
#include "training_parameters.h"
#include "tide/weighted_column_checkpoint.h"
#include "tide/kernel.h"
#include <ATen/core/grad_mode.h>
#include <limits>
namespace tide::device_online {
namespace {
Model configure_boundary_model(const SettleGraph& graph,Model model){
  configure_model(graph.graph(),model);return model;
}
BlockSettleBoundary metadata(const SettleGraph& g,const Continuation& q){
  BlockSettleBoundary out{at::zeros({q.batch_size,int64_t(g.graph().nodes.size())},at::kBool),
    at::full({q.batch_size,int64_t(g.graph().regions.size())},-1,at::kLong),
    at::full({q.batch_size,int64_t(g.graph().inputs.size())},-1,at::kLong),
    at::full({q.batch_size,int64_t(g.graph().inputs.size())},-1,at::kLong)};
  for(const auto& [owner,value]:q.states)out.node_present[owner.first][owner.second]=true;
  for(const auto& [owner,h]:q.history)out.region_times[owner.first][owner.second]=h.last_time;
  for(const auto& [owner,entry]:q.ledger){out.input_counts[owner.first][owner.second]=entry.first;out.input_times[owner.first][owner.second]=entry.second;}
  return out;
}
Index common_cut(const Tensor& last,Index stride){
  if(!last.defined()||last.scalar_type()!=at::kLong||last.dim()!=1||last.numel()<1)
    throw std::invalid_argument("public block cut requires int64 sample positions");
  const auto cpu=last.cpu();const auto position=cpu[0].item<Index>();
  if(position< -1||position>=std::numeric_limits<Index>::max()/stride||cpu.ne(position).any().item<bool>())
    throw std::invalid_argument("public block Continuation requires one representable complete sample cut");
  return (position+1)*stride;
}
}
BlockResidentSession::Impl::Impl(SettleGraph g,Model m,const Continuation& q,at::Device device,
    ResidentOptimizerKind k,std::vector<OptimizerGroup> gs,BlockResidentLimits l,const BlockCheckpoint* checkpoint)
    :graph(std::move(g)),model(configure_boundary_model(graph,std::move(m))),binding(lower_block_settle_model(graph,model,q,l.cache,l.forward_bytes)),
     session(training_detail::session_id()),batch(q.batch_size),cut(q.cut),kind(k),groups(std::move(gs)),limits(l){
  if(at::GradMode::is_enabled()||(k!=ResidentOptimizerKind::sgd&&k!=ResidentOptimizerKind::adamw))
    throw std::invalid_argument("block resident requires no-grad and an explicit optimizer kind");
  std::optional<BlockSettleCheckpoint> saved;
  if(checkpoint){saved=import_block_checkpoint(*checkpoint);kind=checkpoint->optimizer;groups=checkpoint->groups;generation=checkpoint->generation;
    cut=common_cut(checkpoint->last_positions,graph.stride());}
  else if(groups.empty()){OptimizerGroup group;for(const auto& p:model.parameters().owners())group.parameters.push_back(p.canonical);groups.push_back(std::move(group));}
  for(auto& group:groups)for(auto& name:group.parameters)name=binding.registry.canonical_name(name);
  auto boundary=metadata(graph,q);const auto positions=at::full({batch},q.cut/graph.stride()-1,at::kLong);
  BlockSettleLimits limits{l.forward_bytes,l.backward_bytes,l.optimizer_bytes,l.total_bytes,l.workspace_bytes,l.windows,l.timeout_ms};
  owner=std::make_unique<BlockSettleSession>(binding.graph,binding.bindings,binding.registry,binding.roles,binding.initial,binding.bound_initial,
    device,kind==ResidentOptimizerKind::sgd?DeviceOptimizerKind::sgd:DeviceOptimizerKind::adamw,groups,limits,
    saved?&*saved:nullptr,&boundary,positions);
}
void BlockResidentSession::Impl::check()const{
  if(at::GradMode::is_enabled()||failed||closed||!owner)throw std::logic_error("block resident owner is closed/failed or grad enabled");
  for(const auto& w:windows)block_session_detail::validate(w.versions);
}
BlockResidentSession::BlockResidentSession(SettleGraph g,Model m,const Continuation& q,at::Device d,
    ResidentOptimizerKind k,std::vector<OptimizerGroup> groups,BlockResidentLimits l)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(m),q,d,k,std::move(groups),l,nullptr)){}
BlockResidentSession::BlockResidentSession(SettleGraph g,Model m,const Continuation& q,const BlockCheckpoint& c,at::Device d,BlockResidentLimits l)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(m),q,d,c.optimizer,c.groups,l,&c)){}
BlockResidentSession::~BlockResidentSession()=default;
BlockWindow BlockResidentSession::advance(const Tensor& positions,const Tensor& payload,const Tensor& present){
  auto& s=*impl_;s.check();if(!positions.defined()||positions.dim()!=2||positions.size(0)!=s.batch||positions.size(1)<1)
    throw std::invalid_argument("block public positions must have shape B,L");
  const auto cut=common_cut(positions.select(1,positions.size(1)-1),s.graph.stride());
  auto actual=s.owner->advance(positions,payload,present);
  try{auto view=block_settle_observations(actual,{s.session,actual.token,actual.generation});Impl::Window stored{view,actual,{}};
    for(const auto& [name,field]:view.fields)for(const auto& x:{field.values,field.present,field.valid})block_session_detail::remember(stored.versions,x);
    for(const auto& [name,x]:view.metadata)block_session_detail::remember(stored.versions,x);
    s.windows.push_back(std::move(stored));s.cut=cut;return view;
  }catch(...){s.failed=true;throw;}
}
BlockGradients BlockResidentSession::backward(const std::vector<BlockRoots>& roots){
  auto& s=*impl_;s.check();if(roots.size()!=s.windows.size()||roots.empty())throw std::invalid_argument("public block roots must cover all retained windows");
  std::vector<BlockSettleFlowRoots> input;for(size_t j=0;j<roots.size();++j)
    input.push_back(block_settle_roots(s.windows[j].view,roots[j],*s.windows[j].actual.tape));
  return block_settle_gradients(s.owner->backward(input),*s.windows.front().actual.tape);
}
void BlockResidentSession::step(){auto& s=*impl_;s.check();s.owner->step();s.windows.clear();++s.generation;}
void BlockResidentSession::detach(){auto& s=*impl_;s.check();s.owner->detach();s.windows.clear();}
Index BlockResidentSession::cut()const{impl_->check();return impl_->cut;}
Index BlockResidentSession::generation()const{impl_->check();return impl_->generation;}
ResidentOptimizerKind BlockResidentSession::optimizer_kind()const{impl_->check();return impl_->kind;}
std::vector<OptimizerGroup> BlockResidentSession::optimizer_groups()const{impl_->check();return impl_->groups;}
std::map<std::string,int64_t> BlockResidentSession::resource_usage()const{impl_->check();return impl_->owner->resource_usage();}
void BlockResidentSession::close(){if(!impl_||impl_->closed)return;impl_->owner->close();impl_->windows.clear();impl_->closed=true;}
}
