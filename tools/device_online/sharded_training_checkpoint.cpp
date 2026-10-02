#include "sharded_training_internal.h"
#include <algorithm>
#include <map>
#include <stdexcept>
namespace tide::training_detail {
namespace {
using Field=Tensor ResidentOptimizerState::*;
const std::vector<Field> floating={&ResidentOptimizerState::values,&ResidentOptimizerState::first,
  &ResidentOptimizerState::second,&ResidentOptimizerState::maximum};
struct Shape {Index elements,owners;bool first,second,maximum;};
Shape shape(const device_online::ParameterPlan& p,ResidentOptimizerKind kind,const std::vector<OptimizerGroup>& groups) {
  bool momentum=false,maximum=false;for(const auto& g:groups){momentum|=g.momentum!=0;maximum|=g.amsgrad;}
  const bool adam=kind==ResidentOptimizerKind::adamw;
  return {p.elements,std::max<Index>(1,p.owners.size()),adam||momentum,adam,adam&&maximum};
}
ResidentOptimizerState zero_state(const Shape& s) {
  return {at::zeros({s.elements},at::kFloat),at::zeros({s.first?s.elements:1},at::kFloat),
    at::zeros({s.second?s.elements:1},at::kFloat),at::zeros({s.maximum?s.elements:1},at::kFloat),
    at::zeros({s.owners},at::kLong),at::zeros({s.owners,2},at::kFloat)};
}
void validate_state(const ResidentOptimizerState& value,const Shape& s) {
  const std::vector<std::pair<Tensor,std::vector<Index>>> fields={{value.values,{s.elements}},
    {value.first,{s.first?s.elements:1}},{value.second,{s.second?s.elements:1}},{value.maximum,{s.maximum?s.elements:1}},
    {value.steps,{s.owners}},{value.corrections,{s.owners,2}}};
  for(size_t i=0;i<fields.size();++i) {
    const auto& x=fields[i].first;
    if(!x.defined()||!x.device().is_cpu()||x.scalar_type()!=(i==4?at::kLong:at::kFloat)
        ||x.sizes()!=at::IntArrayRef(fields[i].second)||!x.is_contiguous()||x.requires_grad())
      throw std::invalid_argument("sharded checkpoint optimizer layout mismatch");
    if(x.is_floating_point()&&!at::isfinite(x).all().item<bool>())throw std::invalid_argument("nonfinite checkpoint optimizer state");
  }
  if((value.steps<0).any().item<bool>()||(value.second<0).any().item<bool>()||(value.maximum<0).any().item<bool>()
      ||(value.corrections<0).any().item<bool>()||(value.corrections>1).any().item<bool>())
    throw std::invalid_argument("invalid checkpoint optimizer slots/counters");
}
std::map<std::string,size_t> positions(const device_online::ParameterPlan& layout) {
  std::map<std::string,size_t> out;for(size_t i=0;i<layout.owners.size();++i)out.emplace(layout.owners[i].canonical,i);return out;
}
}
ResidentOptimizerState ShardedTrainingOwner::Impl::optimizer_state() const {
  const auto geometry=shape(global_layout,kind,groups);auto out=zero_state(geometry);
  const auto indices=positions(global_layout);const bool live[4]={true,geometry.first,geometry.second,geometry.maximum};
  for(size_t d=0;d<layout.size();++d) {
    const auto local=optimizers[d]->snapshot();
    for(size_t i=0;i<layout[d].owners.size();++i) {
      const auto& owner=layout[d].owners[i];const auto index=indices.at(owner.canonical);
      const auto offset=global_layout.offsets[index],from=layout[d].offsets[i];
      if((offset<0)!=(from<0))throw std::logic_error("checkpoint canonical layout changed");
      if(offset>=0)for(size_t f=0;f<floating.size();++f)if(live[f])
        (out.*floating[f]).narrow(0,offset,owner.value.numel()).copy_((local.*floating[f]).narrow(0,from,owner.value.numel()));
      out.steps[index].copy_(local.steps[i]);out.corrections[index].copy_(local.corrections[i]);
    }
  }
  return out;
}
void ShardedTrainingOwner::Impl::restore_optimizer(const ResidentTrainingCheckpoint& c) {
  if(c.offsets!=global_layout.offsets)throw std::invalid_argument("checkpoint global parameter layout mismatch");
  const auto geometry=shape(global_layout,kind,groups);validate_state(c.state,geometry);
  for(size_t i=0;i<global_layout.owners.size();++i) {
    const auto& owner=global_layout.owners[i];const auto offset=global_layout.offsets[i];
    if(offset>=0&&!at::equal(c.state.values.narrow(0,offset,owner.value.numel()).reshape(owner.value.sizes()).to(owner.value.scalar_type()),owner.value))
      throw std::invalid_argument("checkpoint named and packed parameters disagree");
    const auto step=c.state.steps[i].item<Index>();
    if((offset<0&&step!=0)||(step==0&&c.state.corrections[i].ne(0).any().item<bool>())
        ||(kind==ResidentOptimizerKind::sgd&&c.state.corrections[i].ne(0).any().item<bool>())
        ||(kind==ResidentOptimizerKind::adamw&&step>0&&c.state.corrections[i].le(0).any().item<bool>()))
      throw std::invalid_argument("checkpoint counter/correction mismatch");
  }
  // Every CPU field is validated before any partition is restored. Repartition
  // at this explicit detached boundary is independent of former physical IDs.
  const auto indices=positions(global_layout);const bool live[4]={true,geometry.first,geometry.second,geometry.maximum};
  std::vector<ResidentOptimizerState> parts;
  for(size_t d=0;d<layout.size();++d) {
    const auto& p=layout[d];Shape local{p.values.numel(),std::max<Index>(1,p.owners.size()),geometry.first,geometry.second,geometry.maximum};
    auto part=zero_state(local);
    for(size_t i=0;i<p.owners.size();++i) {
      const auto& owner=p.owners[i];const auto index=indices.at(owner.canonical);
      const auto from=global_layout.offsets[index],offset=p.offsets[i];
      if(offset>=0)for(size_t f=0;f<floating.size();++f)if(live[f])
        (part.*floating[f]).narrow(0,offset,owner.value.numel()).copy_((c.state.*floating[f]).narrow(0,from,owner.value.numel()));
      part.steps[i].copy_(c.state.steps[index]);part.corrections[i].copy_(c.state.corrections[index]);
    }
    parts.push_back(std::move(part));
  }
  for(size_t d=0;d<parts.size();++d)optimizers[d]->restore(parts[d]);
}
ResidentTrainingCheckpoint ShardedTrainingOwner::checkpoint() const {
  const auto& s=*impl_;s.check();
  if(!s.saved.empty()||s.gradients_ready||s.accumulated_batches)throw std::logic_error("checkpoint requires detach or a completed optimizer step");
  ResidentTrainingCheckpoint out;
  out.generation=s.generation;out.next_token=s.next_token;out.continuation=s.flow->snapshot();
  out.aliases=s.model.parameters(false).alias_partitions();out.trainable=s.registry.names();
  out.optimizer=s.kind;out.groups=s.groups;out.offsets=s.global_layout.offsets;out.state=s.optimizer_state();
  out.mode=s.limits.forward.mode;out.zeta=s.limits.forward.zeta;
  for(const auto& owner:s.model.parameters(false).owners())out.parameters.emplace(owner.canonical,owner.value.detach().clone());
  for(size_t i=0;i<s.global_layout.owners.size();++i)if(s.global_layout.offsets[i]>=0) {
    const auto& owner=s.global_layout.owners[i];
    out.parameters.at(owner.canonical)=out.state.values.narrow(0,s.global_layout.offsets[i],owner.value.numel()).reshape(owner.value.sizes()).to(owner.value.scalar_type()).clone();
  }
  return out;
}
} // namespace tide::training_detail
