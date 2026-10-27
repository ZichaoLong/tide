#include "weighted_window_snapshot.h"
#include "tide/parameters.h"
#include <ATen/core/grad_mode.h>
#include <limits>
namespace tide::device_online {
namespace {
Model clone_parameters(Model m){
  std::map<const void*,at::Tensor> copies;
  auto copy=[&](at::Tensor& x){const auto key=x.unsafeGetTensorImpl();
    if(!copies.count(key))copies[key]=x.detach().clone().set_requires_grad(x.requires_grad());x=copies.at(key);};
  for(auto& w:m.nodes){for(auto x:{&w.decay,&w.weight,&w.bias,&w.read})copy(*x);for(auto& [name,x]:w.extra)copy(x);}
  for(auto& w:m.regions)for(auto& [name,x]:w.extra)copy(x);
  for(auto group:{&m.input_scale,&m.output_scale,&m.edge_scale,&m.agg_scale})for(auto& x:*group)copy(x);return m;
}
std::vector<at::Tensor> operands(const WeightedWindowSnapshot& s){
  auto out=s.journal->tensors();out.push_back(s.stages);
  for(const auto& atoms:{s.external,s.pending})for(const auto& x:{atoms.coordinates,atoms.values,atoms.valid})out.push_back(x);
  for(const auto& raw:s.final){out.push_back(raw.floating);out.push_back(raw.integer);}
  out.insert(out.end(),s.present.begin(),s.present.end());out.insert(out.end(),s.history.begin(),s.history.end());return out;
}
}
std::shared_ptr<WeightedWindowSnapshot> WeightedWindowSnapshot::borrow(WeightedFlow& flow){
  if(at::GradMode::is_enabled()||!flow.impl_||flow.impl_->failed||!flow.impl_->ran)
    throw std::invalid_argument("weighted snapshot requires a successful live no-grad window");
  const auto& f=*flow.impl_;auto out=std::make_shared<WeightedWindowSnapshot>();auto& s=*out;
  s.graph=f.graph;s.model=f.model;s.boundary=f.boundary;s.device=f.device;s.limits=f.limits;
  s.samples=f.boundary.batch_size;s.start=f.begin.cpu().item<int64_t>();s.generation=f.generation;s.bytes=f.bytes;s.stages=f.stages;s.external=f.external;s.pending=f.pending->atoms();
  s.journal=std::shared_ptr<WeightedWindowJournal>(f.journal.get(),[](WeightedWindowJournal*){});
  s.programs=f.services->programs();s.programs.push_back(f.program.get());
  for(const auto& n:f.nodes){s.layouts.push_back(n->layout);s.descriptors.emplace_back();
    for(const auto& [name,x]:n->describe->fields)s.descriptors.back().push_back(name);
    s.final.push_back(n->state);s.present.push_back(n->state_present);
    for(const auto& raw:{n->initial,n->state,n->old,n->proposal}){s.held.push_back(raw.floating);s.held.push_back(raw.integer);}
    for(const auto& [name,role]:n->banks.roles()){s.bank_bytes+=role.bank.nbytes();s.held.push_back(role.bank);}
  }
  for(const auto& r:f.regions){s.history.push_back(r->history);
    for(const auto& [name,role]:r->banks.roles()){s.bank_bytes+=role.bank.nbytes();s.held.push_back(role.bank);}}
  auto values=operands(s);s.held.insert(s.held.end(),values.begin(),values.end());return out;
}
std::shared_ptr<WeightedWindowSnapshot> WeightedWindowSnapshot::retain(WeightedFlow& flow,int64_t budget){
  auto out=borrow(flow);auto& s=*out;const auto values=operands(s);long double bytes=0;
  // Each operand receives its own contiguous clone, even if live storage aliases.
  for(const auto& x:values)bytes+=static_cast<long double>(x.numel())*x.element_size();
  if(budget<1||bytes>budget||bytes>std::numeric_limits<int64_t>::max())
    throw std::invalid_argument("retained weighted window exceeds tensor budget before copying");
  s.model=clone_parameters(s.model);s.journal=s.journal->retain(s.journal->allocated_bytes());
  s.stages=s.stages.detach().clone();
  auto copy=[](AtomBatch& a){a={a.coordinates.detach().clone(),a.values.detach().clone(),a.valid.detach().clone()};};
  copy(s.external);copy(s.pending);
  for(auto& raw:s.final)raw={raw.floating.detach().clone(),raw.integer.detach().clone()};
  for(auto& x:s.present)x=x.detach().clone();for(auto& x:s.history)x=x.detach().clone();
  s.bytes=static_cast<int64_t>(bytes);s.programs.clear();s.held=operands(s);
  for(const auto& x:s.held)s.versions_.push_back({x,x._version(),x.const_data_ptr(),x.requires_grad()});
  for(const auto& owner:s.model.parameters(false).owners())s.versions_.push_back({owner.value,owner.value._version(),owner.value.const_data_ptr(),owner.value.requires_grad()});
  return out;
}
void WeightedWindowSnapshot::validate()const{
  for(const auto& x:versions_)if(x.value._version()!=x.version||x.value.const_data_ptr()!=x.data||x.value.requires_grad()!=x.trainable)
    throw std::logic_error("retained weighted window operand or parameter was modified");
}
}
