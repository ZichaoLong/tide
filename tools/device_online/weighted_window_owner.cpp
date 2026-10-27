#include "weighted_window_vjp.h"
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
namespace tide::device_online {
int64_t WeightedWindowAdjoint::remaining(DeviceProgram& p)const{
  std::vector<at::Tensor> held{error_,stage_,boundary_.coordinates,boundary_.present,boundary_.values,boundary_.connected};
  auto row=[&](const WeightedRowAdjoint& r){if(r.value.defined())held.push_back(r.value);if(r.connected.defined())held.push_back(r.connected);};
  auto raw=[&](const WeightedRawAdjoint& r){row({r.value,r.connected});};
  for(const auto& n:roots_.nodes){row(n.content);row(n.control);row(n.full);raw(n.old);raw(n.proposal);raw(n.final);
    for(const auto& [name,r]:n.describe)row(r);}
  row(roots_.pending);row(roots_.messages);row(roots_.fibers);
  auto programs=services_->programs();
  programs.insert(programs.end(),source_->programs.begin(),source_->programs.end());
  held.insert(held.end(),source_->held.begin(),source_->held.end());
  auto nodes=[&](const auto& collection){for(const auto& n:collection){
    for(const auto& raw:{n->initial,n->state,n->old,n->proposal}){held.push_back(raw.floating);held.push_back(raw.integer);}
    held.push_back(n->state_present);for(const auto& [name,role]:n->banks.roles())held.push_back(role.bank);}};
  auto regions=[&](const auto& collection){for(const auto& r:collection){held.push_back(r->history);
    for(const auto& [name,role]:r->banks.roles())held.push_back(role.bank);}};
  nodes(nodes_);regions(regions_);
  const auto used=p.retained_tensor_bytes(held,programs);
  if(used>=budget_)throw std::invalid_argument("weighted complete window adjoint tensor budget exceeded");return budget_-used;
}
WeightedWindowAdjoint::~WeightedWindowAdjoint()=default;
void WeightedWindowAdjoint::check()const{
  if(at::GradMode::is_enabled()||failed_||!program_
      ||(forward_&&(!forward_->impl_||forward_->impl_->failed||generation_!=forward_->impl_->generation)))
    throw std::logic_error("weighted window adjoint owner is closed, failed or stale");
  source_->validate();
}
int64_t WeightedWindowAdjoint::reserved_bytes()const{check();return budget_-remaining(*program_);}
void WeightedWindowAdjoint::run(){
  check();const auto& s=*source_;
  try{c10::impl::VirtualGuardImpl(s.device.type()).synchronizeDevice(s.device.index());services_->submit();
    program_->run(s.limits.timeout_ms);services_->wait(s.limits.timeout_ms);
    const auto code=error_.cpu().item<int32_t>();if(code)throw std::runtime_error("weighted window VJP refused, device error="+std::to_string(code));
  }catch(...){failed_=true;throw;}
}
std::map<std::string,at::Tensor> WeightedWindowAdjoint::records()const{
  check();const auto& s=*source_;auto result=s.journal->decode(s.stages.cpu().item<int64_t>());
  result["right.pending.meta"]=s.pending.coordinates.cpu().clone();
  result["right.pending.present"]=s.pending.valid.cpu().clone();return result;
}
void WeightedWindowAdjoint::close(){if(program_){program_->close();services_->close();program_.reset();services_.reset();}}
}
