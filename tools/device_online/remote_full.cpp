#include "remote_full.h"
#include <c10/core/impl/VirtualGuardImpl.h>

namespace tide::device_online {
RemoteFull::RemoteFull(PackedFull& full,PackedLhFull* lh,PackedSwiGluFull* swiglu,int64_t budget)
  :full_(full),lh_(lh),swiglu_(swiglu),workspace_budget_(budget){}
ActionBatch RemoteFull::append_stage(CannProgram& coordinator,const ActionBatch& actions,
    const at::Tensor& content,const at::Tensor& comparison,const at::Tensor& error,const at::Tensor& chunks) {
  auto result=append_send_stage(coordinator,actions,content,comparison,error,chunks);
  append_receive_stage(coordinator);return result;
}
ActionBatch RemoteFull::append_send_stage(CannProgram& coordinator,const ActionBatch& actions,
    const at::Tensor& content,const at::Tensor& comparison,const at::Tensor& error,const at::Tensor& chunks) {
  if(program_)throw std::logic_error("remote Full stage already constructed");
  const auto remote=full_.kinds().device();
  if(actions.values.device()==remote)throw std::invalid_argument("remote Full requires a distinct peer");
  auto buffer=[&](const at::Tensor& x){return at::zeros(x.sizes(),x.options().device(remote));};
  command_=at::zeros({1},error.options());stop_=at::zeros_like(command_);
  auto work=at::full_like(command_,1),remote_command=buffer(command_);
  ActionBatch input{buffer(actions.coordinates),buffer(actions.values),buffer(actions.valid)};
  auto remote_content=buffer(content),remote_comparison=buffer(comparison),remote_error=buffer(error);
  request_=std::make_unique<PeerExchange>(PeerExchange::Fields{{command_,remote_command},
    {actions.coordinates,input.coordinates},{actions.values,input.values},{actions.valid,input.valid},
    {content,remote_content},{comparison,remote_comparison},{error,remote_error}},workspace_budget_);
  program_=std::make_unique<CannProgram>(remote);auto& p=*program_;p.limit_workspace(workspace_budget_);
  auto head=p.label(),body=p.label(),end=p.label();auto again=at::zeros_like(remote_command);
  p.mark(head);request_->append_receive(p);p.branch(remote_command,{end,body});p.mark(body);
  auto result=full_.append_stage(p,input,remote_comparison,remote_error);
  if(lh_)result=lh_->append_stage(p,result,remote_comparison,remote_error,full_.chunks());
  if(swiglu_)result=swiglu_->append_stage(p,result,remote_content,remote_comparison,remote_error,full_.chunks());
  auto output=at::zeros_like(actions.values);
  response_=std::make_unique<PeerExchange>(PeerExchange::Fields{{result.values,output},
    {remote_error,error},{full_.chunks(),chunks}},workspace_budget_);
  response_->append_send(p);p.branch(again,{head});p.mark(end);p.finish();
  coordinator.copy(command_,work);request_->append_send(coordinator);
  return {actions.coordinates,output,actions.valid};
}
void RemoteFull::append_receive_stage(CannProgram& p){response_->append_receive(p);}
void RemoteFull::append_stop(CannProgram& p) {
  if(!program_)throw std::logic_error("remote Full has no stage");
  p.copy(command_,stop_);request_->append_send(p);
}
void RemoteFull::synchronize_inputs() const {
  const auto d=full_.kinds().device();c10::impl::VirtualGuardImpl(d.type()).synchronizeDevice(d.index());
}
void RemoteFull::submit(){program_->submit();}
void RemoteFull::wait(){program_->wait();}
void RemoteFull::close(){if(program_)program_->close();if(request_)request_->close();if(response_)response_->close();}
int64_t RemoteFull::workspace_bytes() const{return program_?program_->workspace_bytes():0;}
int64_t RemoteFull::retained_tensor_bytes() const{return program_?program_->retained_tensor_bytes():0;}
int64_t RemoteFull::packet_bytes() const{return request_?request_->packet_bytes()+response_->packet_bytes():0;}
} // namespace tide::device_online
