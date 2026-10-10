#include "remote_state.h"
#include <stdexcept>

namespace tide::device_online {
RemoteState::RemoteState(StateOwner& owner,int64_t budget):owner_(owner),budget_(budget) {}
RemoteStateResult RemoteState::append_read_send(DeviceProgram& coordinator,const StateShardBatch& input,
    const at::Tensor& stage,const at::Tensor& error) {
  if(program_)throw std::logic_error("remote state service already constructed");
  const auto remote=owner_.state().values.device();
  if(remote==error.device())throw std::invalid_argument("remote state requires a distinct peer");
  auto buffer=[&](const at::Tensor& x){return at::zeros(x.sizes(),x.options().device(remote));};
  command_=at::zeros_like(error);stop_=at::zeros_like(error);auto work=at::ones_like(error);
  auto remote_command=buffer(command_),remote_error=buffer(error),remote_stage=buffer(stage);
  ReadyBatch ready;ready.atoms={buffer(input.ready.atoms.coordinates),buffer(input.ready.atoms.values),buffer(input.ready.atoms.valid)};
  ready.fibers=buffer(input.ready.fibers);ready.fiber_offsets=buffer(input.ready.fiber_offsets);ready.counts=buffer(input.ready.counts);
  ContentBatch content{buffer(input.content.content),buffer(input.content.scores),buffer(input.content.weighted)};
  request_=std::make_unique<PeerExchange>(PeerExchange::Fields{{command_,remote_command},{error,remote_error},{stage,remote_stage},
    {input.ready.atoms.coordinates,ready.atoms.coordinates},{input.ready.atoms.values,ready.atoms.values},{input.ready.atoms.valid,ready.atoms.valid},
    {input.ready.fibers,ready.fibers},{input.ready.fiber_offsets,ready.fiber_offsets},{input.ready.counts,ready.counts},
    {input.content.content,content.content},{input.content.weighted,content.weighted}},budget_);
  program_=std::make_unique<DeviceProgram>(remote);auto& p=*program_;p.limit_workspace(budget_);
  auto head=p.label(),body=p.label(),end=p.label();auto again=at::zeros_like(remote_command);
  p.mark(head);request_->append_receive(p);p.branch(remote_command,{end,body});p.mark(body);
  const auto read=owner_.append_read(p,ready,content,remote_error);
  stage_error_=at::zeros_like(error);auto scores=at::zeros_like(input.content.scores);
  read_result_=std::make_unique<PeerExchange>(PeerExchange::Fields{{content.scores,scores},{remote_error,stage_error_}},budget_);
  read_result_->append_send(p);
  active_=at::zeros_like(input.ready.atoms.valid);controls_=at::zeros_like(input.content.scores);
  SelectionProposal selected{{},buffer(active_),buffer(controls_),{}};
  selection_=std::make_unique<PeerExchange>(PeerExchange::Fields{{active_,selected.active},{controls_,selected.controls},{stage_error_,remote_error}},budget_);
  selection_->append_receive(p);
  const auto update=owner_.append_update(p,ready,content,selected,read,remote_stage,remote_error);
  auto back=[&](const at::Tensor& x){return at::zeros(x.sizes(),x.options().device(error.device()));};
  RemoteStateResult out{scores,back(update.comparison),back(update.event_meta),back(update.event_values),stage_error_};
  update_result_=std::make_unique<PeerExchange>(PeerExchange::Fields{{update.comparison,out.comparison},{update.event_meta,out.event_meta},
    {update.event_values,out.event_values},{remote_error,stage_error_}},budget_);update_result_->append_send(p);
  commit_error_=at::zeros_like(error);
  decision_=std::make_unique<PeerExchange>(PeerExchange::Fields{{commit_error_,remote_error}},budget_);
  decision_->append_receive(p);owner_.append_commit(p,update,read,selected,remote_error);
  completion_=std::make_unique<PeerExchange>(PeerExchange::Fields{{remote_error,commit_error_}},budget_);
  completion_->append_send(p);p.branch(again,{head});p.mark(end);p.finish();
  coordinator.copy(command_,work);request_->append_send(coordinator);return out;
}
void RemoteState::append_read_receive(DeviceProgram& p){read_result_->append_receive(p);}
void RemoteState::append_update_send(DeviceProgram& p,const SelectionProposal& selected,const at::Tensor& error) {
  p.copy(active_,selected.active);p.copy(controls_,selected.controls);p.copy(stage_error_,error);selection_->append_send(p);
}
void RemoteState::append_update_receive(DeviceProgram& p){update_result_->append_receive(p);}
void RemoteState::append_commit(DeviceProgram& p,const at::Tensor& error) {
  p.copy(commit_error_,error);decision_->append_send(p);completion_->append_receive(p);
}
void RemoteState::append_stop(DeviceProgram& p){p.copy(command_,stop_);request_->append_send(p);}
void RemoteState::submit(){program_->submit();}
void RemoteState::wait(){program_->wait();}
void RemoteState::close() {
  if(program_)program_->close();
  for(auto* x:{request_.get(),read_result_.get(),selection_.get(),update_result_.get(),decision_.get(),completion_.get()})if(x)x->close();
}
int64_t RemoteState::workspace_bytes() const{return program_?program_->workspace_bytes():0;}
int64_t RemoteState::retained_tensor_bytes() const{return program_?program_->retained_tensor_bytes():0;}
int64_t RemoteState::packet_bytes() const {
  int64_t out=0;for(auto* x:{request_.get(),read_result_.get(),selection_.get(),update_result_.get(),decision_.get(),completion_.get()})if(x)out+=x->packet_bytes();return out;
}
} // namespace tide::device_online
