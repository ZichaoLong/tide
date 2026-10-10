#pragma once
#include "state_owner.h"
#include "state_shard_pack.h"
#include "peer_exchange.h"

namespace tide::device_online {
struct RemoteStateResult {at::Tensor scores,comparison,event_meta,event_values,error;};
// Three device protocol phases: propose Read, apply the global selection to a
// state proposal, then commit only after every downstream preflight succeeds.
class RemoteState {
 public:
  explicit RemoteState(StateOwner&,int64_t operator_budget);
  RemoteStateResult append_read_send(DeviceProgram&,const StateShardBatch&,const at::Tensor& stage,const at::Tensor& error);
  void append_read_receive(DeviceProgram&);
  void append_update_send(DeviceProgram&,const SelectionProposal&,const at::Tensor& common_error);
  void append_update_receive(DeviceProgram&);
  void append_commit(DeviceProgram&,const at::Tensor& common_error);
  void append_stop(DeviceProgram&);
  void submit();void wait();void close();
  int64_t workspace_bytes() const;
  int64_t packet_bytes() const;
  int64_t retained_tensor_bytes() const;
 private:
  StateOwner& owner_;
  int64_t budget_;
  at::Tensor command_,stop_,active_,controls_,stage_error_,commit_error_;
  std::unique_ptr<DeviceProgram> program_;
  std::unique_ptr<PeerExchange> request_,read_result_,selection_,update_result_,decision_,completion_;
};
} // namespace tide::device_online
