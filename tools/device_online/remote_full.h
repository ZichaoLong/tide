#pragma once
#include "packed_full.h"
#include "packed_lh_full.h"
#include "packed_swiglu_full.h"
#include "peer_exchange.h"

namespace tide::device_online {
// One service program executes actual selected Full batches on a peer. The
// graph coordinator still decides readiness/selection on device. This internal
// inference increment does not expose remote adjoints or owner sharding.
class RemoteFull {
 public:
  RemoteFull(PackedFull&,PackedLhFull*,PackedSwiGluFull*,int64_t workspace_budget);
  ActionBatch append_stage(CannProgram&,const ActionBatch&,const at::Tensor& content,
                           const at::Tensor& comparison,const at::Tensor& error,const at::Tensor& chunks);
  void append_stop(CannProgram&);
  void submit();
  void wait();
  void synchronize_inputs() const;
  void close();
  int64_t workspace_bytes() const;
  int64_t retained_tensor_bytes() const;
  int64_t packet_bytes() const;
 private:
  PackedFull& full_;
  PackedLhFull* lh_;
  PackedSwiGluFull* swiglu_;
  int64_t workspace_budget_;
  at::Tensor command_,stop_;
  std::unique_ptr<PeerExchange> request_,response_;
  std::unique_ptr<CannProgram> program_;
};
} // namespace tide::device_online
