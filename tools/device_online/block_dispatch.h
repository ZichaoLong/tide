#pragma once
#include "device_program.h"
#include "block_task_plan.h"

namespace tide::device_online {
// Potential tasks, not a CPU-selected execution trace. One task is an indivisible
// semantic unit (e.g. a complete region frame or a certified sequence block).
// Metadata-only execution boundary. Numerical kernels consume selection and
// write emitted; commit publishes actual decisions after their error guards.
// This class does not yet lower a WeightedFlow or implement its numerical VJP.
class BlockDispatcher {
 public:
  BlockDispatcher(const BlockDispatchPlan&,at::Device);
  void append_reset(DeviceProgram&);
  void append_choose(DeviceProgram&);
  void append_commit(DeviceProgram&);
  const at::Tensor& tasks()const{return tasks_;}
  const at::Tensor& dependencies()const{return edges_;}
  const at::Tensor& valid()const{return valid_;}
  const at::Tensor& seeds()const{return seeds_;}
  const at::Tensor& epoch()const{return epoch_;}
  const at::Tensor& state()const{return state_;}
  const at::Tensor& selection()const{return selected_;}
  const at::Tensor& emitted()const{return emitted_;}
  const at::Tensor& metadata()const{return meta_;}
  const at::Tensor& ledger()const{return ledger_;}
  const at::Tensor& journal()const{return journal_;}
  const at::Tensor& branch()const{return branch_;}
  const at::Tensor& error()const{return error_;}
  int64_t retained_bytes()const;
 private:
  void append(DeviceProgram&,int64_t command);
  int64_t slots_,edges_count_,groups_,rows_,journal_capacity_;
  at::Tensor tasks_,edges_,valid_,seeds_,epoch_,state_,proposal_,selected_,emitted_,meta_,ledger_,journal_,branch_,error_;
};
}
