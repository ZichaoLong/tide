#include "block_dispatch.h"
#include "device_launch_tide_block_dispatch.h"
#include <ATen/core/grad_mode.h>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* address(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
}
BlockDispatcher::BlockDispatcher(const BlockDispatchPlan& plan,at::Device device)
  :slots_(plan.tasks.size()),edges_count_(plan.dependencies.size()),groups_(plan.groups),
   rows_(plan.batch_capacity),journal_capacity_(plan.journal_capacity){
  if(at::GradMode::is_enabled()||device.type()!=resident_device_type)
    throw std::invalid_argument("block dispatcher requires explicit no-grad resident device");
  validate_kernel_device(device);
  const long double bytes=90.L*slots_+24.L*edges_count_+9.L*rows_+24.L*journal_capacity_+80;
  if(slots_<1||groups_<1||groups_>=std::numeric_limits<int32_t>::max()||rows_<1||journal_capacity_<1||
     plan.tensor_bytes<1||bytes>plan.tensor_bytes||bytes>std::numeric_limits<int64_t>::max())
    throw std::invalid_argument("invalid block dispatcher capacities");
  std::vector<int64_t> t,e;
  for(const auto& row:plan.tasks)t.insert(t.end(),row.begin(),row.end());
  for(const auto& row:plan.dependencies)e.insert(e.end(),row.begin(),row.end());
  const auto longs=at::TensorOptions().device(device).dtype(at::kLong);
  tasks_=at::tensor(t,at::kLong).reshape({slots_,5}).to(device);
  edges_=at::tensor(e,at::kLong).reshape({edges_count_,3}).to(device);
  valid_=at::ones({slots_},longs.dtype(at::kBool));seeds_=at::zeros_like(valid_);
  epoch_=at::ones({1},longs);state_=at::zeros({slots_,3},longs);proposal_=at::zeros_like(state_);
  selected_=at::full({rows_},slots_,longs);emitted_=at::zeros({rows_},longs.dtype(at::kBool));
  meta_=at::zeros({4},longs);ledger_=at::zeros_like(meta_);journal_=at::full({journal_capacity_,3},-1,longs);
  branch_=at::zeros({1},longs.dtype(at::kInt));error_=at::zeros_like(branch_);
}
void BlockDispatcher::append(DeviceProgram& p,int64_t command){
  const auto tasks=tasks_,edges=edges_,valid=valid_,seeds=seeds_,epoch=epoch_,state=state_,proposal=proposal_,
    selected=selected_,emitted=emitted_,meta=meta_,ledger=ledger_,journal=journal_,branch=branch_,error=error_;
  const auto slots=slots_,edge_count=edges_count_,groups=groups_,rows=rows_,capacity=journal_capacity_;
  p.kernel([=](void* stream){
    check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_dispatch)(1,stream,address(tasks),address(edges),
      address(valid),address(seeds),address(epoch),address(state),address(proposal),address(selected),
      address(emitted),address(meta),address(ledger),address(journal),address(branch),address(error),
      command,slots,edge_count,groups,rows,capacity),"resident block dispatcher");
  },{tasks,edges,valid,seeds,epoch,state,proposal,selected,emitted,meta,ledger,journal,branch,error});
}
void BlockDispatcher::append_reset(DeviceProgram& p){append(p,0);}
void BlockDispatcher::append_choose(DeviceProgram& p){append(p,1);}
void BlockDispatcher::append_commit(DeviceProgram& p){append(p,2);}
int64_t BlockDispatcher::retained_bytes()const{
  int64_t bytes=0;for(const auto& x:{tasks_,edges_,valid_,seeds_,epoch_,state_,proposal_,selected_,emitted_,meta_,ledger_,journal_,branch_,error_})bytes+=x.nbytes();
  return bytes;
}
}
