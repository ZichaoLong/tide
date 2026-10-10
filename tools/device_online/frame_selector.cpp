#include "device_backend.h"
#include "frame_selector.h"
#include "device_launch_tide_frame_select.h"
#include <ATen/core/grad_mode.h>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* address(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void validate(const at::Tensor& x,at::IntArrayRef shape,at::ScalarType type,at::Device device) {
  if(!x.defined()||x.sizes()!=shape||x.scalar_type()!=type||x.device()!=device
      ||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid frame selector buffer");
}
}
FrameSelector::FrameSelector(std::vector<int64_t> owners,std::vector<SelectionPolicy> policies,
                             int64_t samples,at::Device device,int64_t budget)
    :nodes_(owners.size()),regions_(policies.size()),samples_(samples),budget_(budget),device_(device) {
  if(nodes_<1||regions_<1||samples<1||budget<1||at::GradMode::is_enabled()
      ||device.type()!=tide::device_online::resident_device_type||nodes_>std::numeric_limits<int64_t>::max()/32/samples
      ||regions_>std::numeric_limits<int64_t>::max()/32/samples)
    throw std::invalid_argument("frame selector requires bounded dimensions, NPU and no-grad");
  for(auto r:owners)if(r<0||r>=regions_)throw std::invalid_argument("invalid selector region owner");
  std::vector<int64_t> rows;
  for(auto p:policies) {if(p.budget<0)throw std::invalid_argument("negative selection budget");rows.insert(rows.end(),{p.budget,p.count_priority,p.positive_only});}
  validate_kernel_device(device);
  owner_=at::tensor(owners,at::kLong).to(device);policies_=at::tensor(rows,at::kLong).reshape({regions_,3}).to(device);
}
SelectionHistory FrameSelector::initial() const {
  auto opts=at::TensorOptions().device(device_).dtype(at::kLong);
  return {at::zeros({samples_,nodes_},opts),at::zeros({samples_,nodes_},opts.dtype(at::kBool)),
          at::full({samples_,regions_},-1,opts),at::zeros({samples_,regions_},opts.dtype(at::kBool))};
}
SelectionProposal FrameSelector::append_stage(DeviceProgram& p,const ReadyBatch& ready,const at::Tensor& scores,
                                             const SelectionHistory& old,const at::Tensor& error) const {
  if(!ready.fibers.defined()||ready.fibers.dim()!=2||ready.fibers.size(0)<1)
    throw std::invalid_argument("selector requires nonempty physical frame slots");
  const auto capacity=ready.fibers.size(0);
  // Approximate extra scratch: three padded FP32 tables, fiber indexes,
  // outputs and complete proposed history. Separate from whole-model memory.
  const auto fixed=samples_*nodes_*9+samples_*regions_*9;
  if(fixed>=budget_||nodes_>std::numeric_limits<int64_t>::max()/16-16
      ||capacity>(budget_-fixed)/(nodes_*12+64))
    throw std::invalid_argument("frame selector scratch budget exceeded");
  validate(ready.fibers,{capacity,4},at::kLong,device_);validate(ready.frames,{capacity,3},at::kLong,device_);
  validate(ready.frame_offsets,{capacity+1},at::kLong,device_);validate(ready.frame_fibers,{capacity},at::kLong,device_);
  validate(ready.counts,{3},at::kLong,device_);validate(scores,{capacity},at::kFloat,device_);validate(error,{1},at::kInt,device_);
  validate(old.counts,{samples_,nodes_},at::kLong,device_);validate(old.seen,{samples_,nodes_},at::kBool,device_);
  validate(old.last_time,{samples_,regions_},at::kLong,device_);validate(old.present,{samples_,regions_},at::kBool,device_);
  SelectionProposal out{{at::empty_like(old.counts),at::empty_like(old.seen),at::empty_like(old.last_time),at::empty_like(old.present)},
    at::zeros({capacity},old.seen.options()),at::zeros_like(scores),at::zeros_like(error)};
  p.copy(out.history.counts,old.counts);p.copy(out.history.seen,old.seen);
  p.copy(out.history.last_time,old.last_time);p.copy(out.history.present,old.present);
  auto padded=at::empty({capacity,nodes_},scores.options()),probabilities=at::empty_like(padded);
  auto joined=at::zeros({capacity*nodes_+1},scores.options()),order=at::empty({capacity},ready.fibers.options());
  const auto owner=owner_,policies=policies_;const auto nodes=nodes_,regions=regions_,samples=samples_;
  p.kernel([=](void* stream) {
    check_device_launch(TIDE_LAUNCH_KERNEL(tide_frame_select)(1,stream,address(ready.fibers),address(ready.frames),
      address(ready.frame_offsets),address(ready.frame_fibers),address(ready.counts),address(owner),address(policies),
      address(scores),address(out.history.counts),address(out.history.seen),address(out.history.last_time),
      address(out.history.present),address(padded),address(order),address(out.active),address(out.branch),address(error),
      capacity,nodes,regions,samples),"device frame selection proposal");
  },{ready.fibers,ready.frames,ready.frame_offsets,ready.frame_fibers,ready.counts,owner,policies,scores,
     out.history.counts,out.history.seen,out.history.last_time,out.history.present,padded,order,out.active,out.branch,error});
  auto compute=p.label(),done=p.label();p.branch(out.branch,{done,compute});p.mark(compute);
  p.softmax(padded,1,probabilities);p.copy(joined.narrow(0,0,capacity*nodes_),probabilities.reshape({-1}));
  p.index_select(joined,0,order,out.controls);p.mark(done);return out;
}
void FrameSelector::append_commit(DeviceProgram& p,const SelectionHistory& history,const SelectionProposal& out,
                                  const at::Tensor& error) const {
  auto zero=at::zeros_like(error),ok=at::zeros({1},history.seen.options()),index=at::zeros_like(error);
  auto check=p.label(),commit=p.label(),done=p.label();p.branch(out.branch,{done,check});p.mark(check);
  p.equal(error,zero,ok);p.cast_index(ok,index);p.branch(index,{done,commit});p.mark(commit);
  p.copy(history.counts,out.history.counts);p.copy(history.seen,out.history.seen);
  p.copy(history.last_time,out.history.last_time);p.copy(history.present,out.history.present);p.mark(done);
}
} // namespace tide::device_online
