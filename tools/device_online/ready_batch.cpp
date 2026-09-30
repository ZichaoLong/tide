#include "ready_batch.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_closure.h"
#include "aclrtlaunch_tide_ready_pack.h"
#include <ATen/core/grad_mode.h>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* address(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void validate(const at::Tensor& x,at::IntArrayRef shape,at::ScalarType dtype,at::Device device) {
  if(!x.defined()||x.sizes()!=shape||x.scalar_type()!=dtype||x.device()!=device
      ||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid device ready buffer");
}
}
DeviceReady::DeviceReady(const std::vector<int64_t>& owners,int64_t regions,const std::vector<Wire>& wires,
                         int64_t samples,at::Device device,bool prefill)
    :nodes_(owners.size()),regions_(regions),samples_(samples),device_(device),prefill_(prefill),
     topology_(owners,regions,wires,samples,device) {
  if(at::GradMode::is_enabled()||device.type()!=c10::DeviceType::PrivateUse1)
    throw std::invalid_argument("device ready stage requires NPU and explicit no-grad");
  CannApi api;auto soc=CannApi::symbol<const char*(*)()>(api.runtime,"aclrtGetSocName")();
  if(!soc||std::string(soc)!=TIDE_ASCENDC_SOC)throw std::runtime_error("ready kernel differs from actual SoC");
}
ReadyBatch DeviceReady::append_stage(CannProgram& p,const AtomBatch& q,const at::Tensor& stop,
                                    const at::Tensor& error) const {
  if(!q.coordinates.defined()||q.coordinates.dim()!=2||q.coordinates.size(0)<1
      ||!q.values.defined()||q.values.dim()!=2||q.values.size(1)<1)
    throw std::invalid_argument("ready stage requires nonempty physical queue");
  const auto capacity=q.coordinates.size(0),width=q.values.size(1);
  const auto dtype=q.values.scalar_type();
  if((dtype!=at::kFloat&&dtype!=at::kHalf)||capacity>=std::numeric_limits<int64_t>::max()/8/std::max<int64_t>(6,width))
    throw std::invalid_argument("ready stage dtype/dimensions unavailable");
  validate(q.coordinates,{capacity,6},at::kLong,device_);validate(q.values,{capacity,width},dtype,device_);
  validate(q.valid,{capacity},at::kBool,device_);validate(stop,{1},at::kLong,device_);validate(error,{1},at::kInt,device_);
  const auto longs=q.coordinates.options();ReadyBatch out;
  out.atoms={at::empty_like(q.coordinates),at::zeros_like(q.values),at::zeros_like(q.valid)};
  out.consumed=at::zeros({capacity},error.options());out.branch=at::zeros_like(error);
  out.fiber_offsets=at::zeros({capacity+1},longs);out.fibers=at::zeros({capacity,4},longs);
  out.frame_offsets=at::zeros({capacity+1},longs);out.frame_fibers=at::zeros({capacity},longs);
  out.frames=at::zeros({capacity,3},longs);out.counts=at::zeros({3},longs);
  auto work=at::empty({samples_*regions_*2+samples_},longs),order=at::empty({capacity},longs);
  auto joined=at::zeros({capacity+1,width},q.values.options());
  const auto owners=topology_.owners(),distances=topology_.distances();
  const auto nodes=nodes_,regions=regions_,samples=samples_;const int64_t prefill=prefill_;
  p.kernel([=](void* stream) {
    CannApi::check(ACLRT_LAUNCH_KERNEL(tide_closure)(1,stream,address(q.coordinates),address(q.valid),
      address(owners),address(distances),address(work),address(stop),address(out.consumed),
      address(out.branch),address(error),capacity,nodes,regions,samples,prefill),"device batch closure");
  },{q.coordinates,q.valid,owners,distances,work,stop,out.consumed,out.branch,error});
  p.kernel([=](void* stream) {
    CannApi::check(ACLRT_LAUNCH_KERNEL(tide_ready_pack)(1,stream,address(q.coordinates),address(q.valid),
      address(out.consumed),address(owners),address(order),address(out.atoms.coordinates),address(out.atoms.valid),
      address(out.fiber_offsets),address(out.fibers),address(out.frame_offsets),address(out.frame_fibers),
      address(out.frames),address(out.counts),address(out.branch),address(error),capacity),"pack device ready fibers");
  },{q.coordinates,q.valid,out.consumed,owners,order,out.atoms.coordinates,out.atoms.valid,out.fiber_offsets,
     out.fibers,out.frame_offsets,out.frame_fibers,out.frames,out.counts,out.branch,error});
  auto pack=p.label(),done=p.label();p.branch(out.branch,{done,pack});p.mark(pack);
  p.copy(joined.narrow(0,0,capacity),q.values);p.index_select(joined,0,order,out.atoms.values);
  p.mark(done);return out;
}
} // namespace tide::device_online
