#include "device_backend.h"
#include "broadcast_router.h"
#include "device_launch_tide_broadcast_route.h"
#include <ATen/core/grad_mode.h>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* address(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void validate(const at::Tensor& x,at::IntArrayRef shape,at::ScalarType dtype,at::Device device) {
  if(!x.defined()||x.sizes()!=shape||x.scalar_type()!=dtype||x.device()!=device
      ||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid device broadcast buffer");
}
}
BroadcastRouter::BroadcastRouter(int64_t nodes,int64_t samples,const std::vector<Wire>& edges,
                                 int64_t capacity,at::Device device)
    :nodes_(nodes),samples_(samples),edges_(edges.size()),capacity_(capacity),device_(device) {
  if(nodes<1||samples<1||capacity<1||nodes>=std::numeric_limits<int64_t>::max()/8
      ||capacity>std::numeric_limits<int64_t>::max()/48||at::GradMode::is_enabled()
      ||device.type()!=tide::device_online::resident_device_type)
    throw std::invalid_argument("broadcast router requires bounded dimensions, NPU and no-grad");
  validate_kernel_device(device);
  std::vector<std::vector<int64_t>> outgoing(nodes);
  std::vector<int64_t> offsets{0},order,targets,delays;
  for(int64_t e=0;e<edges_;++e) {
    const auto [source,target,delay]=edges[e];
    if(source<0||source>=nodes||target<0||target>=nodes||delay<1)
      throw std::invalid_argument("invalid positive-delay physical wire");
    outgoing[source].push_back(e);targets.push_back(target);delays.push_back(delay);
  }
  for(const auto& group:outgoing) {order.insert(order.end(),group.begin(),group.end());offsets.push_back(order.size());}
  if(edges_==0){order.push_back(0);targets.push_back(0);delays.push_back(1);}
  offsets_=at::tensor(offsets,at::kLong).to(device);edges_by_source_=at::tensor(order,at::kLong).to(device);
  targets_=at::tensor(targets,at::kLong).to(device);delays_=at::tensor(delays,at::kLong).to(device);
}
AtomBatch BroadcastRouter::append_stage(DeviceProgram& p,const ActionBatch& action,
                                       const at::Tensor& scales,const at::Tensor& error) const {
  if(!action.coordinates.defined()||action.coordinates.dim()!=2||action.coordinates.size(0)<1
      ||!action.values.defined()||action.values.dim()!=2||action.values.size(1)<1)
    throw std::invalid_argument("broadcast requires nonempty physical action slots");
  const auto rows=action.coordinates.size(0),width=action.values.size(1);
  const auto dtype=action.values.scalar_type();
  if(dtype!=at::kFloat&&dtype!=at::kHalf)throw std::invalid_argument("broadcast requires FP32/FP16 payloads");
  const auto maximum=std::numeric_limits<int64_t>::max();
  if(rows>=maximum/8/std::max<int64_t>(width,4)||capacity_>maximum/8/width)
    throw std::invalid_argument("broadcast scratch overflow");
  validate(action.coordinates,{rows,4},at::kLong,device_);validate(action.values,{rows,width},dtype,device_);
  validate(action.valid,{rows},at::kBool,device_);validate(scales,{std::max<int64_t>(1,edges_),1},dtype,device_);
  validate(error,{1},at::kInt,device_);
  const auto longs=action.coordinates.options();
  AtomBatch out{at::empty({capacity_,6},longs),at::zeros({capacity_,width},action.values.options()),
                at::zeros({capacity_},action.valid.options())};
  auto value_order=at::empty({capacity_},longs),scale_order=at::empty({capacity_},longs);
  auto branch=at::zeros_like(error);
  auto values=at::zeros({rows+1,width},action.values.options());
  auto scale_values=at::zeros({std::max<int64_t>(1,edges_)+1,1},scales.options());
  auto selected=at::empty_like(out.values),selected_scales=at::empty({capacity_,1},scales.options());
  const auto offsets=offsets_,edge_ids=edges_by_source_,targets=targets_,delays=delays_;
  const auto capacity=capacity_,nodes=nodes_,samples=samples_,edge_count=edges_;
  p.kernel([=](void* stream) {
    check_device_launch(TIDE_LAUNCH_KERNEL(tide_broadcast_route)(1,stream,address(action.coordinates),
      address(action.valid),address(offsets),address(edge_ids),address(targets),address(delays),
      address(out.coordinates),address(out.valid),address(value_order),address(scale_order),
      address(branch),address(error),rows,capacity,nodes,samples,edge_count),"device broadcast route");
  },{action.coordinates,action.valid,offsets,edge_ids,targets,delays,out.coordinates,out.valid,value_order,scale_order,branch,error});
  auto emit=p.label(),done=p.label();p.branch(branch,{done,emit});p.mark(emit);
  p.copy(values.narrow(0,0,rows),action.values);
  p.copy(scale_values.narrow(0,0,std::max<int64_t>(1,edges_)),scales);
  p.index_select(values,0,value_order,selected);p.index_select(scale_values,0,scale_order,selected_scales);
  p.multiply(selected,selected_scales,out.values);p.mark(done);return out;
}
} // namespace tide::device_online
