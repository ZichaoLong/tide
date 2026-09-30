#include "queue_transaction.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_queue_propose.h"
#include <ATen/core/grad_mode.h>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* address(const at::Tensor& value){return static_cast<uint8_t*>(value.data_ptr());}
void buffer(const at::Tensor& t,at::IntArrayRef shape,at::ScalarType dtype,at::Device device) {
  if(!t.defined()||t.sizes()!=shape||t.scalar_type()!=dtype||t.device()!=device
      ||!t.is_contiguous()||t.requires_grad())
    throw std::invalid_argument("invalid contiguous device queue buffer");
}
}
QueueTransaction::QueueTransaction(int64_t capacity,int64_t width,int64_t nodes,int64_t samples,
                                 at::TensorOptions opts,const at::Tensor& shared_error)
    :capacity_(capacity),width_(width),nodes_(nodes),samples_(samples) {
  if(at::GradMode::is_enabled()||opts.device().type()!=c10::DeviceType::PrivateUse1)
    throw std::invalid_argument("device queue transaction requires NPU and explicit no-grad");
  if(capacity<1||width<1||nodes<1||samples<1
      ||capacity>std::numeric_limits<int64_t>::max()/8/std::max<int64_t>(6,width))
    throw std::invalid_argument("invalid device queue dimensions");
  const auto dtype=c10::typeMetaToScalarType(opts.dtype());
  if(dtype!=at::kFloat&&dtype!=at::kHalf)
    throw std::invalid_argument("device queue payload requires explicit FP32/FP16");
  CannApi api;
  auto soc=CannApi::symbol<const char*(*)()>(api.runtime,"aclrtGetSocName")();
  if(!soc||std::string(soc)!=TIDE_ASCENDC_SOC)
    throw std::runtime_error("device queue kernel differs from actual SoC");
  atoms_={at::zeros({capacity,6},opts.dtype(at::kLong)),at::zeros({capacity,width},opts),
          at::zeros({capacity},opts.dtype(at::kBool))};
  if(shared_error.defined()) {buffer(shared_error,{1},at::kInt,opts.device());error_=shared_error;}
  else error_=at::zeros({1},opts.dtype(at::kInt));
  stats_=at::zeros({2},opts.dtype(at::kLong));
}
QueueProposal QueueTransaction::propose_stage(CannProgram& program,const at::Tensor& consumed,const AtomBatch& in) {
  const auto device=atoms_.values.device();
  buffer(consumed,{capacity_},at::kInt,device);
  if(!in.coordinates.defined()||in.coordinates.dim()!=2||in.coordinates.size(0)<1)
    throw std::invalid_argument("incoming device queue requires at least one physical slot");
  const auto arrivals=in.coordinates.size(0);
  const auto max=std::numeric_limits<int64_t>::max();
  if(arrivals>max/8/std::max<int64_t>(6,width_)-capacity_-1)
    throw std::invalid_argument("device queue scratch size overflow");
  buffer(in.coordinates,{arrivals,6},at::kLong,device);
  buffer(in.values,{arrivals,width_},atoms_.values.scalar_type(),device);
  buffer(in.valid,{arrivals},at::kBool,device);
  const auto longs=atoms_.coordinates.options();
  auto coords=at::empty_like(atoms_.coordinates),valid=at::empty_like(atoms_.valid);
  auto order=at::empty({capacity_},longs),stats=at::empty_like(stats_);
  auto branch=at::zeros({1},error_.options());
  // Last row is an explicit zero sentinel. Inactive payloads, including NaNs,
  // are never selected to fill unused output slots.
  auto joined=at::zeros({capacity_+arrivals+1,width_},atoms_.values.options());
  const auto old=atoms_;const auto error=error_,old_stats=stats_;
  const auto capacity=capacity_,nodes=nodes_,samples=samples_;
  program.kernel([=](void* stream) {
    CannApi::check(ACLRT_LAUNCH_KERNEL(tide_queue_propose)(1,stream,address(old.coordinates),
      address(old.valid),address(consumed),address(in.coordinates),address(in.valid),
      address(coords),address(valid),address(order),address(old_stats),address(stats),
      address(branch),address(error),capacity,arrivals,nodes,samples),"propose device queue transaction");
  },{old.coordinates,old.valid,consumed,in.coordinates,in.valid,coords,valid,order,old_stats,stats,branch,error});
  const auto snapshot=program.label(),done=program.label();
  program.branch(branch,{done,snapshot});program.mark(snapshot);
  program.copy(joined.narrow(0,0,capacity_),old.values);
  program.copy(joined.narrow(0,capacity_,arrivals),in.values);
  program.mark(done);
  QueueProposal proposal;proposal.old_=old;proposal.incoming_=in;
  proposal.proposed_={coords,{},valid};proposal.order_=order;proposal.stats_=stats;proposal.joined_=joined;
  return proposal;
}
void QueueTransaction::commit_stage(CannProgram& program,const QueueProposal& q) {
  if(!q.old_.values.defined()||q.old_.values.unsafeGetTensorImpl()!=atoms_.values.unsafeGetTensorImpl())
    throw std::invalid_argument("queue proposal belongs to a different queue");
  auto zero=at::zeros_like(error_),ok=at::zeros({1},atoms_.valid.options()),branch=at::zeros_like(error_);
  program.equal(error_,zero,ok);program.cast_index(ok,branch);
  const auto commit=program.label(),done=program.label();program.branch(branch,{done,commit});program.mark(commit);
  program.index_select(q.joined_,0,q.order_,atoms_.values);
  program.copy(atoms_.coordinates,q.proposed_.coordinates);program.copy(atoms_.valid,q.proposed_.valid);
  program.copy(stats_,q.stats_);program.mark(done);
}
void QueueTransaction::append_stage(CannProgram& program,const at::Tensor& consumed,const AtomBatch& in) {
  auto proposal=propose_stage(program,consumed,in);commit_stage(program,proposal);
}
} // namespace tide::device_online
