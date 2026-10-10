#include "../peer_exchange.h"
#include "program_internal.h"
#include "peer_kernel.h"
#include <mutex>
#include <set>

namespace tide::device_online {
using namespace cuda_backend;
namespace {
std::mutex enable_mutex;
int64_t validate(const PeerExchange::Fields& fields,int64_t budget,uint32_t timeout) {
  if(fields.empty()||budget<1||!timeout||timeout>60000||!fields[0].first.defined()||!fields[0].second.defined())
    throw std::invalid_argument("CUDA peer packet requires fields and bounded positive capacity/timeout");
  auto from=fields[0].first.device(),to=fields[0].second.device();
  if(!from.is_cuda()||!to.is_cuda()||from==to)throw std::invalid_argument("CUDA peer packet requires different CUDA devices");
  int64_t bytes=0;std::vector<std::pair<uintptr_t,uintptr_t>> destinations;
  for(const auto& [x,y]:fields) {
    if(!x.defined()||!y.defined()||x.device()!=from||y.device()!=to||x.sizes()!=y.sizes()||x.scalar_type()!=y.scalar_type()
        ||!x.is_contiguous()||!y.is_contiguous()||x.requires_grad()||y.requires_grad()||!x.numel())throw std::invalid_argument("invalid CUDA peer field");
    type(x.scalar_type());auto size=x.nbytes();
    if(size>uint64_t(budget-bytes))throw std::invalid_argument("CUDA peer packet exceeds byte capacity");bytes+=size;
    auto first=reinterpret_cast<uintptr_t>(y.const_data_ptr());
    if(size>std::numeric_limits<uintptr_t>::max()-first)throw std::overflow_error("CUDA peer address overflow");
    auto last=first+size;for(auto [a,b]:destinations)if(first<b&&a<last)throw std::invalid_argument("overlapping CUDA peer destinations");
    destinations.emplace_back(first,last);
  }
  return bytes;
}
void enable(at::Device from,at::Device to) {
  validate_kernel_device(from);validate_kernel_device(to);c10::cuda::CUDAGuard guard(from);
  int can=0,atomics=0;check(cudaDeviceCanAccessPeer(&can,from.index(),to.index()),"query CUDA peer access");
  check(cudaDeviceGetP2PAttribute(&atomics,cudaDevP2PAttrNativeAtomicSupported,from.index(),to.index()),"query CUDA system peer atomics");
  if(!can||!atomics)throw std::invalid_argument("CUDA resident peers require bidirectional access and native system atomics");
  std::lock_guard<std::mutex> lock(enable_mutex);
  auto result=cudaDeviceEnablePeerAccess(to.index(),0);
  if(result==cudaErrorPeerAccessAlreadyEnabled)(void)cudaGetLastError();else check(result,"enable CUDA peer access");
}
}
struct PeerExchange::Impl {
  portable_torch::RuntimeResource resource;
  Fields fields;
  at::Tensor flags,table;
  int64_t bytes;
  uint64_t timeout;
  Impl(Fields f,int64_t b,uint32_t t):fields(std::move(f)),bytes(b),timeout(uint64_t(t)*1000000) {
    auto from=fields[0].first.device(),to=fields[0].second.device();enable(from,to);enable(to,from);
    {c10::cuda::CUDAGuard guard(from);flags=at::zeros({2},at::TensorOptions().device(from).dtype(at::kLong));
      // Construction boundary, before either graph can reference the mailbox.
      wait_stream(c10::cuda::getCurrentCUDAStream(from.index()).stream(),10000);}
    std::vector<PeerField> layout;
    for(const auto& [x,y]:fields)layout.push_back({static_cast<const uint8_t*>(x.const_data_ptr()),static_cast<uint8_t*>(y.data_ptr()),x.nbytes()});
    auto cpu=at::from_blob(layout.data(),{int64_t(layout.size()*sizeof(PeerField))},at::kByte);
    {c10::cuda::CUDAGuard guard(to);table=cpu.to(to);wait_stream(c10::cuda::getCurrentCUDAStream(to.index()).stream(),10000);}
  }
};
PeerExchange::PeerExchange(Fields fields,int64_t budget,uint32_t timeout) {
  const auto bytes=validate(fields,budget,timeout);impl_=std::make_shared<Impl>(std::move(fields),bytes,timeout);
}
PeerExchange::~PeerExchange()=default;
void PeerExchange::append_send(DeviceProgram& p) {
  if(!impl_)throw std::logic_error("CUDA peer packet closed");auto owner=impl_;owner->resource.check();
  std::vector<at::Tensor> buffers{owner->flags};for(const auto& f:owner->fields)buffers.push_back(f.first);
  p.kernel([owner](void* s){peer_send(static_cast<cudaStream_t>(s),reinterpret_cast<uint64_t*>(owner->flags.data_ptr()),owner->timeout);},buffers);
}
void PeerExchange::append_receive(DeviceProgram& p) {
  if(!impl_)throw std::logic_error("CUDA peer packet closed");auto owner=impl_;owner->resource.check();
  std::vector<at::Tensor> buffers{owner->table};for(const auto& f:owner->fields)buffers.push_back(f.second);
  p.kernel([owner](void* s){peer_receive(static_cast<cudaStream_t>(s),reinterpret_cast<uint64_t*>(owner->flags.data_ptr()),
    reinterpret_cast<const PeerField*>(owner->table.const_data_ptr()),owner->fields.size(),owner->timeout);},buffers);
}
int64_t PeerExchange::packet_bytes() const {if(!impl_)throw std::logic_error("CUDA peer packet closed");return impl_->bytes;}
void PeerExchange::close(){if(!impl_)return;if(impl_.use_count()!=1)throw std::logic_error("CUDA peer packet retained by programs");impl_->resource.close();impl_.reset();}
}
