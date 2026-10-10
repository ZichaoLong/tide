#include "peer_exchange.h"
#include "peer_api.h"
#include "cann_api.h"
#include "portable_torch/runtime.hpp"
#include <c10/core/DeviceGuard.h>
#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <unistd.h>

namespace tide::device_online {
namespace {
std::mutex enable_mutex;
std::set<std::pair<int,int>> enabled;
std::atomic<bool> quarantined{false};
void live() {
  if(quarantined.load())throw std::logic_error("CANN peer resources quarantined; terminate worker");
}
int64_t validate(const PeerExchange::Fields& fields,int64_t budget,uint32_t timeout) {
  live();
  if(fields.empty()||budget<1||!timeout||timeout>60000)
    throw std::invalid_argument("peer packet requires fields, positive capacity and bounded timeout");
  if(!fields[0].first.defined()||!fields[0].second.defined())throw std::invalid_argument("undefined peer field");
  const auto from=fields[0].first.device(),to=fields[0].second.device();
  if(from.type()!=c10::DeviceType::PrivateUse1||to.type()!=from.type()||from==to)
    throw std::invalid_argument("peer packet requires two different NPUs");
  int64_t bytes=0;std::vector<std::pair<uintptr_t,uintptr_t>> destinations;
  for(const auto& [x,y]:fields) {
    if(!x.defined()||!y.defined()||x.device()!=from||y.device()!=to||x.sizes()!=y.sizes()
        ||x.scalar_type()!=y.scalar_type()||!x.is_contiguous()||!y.is_contiguous()
        ||x.requires_grad()||y.requires_grad()||!x.numel())throw std::invalid_argument("invalid fixed peer field");
    const auto dtype=x.scalar_type();
    if(dtype!=at::kFloat&&dtype!=at::kHalf&&dtype!=at::kLong&&dtype!=at::kInt&&dtype!=at::kBool&&dtype!=at::kByte)
      throw std::invalid_argument("unsupported peer field dtype");
    const auto size=x.nbytes();
    if(size>uint64_t(budget-bytes))throw std::invalid_argument("peer packet exceeds byte capacity");
    bytes+=size;
    const auto first=reinterpret_cast<uintptr_t>(y.const_data_ptr());
    if(size>std::numeric_limits<uintptr_t>::max()-first)throw std::overflow_error("peer field address overflow");
    const auto last=first+size;
    for(auto [a,b]:destinations)if(first<b&&a<last)throw std::invalid_argument("overlapping peer destinations");
    destinations.emplace_back(first,last);
  }
  return bytes;
}
}
struct PeerExchange::Impl {
  struct State {
    portable_torch::RuntimeResource resource;
    PeerApi api;
    Fields fields;
    at::Device source,destination;
    int64_t bytes;
    uint32_t timeout;
    void *ready_local=nullptr,*ready_remote=nullptr,*consumed_local=nullptr,*consumed_remote=nullptr;
    State(Fields f,int64_t b,uint32_t t):fields(std::move(f)),source(fields[0].first.device()),
      destination(fields[0].second.device()),bytes(b),timeout(t){}
    void notify(at::Device from,at::Device to,void*& local,void*& remote) {
      char key[256]{};
      {c10::DeviceGuard guard(to);CannApi::check(api.create(&local,0),"create peer notify");
       CannApi::check(api.export_key(local,key,sizeof(key),0),"export peer notify");
       int32_t pid=getpid();CannApi::check(api.whitelist(local,&pid,1),"permit peer notify import");}
      {c10::DeviceGuard guard(from);CannApi::check(api.import_key(&remote,key,0),"import peer notify");}
    }
    void initialize() {
      // A destination stream pulls remote memory; both directions are needed
      // by the reusable ready/consumed protocol and reverse-direction packets.
      for(auto [from,to]:{std::make_pair(source,destination),std::make_pair(destination,source)}) {
        c10::DeviceGuard guard(from);int32_t can=0;
        CannApi::check(api.can(&can,from.index(),to.index()),"query peer access");
        if(!can)throw std::invalid_argument("requested NPUs have no peer access");
        std::lock_guard<std::mutex> lock(enable_mutex);const auto pair=std::make_pair(int(from.index()),int(to.index()));
        if(!enabled.count(pair)){CannApi::check(api.enable(to.index(),0),"enable peer access");enabled.insert(pair);}
      }
      notify(source,destination,ready_local,ready_remote);
      notify(destination,source,consumed_local,consumed_remote);
    }
    void release() {
      auto destroy=[&](at::Device device,void*& handle) {
        if(handle){c10::DeviceGuard guard(device);CannApi::check(api.destroy(handle),"destroy peer notify");handle=nullptr;}
      };
      destroy(source,ready_remote);destroy(destination,ready_local);
      destroy(destination,consumed_remote);destroy(source,consumed_local);
      fields.clear();resource.close();
    }
  };
  std::unique_ptr<State> state;
  Impl(Fields fields,int64_t bytes,uint32_t timeout):state(std::make_unique<State>(std::move(fields),bytes,timeout)) {
    try {state->initialize();}catch(...){release_noexcept();throw;}
  }
  void release_noexcept() noexcept {
    if(!state)return;
    try {state->release();state.reset();}
    catch(const std::exception& e) {
      std::cerr<<"CANN peer resources quarantined until process exit: "<<e.what()<<'\n';
      quarantined.store(true);state->resource.quarantine();(void)state.release();
    }
  }
  ~Impl(){release_noexcept();}
};
PeerExchange::PeerExchange(Fields fields,int64_t budget,uint32_t timeout) {
  const auto bytes=validate(fields,budget,timeout);
  impl_=std::make_shared<Impl>(std::move(fields),bytes,timeout);
}
PeerExchange::~PeerExchange()=default;
void PeerExchange::append_send(DeviceProgram& p) {
  live();if(!impl_)throw std::logic_error("peer packet is closed");
  auto owner=impl_;auto& s=*owner->state;s.resource.check();std::vector<at::Tensor> tensors;
  for(const auto& f:s.fields)tensors.push_back(f.first);
  p.kernel([owner](void* stream){auto& s=*owner->state;
    CannApi::check(s.api.record(s.ready_remote,stream),"record peer ready");
    CannApi::check(s.api.wait_reset(s.consumed_local,stream,s.timeout),"wait peer consumption");
  },tensors);
}
void PeerExchange::append_receive(DeviceProgram& p) {
  live();if(!impl_)throw std::logic_error("peer packet is closed");
  auto owner=impl_;auto& s=*owner->state;s.resource.check();std::vector<at::Tensor> tensors;
  for(const auto& f:s.fields)tensors.push_back(f.second);
  p.kernel([owner](void* stream){auto& s=*owner->state;
    CannApi::check(s.api.wait_reset(s.ready_local,stream,s.timeout),"wait peer ready");
    for(const auto& [x,y]:s.fields) // Fixed packet fields; this is construction, not an event loop.
      CannApi::check(s.api.copy(y.data_ptr(),y.nbytes(),x.const_data_ptr(),x.nbytes(),3,stream),"copy peer packet");
    CannApi::check(s.api.record(s.consumed_remote,stream),"record peer consumption");
  },tensors);
}
int64_t PeerExchange::packet_bytes() const {
  if(!impl_)throw std::logic_error("peer packet is closed");return impl_->state->bytes;
}
void PeerExchange::close() {
  if(!impl_)return;
  if(impl_.use_count()!=1)throw std::logic_error("peer packet still retained by runtime programs");
  impl_->state->release();impl_.reset();
}
} // namespace tide::device_online
