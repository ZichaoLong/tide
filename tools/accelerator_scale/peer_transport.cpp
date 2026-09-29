#include "peer_transport.h"
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <ATen/core/grad_mode.h>
#if PORTABLE_TORCH_ENABLE_NPU
#include <torch/csrc/autograd/autograd.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <c10/core/DeviceGuard.h>
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace accelerator_scale {
namespace {
std::atomic<PeerTransport*> active{nullptr};
#if PORTABLE_TORCH_ENABLE_NPU
std::mutex peer_enable_mutex;
std::set<std::pair<int,int>> enabled_pairs;
void checked(int code,const char* name) {
  if(code)throw std::runtime_error(std::string("NPU peer ")+name+": ACL error "+std::to_string(code));
}
// Optional CANN9 runtime API. No toolkit/site paths or compiler Python runtime
// enter the portable core. Missing symbols are an explicit capability failure.
struct Api {
  void* handle=dlopen("libacl_rt.so",RTLD_NOW|RTLD_LOCAL);
  template<class F> F symbol(const char* name) {
    auto p=handle?dlsym(handle,name):nullptr;
    if(!p)throw std::runtime_error(std::string("NPU peer runtime lacks ")+name);
    return reinterpret_cast<F>(p);
  }
  using Notify=void*;
  decltype(&aclrtDeviceCanAccessPeer) can=symbol<decltype(can)>("aclrtDeviceCanAccessPeer");
  decltype(&aclrtDeviceEnablePeerAccess) enable=symbol<decltype(enable)>("aclrtDeviceEnablePeerAccess");
  decltype(&aclrtMemcpyAsync) copy=symbol<decltype(copy)>("aclrtMemcpyAsync");
  int(*create)(Notify*,uint64_t)=symbol<decltype(create)>("aclrtCreateNotify");
  int(*destroy)(Notify)=symbol<decltype(destroy)>("aclrtDestroyNotify");
  int(*export_key)(Notify,char*,size_t,uint64_t)=symbol<decltype(export_key)>("aclrtNotifyGetExportKey");
  int(*whitelist)(Notify,int32_t*,size_t)=symbol<decltype(whitelist)>("aclrtNotifySetImportPid");
  int(*import_key)(Notify*,const char*,uint64_t)=symbol<decltype(import_key)>("aclrtNotifyImportByKey");
  int(*record)(Notify,aclrtStream)=symbol<decltype(record)>("aclrtRecordNotify");
  int(*wait_reset)(Notify,aclrtStream,uint32_t)=symbol<decltype(wait_reset)>("aclrtWaitAndResetNotify");
  ~Api(){if(handle)dlclose(handle);}
};
struct Channel {
  Api& api;at::Device source,destination;void* local=nullptr;void* remote=nullptr;
  Channel(Api& a,at::Device from,at::Device to):api(a),source(from),destination(to) {
    try {
      char key[256]{};
      {c10::DeviceGuard guard(to);checked(api.create(&local,0),"create notify");
       checked(api.export_key(local,key,sizeof(key),0),"export notify");
       int32_t pid=getpid();checked(api.whitelist(local,&pid,1),"notify self permission");}
      {c10::DeviceGuard guard(from);checked(api.import_key(&remote,key,0),"import notify");}
    }catch(...){close();throw;}
  }
  void close() noexcept {
    try{if(remote){c10::DeviceGuard guard(source);api.destroy(remote);remote=nullptr;}
        if(local){c10::DeviceGuard guard(destination);api.destroy(local);local=nullptr;}}catch(...){}
  }
  ~Channel(){close();}
};
#endif
}
struct PeerTransport::Impl {
#if PORTABLE_TORCH_ENABLE_NPU
  Api api;
  std::map<std::string,std::vector<std::unique_ptr<Channel>>> channels;
  std::map<std::string,size_t> cursor;
  std::vector<at::Tensor> retained;
  std::vector<std::pair<at::Tensor,at::Tensor>> bridges;
  std::mutex mutex;
  size_t count=0,capacity;
  bool capturing=false;
  explicit Impl(size_t cap):capacity(cap){}
#endif
};
PeerTransport::PeerTransport(const std::vector<at::Device>& devices,size_t capacity) {
#if PORTABLE_TORCH_ENABLE_NPU
  if(active.load())throw std::logic_error("only one peer replay session may be active");
  impl_=std::make_unique<Impl>(capacity);
  for(auto from:devices)for(auto to:devices)if(from!=to) {
    c10::DeviceGuard guard(from);int can=0;
    checked(impl_->api.can(&can,from.index(),to.index()),"peer capability");
    if(!can)throw std::runtime_error("NPU peer access is unavailable for the requested placement");
    std::lock_guard<std::mutex> lock(peer_enable_mutex);
    const auto pair=std::make_pair(int(from.index()),int(to.index()));
    if(!enabled_pairs.count(pair)) {
      checked(impl_->api.enable(to.index(),0),"enable peer access");enabled_pairs.insert(pair);
    }
  }
  PeerTransport* expected=nullptr;
  if(!active.compare_exchange_strong(expected,this))throw std::logic_error("concurrent peer replay session");
#else
  throw std::invalid_argument("this build has no NPU peer transport");
#endif
}
PeerTransport::~PeerTransport(){PeerTransport* expected=this;active.compare_exchange_strong(expected,nullptr);}
void PeerTransport::prepare_capture() {
#if PORTABLE_TORCH_ENABLE_NPU
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->retained.clear();impl_->bridges.clear();impl_->cursor.clear();impl_->capturing=true;
#endif
}
void PeerTransport::finish_capture() {
#if PORTABLE_TORCH_ENABLE_NPU
  for(const auto& [key,pool]:impl_->channels)
    if(impl_->cursor.at(key)!=pool.size())throw std::runtime_error("peer capture transfer inventory differs from warmup");
#endif
}
std::pair<size_t,int64_t> PeerTransport::inventory() const {
#if PORTABLE_TORCH_ENABLE_NPU
  std::lock_guard<std::mutex> lock(impl_->mutex);int64_t bytes=0;
  for(const auto& tensor:impl_->retained)bytes+=tensor.nbytes();
  return {impl_->count,bytes};
#else
  return {0,0};
#endif
}
at::Tensor PeerTransport::copy(const at::Tensor& input,at::Device destination) {
#if PORTABLE_TORCH_ENABLE_NPU
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto source=input.device();if(source==destination)return input;
  if(source.type()!=c10::DeviceType::PrivateUse1 || destination.type()!=source.type())
    throw std::invalid_argument("peer copy requires two NPUs");
  auto key=source.str()+">"+destination.str()+":"+std::to_string(int(input.scalar_type()));
  for(auto dim:input.sizes())key+=":"+std::to_string(dim);
  auto& pool=impl_->channels[key];const auto index=impl_->cursor[key]++;
  if(index==pool.size()) {
    if(impl_->capturing)throw std::runtime_error("peer capture exceeded its warmup transfer inventory");
    if(impl_->count>=impl_->capacity)throw std::runtime_error("bounded peer notification capacity exceeded");
    pool.push_back(std::make_unique<Channel>(impl_->api,source,destination));++impl_->count;
  }
  auto& channel=*pool.at(index);
  // Keep all raw-API buffers alive through the window/capture. The peer API
  // cannot tell Torch's destination allocator about a foreign device stream.
  const bool differentiable=at::GradMode::is_enabled() && input.requires_grad();
  at::NoGradGuard no_grad;at::Tensor x,y;
  {c10::DeviceGuard guard(source);x=input.clone(at::MemoryFormat::Contiguous);}
  {c10::DeviceGuard guard(destination);y=at::empty(input.sizes(),input.options().device(destination));}
  auto src=c10_npu::getCurrentNPUStream(source.index()),dst=c10_npu::getCurrentNPUStream(destination.index());
  {c10::DeviceGuard guard(source);
   checked(impl_->api.record(channel.remote,src.stream()),"record source readiness");}
  {c10::DeviceGuard guard(destination);
   checked(impl_->api.wait_reset(channel.local,dst.stream(),30000),"wait source readiness");
   // Pull on the consumer stream: its copy and subsequent kernels share one
   // device ordering domain. The immutable source clone remains retained until
   // all models finish, so no source-buffer-reuse acknowledgement is needed.
   checked(impl_->api.copy(y.data_ptr(),y.nbytes(),x.const_data_ptr(),x.nbytes(),ACL_MEMCPY_DEVICE_TO_DEVICE,dst.stream()),"async peer pull");}
  if(differentiable){y.set_requires_grad(true);impl_->bridges.emplace_back(input,y);}
  impl_->retained.push_back(x);impl_->retained.push_back(y);return y;
#else
  throw std::logic_error("NPU peer unavailable");
#endif
}
at::Tensor replay_peer_copy(const at::Tensor& x,at::Device d) {
  if(x.device()==d)return x;
#if PORTABLE_TORCH_ENABLE_NPU
  if(active.load() && x.device().type()==c10::DeviceType::PrivateUse1 && d.type()==x.device().type())
    return active.load()->copy(x,d);
#endif
  return x.clone(at::MemoryFormat::Contiguous).to(d);
}
std::vector<at::Tensor> PeerTransport::vjp(const at::Tensor& value,
    const std::vector<at::Tensor>& leaves,const at::Tensor& cotangent,bool retain) {
#if PORTABLE_TORCH_ENABLE_NPU
  // Torch's cross-device autograd edges insert ordinary cross-model Events,
  // which cannot be captured. Differentiate each device-local graph and carry
  // cotangents through the same explicit Notify protocol in reverse tape order.
  // This is a first-order reference implementation; repeated local traversals
  // favor auditability over a fused reverse scheduler.
  using Tensors=std::vector<at::Tensor>;
  std::map<std::string,std::pair<Tensors,Tensors>> roots;
  roots[value.device().str()]={{value},{cotangent}};
  const auto bridges=impl_->bridges;
  for(auto it=bridges.rbegin();it!=bridges.rend();++it) {
    const auto& [source,destination]=*it;auto found=roots.find(destination.device().str());
    if(found==roots.end())continue;
    auto& [outputs,weights]=found->second;c10::DeviceGuard guard(destination.device());
    auto gradient=torch::autograd::grad(outputs,{destination},weights,true,false,true)[0];
    if(!gradient.defined())continue; // static autograd connectivity, no tensor decision
    at::NoGradGuard no_grad;auto moved=copy(gradient,source.device());
    auto& upstream=roots[source.device().str()];upstream.first.push_back(source);upstream.second.push_back(moved);
  }
  std::vector<at::Tensor> result(leaves.size());std::map<std::string,std::vector<size_t>> owners;
  for(size_t i=0;i<leaves.size();++i)owners[leaves[i].device().str()].push_back(i);
  for(const auto& [device,indices]:owners) {
    auto found=roots.find(device);if(found==roots.end())continue;
    Tensors inputs;for(auto i:indices)inputs.push_back(leaves[i]);
    c10::DeviceGuard guard(inputs[0].device());auto& [outputs,weights]=found->second;
    auto gradients=torch::autograd::grad(outputs,inputs,weights,retain,false,true);
    for(size_t i=0;i<indices.size();++i)result[indices[i]]=gradients[i];
  }
  return result;
#else
  throw std::logic_error("NPU peer VJP unavailable");
#endif
}
std::optional<std::vector<at::Tensor>> replay_peer_vjp(const at::Tensor& value,
    const std::vector<at::Tensor>& leaves,const at::Tensor& cotangent,bool retain) {
  auto* session=active.load();if(!session)return std::nullopt;
  return session->vjp(value,leaves,cotangent,retain);
}
}  // namespace accelerator_scale
