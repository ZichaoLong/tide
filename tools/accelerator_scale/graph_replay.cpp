#include "graph_replay.h"
#include "peer_transport.h"
#include <stdexcept>
#if PORTABLE_TORCH_ENABLE_NPU
#include <torch_npu/csrc/core/npu/NPUGraph.h>
#include <c10/core/DeviceGuard.h>
#endif

namespace accelerator_scale {
struct GraphReplay::Impl {
#if PORTABLE_TORCH_ENABLE_NPU
  std::vector<at::Device> devices;
  std::vector<c10_npu::NPUStream> streams,previous;
  std::vector<std::unique_ptr<c10_npu::NPUGraph>> graphs;
  std::unique_ptr<PeerTransport> peer;
  explicit Impl(const std::vector<at::Device>& ds):devices(ds) {
    for(auto d:devices) {
      if(d.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("graph replay requires NPUs");
      c10::DeviceGuard guard(d);previous.push_back(c10_npu::getCurrentNPUStream(d.index()));
      streams.push_back(c10_npu::getNPUStreamFromPool(d.index()));c10_npu::setCurrentNPUStream(streams.back());
      graphs.push_back(std::make_unique<c10_npu::NPUGraph>());
    }
    if(devices.size()>1)peer=std::make_unique<PeerTransport>(devices);
  }
  ~Impl() {
    // Each captured model belongs to its original device context.
    for(size_t i=0;i<graphs.size();++i){c10::DeviceGuard guard(devices[i]);graphs[i]->reset();}
    peer.reset();
    for(auto s:previous)c10_npu::setCurrentNPUStream(s);
  }
#endif
  bool captured=false;
};
GraphReplay::GraphReplay(at::Device d):GraphReplay(std::vector<at::Device>{d}){}
GraphReplay::GraphReplay(const std::vector<at::Device>& devices) {
#if PORTABLE_TORCH_ENABLE_NPU
  if(devices.empty())throw std::invalid_argument("graph replay requires a device inventory");
  impl_=std::make_unique<Impl>(devices);
#else
  throw std::invalid_argument("this build has no NPU graph replay adapter");
#endif
}
GraphReplay::~GraphReplay() = default;
void GraphReplay::capture(const std::function<void()>& fn) {
#if PORTABLE_TORCH_ENABLE_NPU
  if(impl_->captured)throw std::logic_error("capture requires a new graph");
  synchronize();if(impl_->peer)impl_->peer->prepare_capture();size_t begun=0;
  try {
    for(size_t i=0;i<impl_->graphs.size();++i) {
      c10::DeviceGuard guard(impl_->devices[i]);
      impl_->graphs[i]->capture_begin({0,0},aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);++begun;
    }
    fn();
    while(begun){const auto i=--begun;c10::DeviceGuard guard(impl_->devices[i]);impl_->graphs[i]->capture_end();}
    if(impl_->peer)impl_->peer->finish_capture();impl_->captured=true;
  }catch(...) {
    while(begun){const auto i=--begun;try{c10::DeviceGuard guard(impl_->devices[i]);impl_->graphs[i]->capture_end();}catch(...) {}}
    throw;
  }
  synchronize();
#else
  throw std::logic_error("NPU graph unavailable");
#endif
}
void GraphReplay::replay() {
#if PORTABLE_TORCH_ENABLE_NPU
  if(!impl_->captured)throw std::logic_error("capture before replay");
  for(size_t i=0;i<impl_->graphs.size();++i){c10::DeviceGuard guard(impl_->devices[i]);impl_->graphs[i]->replay();}
#else
  throw std::logic_error("NPU graph unavailable");
#endif
}
void GraphReplay::synchronize() {
#if PORTABLE_TORCH_ENABLE_NPU
  for(auto& stream:impl_->streams)stream.synchronize();
#endif
}
std::pair<size_t,int64_t> GraphReplay::peer_inventory() const {
#if PORTABLE_TORCH_ENABLE_NPU
  if(impl_->peer)return impl_->peer->inventory();
#endif
  return {0,0};
}
}  // namespace accelerator_scale
