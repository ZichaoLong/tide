#include "cann_sequence.h"
#include "cann_api.h"
#include "peer_api.h"
#include "portable_torch/runtime.hpp"
#include <c10/core/DeviceGuard.h>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace tide::device_online {
namespace {
struct Signal {
  portable_torch::RuntimeResource resource;
  PeerApi api;
  at::Device device;
  void* handle=nullptr;
  explicit Signal(at::Device d):device(d) {
    c10::DeviceGuard guard(d);CannApi::check(api.create(&handle,0),"create local program completion");
  }
  void close() {
    if(handle){c10::DeviceGuard guard(device);CannApi::check(api.destroy(handle),"destroy local program completion");handle=nullptr;}
    resource.close();
  }
  ~Signal() {
    try{close();}catch(const std::exception& e){resource.quarantine();std::cerr<<"program completion quarantined: "<<e.what()<<'\n';}
  }
};
}
struct CannSequence::Impl {
  at::Device device;
  int64_t capacity,workspace;
  const std::thread::id thread=std::this_thread::get_id();
  // Programs release their callbacks before the final signal owner. A failed
  // program drain quarantines its callbacks, which keep its signal alive.
  std::vector<std::shared_ptr<Signal>> signals;
  std::vector<std::unique_ptr<CannProgram>> programs;
  bool finished=false,closed=false,failed=false,in_flight=false;
  Impl(at::Device d,int64_t n,int64_t w):device(d),capacity(n),workspace(w){}
  void check() const {
    if(thread!=std::this_thread::get_id())throw std::logic_error("program sequence requires its constructing thread");
    if(closed||failed)throw std::logic_error("program sequence is closed or failed");
  }
};
CannSequence::CannSequence(at::Device d,int64_t capacity,int64_t workspace) {
  if(d.type()!=c10::DeviceType::PrivateUse1||d.index()<0||capacity<1||workspace<1)
    throw std::invalid_argument("program sequence requires explicit NPU and positive finite limits");
  impl_=std::make_unique<Impl>(d,capacity,workspace);
}
CannSequence::~CannSequence()=default;
CannProgram& CannSequence::append() {
  auto& s=*impl_;s.check();if(s.finished)throw std::logic_error("program sequence is already finished");
  if(s.programs.size()>=size_t(s.capacity))throw std::invalid_argument("program sequence capacity exceeded");
  try {
    auto p=std::make_unique<CannProgram>(s.device);p->limit_workspace(s.workspace);
    if(!s.programs.empty()) {
      auto signal=std::make_shared<Signal>(s.device);s.signals.push_back(signal);
      s.programs.back()->kernel([signal](void* stream){signal->resource.check();
        CannApi::check(signal->api.record(signal->handle,stream),"record preceding program completion");},{});
      p->kernel([signal](void* stream){signal->resource.check();
        CannApi::check(signal->api.wait_reset(signal->handle,stream,10000),"await preceding program completion");},{});
    }
    s.programs.push_back(std::move(p));return *s.programs.back();
  }catch(...){s.failed=true;throw;}
}
void CannSequence::finish() {
  auto& s=*impl_;s.check();if(s.finished||s.programs.empty())throw std::logic_error("invalid program sequence finish");
  try{for(auto& p:s.programs)p->finish();s.finished=true;}catch(...){s.failed=true;throw;}
}
void CannSequence::submit() {
  auto& s=*impl_;s.check();if(!s.finished||s.in_flight)throw std::logic_error("program sequence is not executable");
  try{s.in_flight=true;for(auto& p:s.programs)p->submit();}catch(...){s.failed=true;throw;}
}
void CannSequence::wait(int32_t timeout) {
  auto& s=*impl_;s.check();if(!s.in_flight||timeout<=0)throw std::logic_error("program sequence has no waitable submission");
  std::exception_ptr failure;
  for(auto& p:s.programs)try{p->wait(timeout);}catch(...){if(!failure)failure=std::current_exception();}
  if(failure){s.failed=true;std::rethrow_exception(failure);}s.in_flight=false;
}
void CannSequence::run(int32_t timeout) {
  if(timeout<=0)throw std::invalid_argument("program sequence wait must be positive");submit();wait(timeout);
}
void CannSequence::close() {
  auto& s=*impl_;
  if(s.thread!=std::this_thread::get_id())throw std::logic_error("program sequence requires its constructing thread");
  if(s.closed)return;
  try {
    for(auto& p:s.programs)p->close();s.programs.clear();
    for(auto& signal:s.signals)signal->close();s.signals.clear();s.closed=true;s.in_flight=false;
  }catch(...){s.failed=true;throw;}
}
size_t CannSequence::size() const {return impl_->programs.size();}
} // namespace tide::device_online
