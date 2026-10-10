#include "../device_sequence.h"
#include <thread>
#include <stdexcept>

namespace tide::device_online {
struct DeviceSequence::Impl {
  at::Device device;
  int64_t capacity,workspace;
  const std::thread::id thread=std::this_thread::get_id();
  std::vector<std::unique_ptr<DeviceProgram>> programs;
  bool finished=false,closed=false,failed=false,in_flight=false;
  Impl(at::Device d,int64_t n,int64_t w):device(d),capacity(n),workspace(w){}
  void check() const {
    if(thread!=std::this_thread::get_id())throw std::logic_error("CUDA sequence requires its constructing thread");
    if(closed||failed)throw std::logic_error("CUDA sequence is closed or failed");
  }
};
DeviceSequence::DeviceSequence(at::Device d,int64_t capacity,int64_t workspace) {
  if(!d.is_cuda()||d.index()<0||capacity<1||workspace<1)throw std::invalid_argument("CUDA sequence requires explicit device and positive capacities");
  impl_=std::make_unique<Impl>(d,capacity,workspace);
}
DeviceSequence::~DeviceSequence()=default;
DeviceProgram& DeviceSequence::append() {
  auto& s=*impl_;s.check();if(s.finished||s.programs.size()>=size_t(s.capacity))throw std::logic_error("CUDA sequence cannot append");
  try {
    auto p=std::make_unique<DeviceProgram>(s.device);p->limit_workspace(s.workspace);
    if(!s.programs.empty())p->order_after(*s.programs.back());
    s.programs.push_back(std::move(p));return *s.programs.back();
  }catch(...){s.failed=true;throw;}
}
void DeviceSequence::finish() {
  auto& s=*impl_;s.check();if(s.finished||s.programs.empty())throw std::logic_error("invalid CUDA sequence finish");
  try{for(auto& p:s.programs)p->finish();s.finished=true;}catch(...){s.failed=true;throw;}
}
void DeviceSequence::submit() {
  auto& s=*impl_;s.check();if(!s.finished||s.in_flight)throw std::logic_error("CUDA sequence not executable");
  try{s.in_flight=true;for(auto& p:s.programs)p->submit();}catch(...){s.failed=true;throw;}
}
void DeviceSequence::wait(int32_t timeout) {
  auto& s=*impl_;s.check();if(!s.in_flight||timeout<=0)throw std::logic_error("CUDA sequence has no waitable submission");
  std::exception_ptr failure;
  for(auto& p:s.programs)try{p->wait(timeout);}catch(...){if(!failure)failure=std::current_exception();}
  if(failure){s.failed=true;std::rethrow_exception(failure);}s.in_flight=false;
}
void DeviceSequence::run(int32_t timeout){if(timeout<=0)throw std::invalid_argument("positive timeout required");submit();wait(timeout);}
void DeviceSequence::close() {
  auto& s=*impl_;
  if(s.thread!=std::this_thread::get_id())throw std::logic_error("CUDA sequence requires its constructing thread");
  if(s.closed)return;
  try{for(auto& p:s.programs)p->close();s.programs.clear();s.closed=true;s.in_flight=false;}catch(...){s.failed=true;throw;}
}
size_t DeviceSequence::size() const{return impl_->programs.size();}
}
