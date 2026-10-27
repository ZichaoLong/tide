#include "task_services.h"
#include "cann_api.h"
#include "peer_api.h"
#include "portable_torch/runtime.hpp"
#include <c10/core/DeviceGuard.h>
#include <chrono>
#include <iostream>
namespace tide::device_online {
namespace {
struct Signal {
  portable_torch::RuntimeResource resource;PeerApi api;at::Device device;void* handle=nullptr;
  explicit Signal(at::Device d):device(d){c10::DeviceGuard guard(d);CannApi::check(api.create(&handle,0),"create local service notification");}
  ~Signal(){try{close();}catch(const std::exception& e){resource.quarantine();std::cerr<<"local service notification quarantined: "<<e.what()<<'\n';}}
  void close(){if(handle){c10::DeviceGuard guard(device);CannApi::check(api.destroy(handle),"destroy local service notification");handle=nullptr;}resource.close();}
};
void record(DeviceProgram& p,const std::shared_ptr<Signal>& signal){
  p.kernel([signal](void* stream){signal->resource.check();CannApi::check(signal->api.record(signal->handle,stream),"signal local service");},{});
}
void await(DeviceProgram& p,const std::shared_ptr<Signal>& signal){
  p.kernel([signal](void* stream){signal->resource.check();CannApi::check(signal->api.wait_reset(signal->handle,stream,60000),"await local service");},{});
}
struct Service {
  std::shared_ptr<Signal> request,done;at::Tensor command,work,stop;
  std::unique_ptr<DeviceProgram> program;size_t head,end;
  bool ended=false;
};
}
struct TaskServices::Impl {
  DeviceProgram& coordinator;at::Device device;int64_t budget;
  std::vector<std::unique_ptr<Service>> services;bool stopped=false,finished=false;
  Impl(DeviceProgram& p,at::Device d,int64_t b):coordinator(p),device(d),budget(b){}
};
TaskServices::TaskServices(DeviceProgram& p,at::Device d,int64_t budget):impl_(std::make_unique<Impl>(p,d,budget)){}
TaskServices::~TaskServices()=default;
DeviceProgram& TaskServices::begin(){
  auto& s=*impl_;if(s.stopped||s.finished||(!s.services.empty()&&!s.services.back()->ended))throw std::logic_error("invalid local service construction order");
  if(s.budget<1)throw std::invalid_argument("complete local service workspace budget exceeded");
  auto x=std::make_unique<Service>();x->request=std::make_shared<Signal>(s.device);x->done=std::make_shared<Signal>(s.device);
  auto opts=at::TensorOptions().device(s.device).dtype(at::kInt);x->command=at::zeros({1},opts);x->work=at::ones_like(x->command);x->stop=at::zeros_like(x->command);
  x->program=std::make_unique<DeviceProgram>(s.device);auto& p=*x->program;p.limit_workspace(s.budget);
  x->head=p.label();const auto body=p.label();x->end=p.label();p.mark(x->head);await(p,x->request);p.branch(x->command,{x->end,body});p.mark(body);
  s.coordinator.copy(x->command,x->work);record(s.coordinator,x->request);
  s.services.push_back(std::move(x));return p;
}
void TaskServices::end(DeviceProgram& p){
  auto& s=*impl_;if(s.services.empty())throw std::logic_error("missing local service");auto& x=*s.services.back();
  if(x.program.get()!=&p||x.ended)throw std::logic_error("local service boundary mismatch");
  // No numerical operations follow done before the next request. In
  // particular the unconditional jump does not read/update shared workspace.
  record(p,x.done);p.branch(x.stop,{x.head});p.mark(x.end);record(p,x.done);
  await(s.coordinator,x.done);x.ended=true;
}
void TaskServices::append_stop(){
  auto& s=*impl_;if(s.stopped||(!s.services.empty()&&!s.services.back()->ended))throw std::logic_error("invalid local service stop boundary");
  for(auto& x:s.services){s.coordinator.copy(x->command,x->stop);record(s.coordinator,x->request);await(s.coordinator,x->done);}s.stopped=true;
}
void TaskServices::finish(){
  auto& s=*impl_;if(!s.stopped||s.finished)throw std::logic_error("invalid local service finish");
  if(workspace_bytes()>s.budget)throw std::invalid_argument("complete local service workspace budget exceeded");
  // Coordinator -> request -> worker -> done -> coordinator is a strict
  // device happens-before chain. One high-water arena serves every stage.
  auto workspace=at::empty({workspace_bytes()},at::TensorOptions().device(s.device).dtype(at::kByte));
  s.coordinator.shared_workspace(workspace);
  for(auto& x:s.services)x->program->shared_workspace(workspace);
  for(auto& x:s.services)x->program->finish();s.finished=true;
}
void TaskServices::submit(){if(!impl_->finished)throw std::logic_error("unfinished local services");for(auto& x:impl_->services)x->program->submit();}
void TaskServices::wait(int32_t timeout){
  const auto start=std::chrono::steady_clock::now();
  for(auto& x:impl_->services){const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count();
    if(elapsed>=timeout)throw std::runtime_error("local service boundary timeout");x->program->wait(timeout-elapsed);}
}
void TaskServices::close(){
  for(auto& x:impl_->services)x->program->close();
  for(auto& x:impl_->services){x->request->close();x->done->close();}impl_->services.clear();
}
int64_t TaskServices::workspace_bytes()const{int64_t total=impl_->coordinator.workspace_bytes();for(const auto& x:impl_->services)total=std::max(total,x->program->workspace_bytes());return total;}
std::vector<const DeviceProgram*> TaskServices::programs()const{std::vector<const DeviceProgram*> p{&impl_->coordinator};for(const auto& x:impl_->services)p.push_back(x->program.get());return p;}
}
