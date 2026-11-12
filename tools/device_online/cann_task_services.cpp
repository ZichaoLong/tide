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
  std::unique_ptr<DeviceProgram> program;DeviceProgram* parent=nullptr;size_t head,end;
  std::vector<size_t> children;
  bool ended=false;
};
void stop_child(DeviceProgram& parent,const Service& child){
  parent.copy(child.command,child.stop);record(parent,child.request);await(parent,child.done);
}
}
struct TaskServices::Impl {
  DeviceProgram& coordinator;at::Device device;int64_t budget;
  std::vector<std::unique_ptr<Service>> services;std::vector<size_t> open;bool stopped=false,finished=false;
  int64_t phases=0,maximum_nesting=0;
  Impl(DeviceProgram& p,at::Device d,int64_t b):coordinator(p),device(d),budget(b){}
};
TaskServices::TaskServices(DeviceProgram& p,at::Device d,int64_t budget):impl_(std::make_unique<Impl>(p,d,budget)){}
TaskServices::~TaskServices()=default;
DeviceProgram& TaskServices::begin(){
  auto& s=*impl_;if(s.stopped||s.finished)throw std::logic_error("invalid local service construction order");
  if(s.budget<1)throw std::invalid_argument("complete local service workspace budget exceeded");
  auto x=std::make_unique<Service>();x->request=std::make_shared<Signal>(s.device);x->done=std::make_shared<Signal>(s.device);
  x->parent=s.open.empty()?&s.coordinator:s.services[s.open.back()]->program.get();
  auto opts=at::TensorOptions().device(s.device).dtype(at::kInt);x->command=at::zeros({1},opts);x->work=at::ones_like(x->command);x->stop=at::zeros_like(x->command);
  x->program=std::make_unique<DeviceProgram>(s.device);auto& p=*x->program;p.limit_workspace(s.budget);
  x->head=p.label();const auto body=p.label();x->end=p.label();p.mark(x->head);await(p,x->request);p.branch(x->command,{x->end,body});p.mark(body);
  x->parent->copy(x->command,x->work);record(*x->parent,x->request);
  if(!s.open.empty())s.services[s.open.back()]->children.push_back(s.services.size());
  s.open.push_back(s.services.size());s.services.push_back(std::move(x));++s.phases;
  s.maximum_nesting=std::max(s.maximum_nesting,static_cast<int64_t>(s.open.size()));return p;
}
void TaskServices::end(DeviceProgram& p){
  auto& s=*impl_;if(s.open.empty())throw std::logic_error("missing local service");auto& x=*s.services[s.open.back()];
  if(x.program.get()!=&p||x.ended)throw std::logic_error("local service boundary mismatch");
  // No numerical operations follow done before the next request. In
  // particular the unconditional jump does not read/update shared workspace.
  record(p,x.done);p.branch(x.stop,{x.head});p.mark(x.end);
  // A stop request arrives only at the idle head, after every numerical child
  // call has returned. Recursively stop those same children on their owning
  // parent stream; root completion then certifies the whole subtree is idle.
  for(auto index:x.children)stop_child(p,*s.services[index]);
  record(p,x.done);
  await(*x.parent,x.done);x.ended=true;s.open.pop_back();
}
void TaskServices::append_stop(){
  auto& s=*impl_;if(s.stopped||!s.open.empty())throw std::logic_error("invalid local service stop boundary");
  // Root owns only direct children. Emitting every descendant's stop here
  // needlessly grows one captured command stream with the whole service tree.
  for(auto& x:s.services)if(x->parent==&s.coordinator)stop_child(s.coordinator,*x);
  s.stopped=true;
}
void TaskServices::finish(){
  auto& s=*impl_;if(!s.stopped||s.finished)throw std::logic_error("invalid local service finish");
  if(workspace_bytes()>s.budget)throw std::invalid_argument("complete local service workspace budget exceeded");
  // Coordinator -> request -> worker -> done -> coordinator is a strict
  // device happens-before chain, including nested parents suspended on a child.
  // One high-water arena serves every stage; no sibling work overlaps.
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
TaskServiceStatistics TaskServices::statistics()const{
  const auto& s=*impl_;TaskServiceStatistics out{s.phases,0,s.maximum_nesting,workspace_bytes()};
  for(const auto* p:programs()){
    const auto used=p->resources();out.programs+=used.programs;out.streams+=used.streams;out.boundary_events+=used.boundary_events;
    if(p!=&s.coordinator)out.worker_programs+=used.programs;
  }
  for(const auto& x:s.services)out.notifications+=int64_t(x->request->handle!=nullptr)+int64_t(x->done->handle!=nullptr);
  return out;
}
std::vector<const DeviceProgram*> TaskServices::programs()const{std::vector<const DeviceProgram*> p{&impl_->coordinator};for(const auto& x:impl_->services)p.push_back(x->program.get());return p;}
}
