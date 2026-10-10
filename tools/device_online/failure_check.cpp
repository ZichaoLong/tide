#include "device_program.h"
#include "cann_api.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace tide::device_online {
struct DeviceProgramTestAccess {
  static std::unique_ptr<DeviceProgram> make(at::Device device,const std::function<void(CannApi&)>& inject) {
    return std::unique_ptr<DeviceProgram>(new DeviceProgram(device,inject));
  }
};
}
namespace {
using namespace tide::device_online;
void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
template<class F> void rejects(F f,const char* reason) {
  bool refused=false;try{f();}catch(const std::exception&){refused=true;}require(refused,reason);
}
enum class Fault { construction, finish, submit, wait, close, quarantine };
struct Injection {
  Fault fault;
  int creates=0,executes=0,syncs=0,unbinds=0,destroyed=0,early_release=0;
  bool submitted=false,drained=false;
  decltype(CannApi::create_stream) create;
  decltype(CannApi::end_model) end;
  decltype(CannApi::execute) execute;
  decltype(CannApi::sync_stream) sync;
  decltype(CannApi::unbind_stream) unbind;
  decltype(CannApi::destroy_stream) destroy_stream;
  decltype(CannApi::destroy_model) destroy_model;
  decltype(CannApi::destroy_tensor) destroy_tensor;
  static Injection* current;
  explicit Injection(Fault f):fault(f){current=this;}
  void releasing(){if(submitted&&!drained)++early_release;}
  void install(CannApi& api) {
    create=api.create_stream;end=api.end_model;execute=api.execute;sync=api.sync_stream;
    unbind=api.unbind_stream;destroy_stream=api.destroy_stream;destroy_model=api.destroy_model;destroy_tensor=api.destroy_tensor;
    api.create_stream=[](void** out,uint32_t priority,uint32_t flags){auto& s=*current;
      if(++s.creates==2&&s.fault==Fault::construction)return 910001;
      return s.create(out,priority,flags);};
    api.end_model=[](void* model,void* stream){auto& s=*current;int code=s.end(model,stream);
      return code?code:s.fault==Fault::finish?910002:0;};
    api.execute=[](void* model,void* stream){auto& s=*current;++s.executes;
      int code=s.execute(model,stream);s.submitted=code==0;s.drained=false;
      return code?code:s.fault==Fault::submit?910003:0;};
    api.sync_stream=[](void* stream,int32_t timeout){auto& s=*current;++s.syncs;
      if(s.fault==Fault::quarantine||(s.fault==Fault::wait&&s.syncs==1))return 910004;
      int code=s.sync(stream,timeout);if(!code)s.drained=true;return code;};
    api.unbind_stream=[](void* model,void* stream){auto& s=*current;s.releasing();
      if(++s.unbinds==1&&s.fault==Fault::close)return 910005;
      return s.unbind(model,stream);};
    api.destroy_stream=[](void* stream){auto& s=*current;s.releasing();++s.destroyed;return s.destroy_stream(stream);};
    api.destroy_model=[](void* model){auto& s=*current;s.releasing();++s.destroyed;return s.destroy_model(model);};
    api.destroy_tensor=[](const void* tensor){auto& s=*current;s.releasing();++s.destroyed;return s.destroy_tensor(tensor);};
  }
};
Injection* Injection::current=nullptr;

void run_case(at::Device device,Fault fault,portable_torch::RuntimeSession& runtime,
              const portable_torch::RuntimeOptions& options) {
  at::NoGradGuard guard;Injection injection(fault);
  auto make=[&]{return DeviceProgramTestAccess::make(device,[&](CannApi& api){injection.install(api);});};
  if(fault==Fault::construction) {
    rejects([&]{auto program=make();},"injected construction failure was swallowed");
    require(injection.destroyed==1,"partial construction leaked its first stream");
    require(injection.early_release==0,"construction cleanup ran before completion");return;
  }
  auto program=make();
  rejects([&]{runtime.close();},"last runtime session closed with a live program");
  auto value=at::zeros({8},at::TensorOptions().device(device).dtype(at::kFloat));
  auto one=at::ones_like(value);
  c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl> owner(one.getIntrusivePtr());
  program->add(value,one);one.reset();
  require(!owner.expired(),"program failed to retain its input owner");
  if(fault==Fault::finish) {
    rejects([&]{program->finish();},"injected finish failure was swallowed");
    rejects([&]{program->run();},"failed construction became executable");
    program->close();require(owner.expired(),"closed construction retained its tensor");return;
  }
  program->finish();portable_torch::synchronize(device);
  if(fault==Fault::close) {
    program->run();
    rejects([&]{program->close();},"injected unbind failure was swallowed");
    require(!owner.expired()&&injection.destroyed==0,"failed unbind released owners");
    rejects([&]{program->run();},"program resumed after failed close");
    program->close();
  }else {
    rejects([&]{program->run();},"injected asynchronous failure was swallowed");
    rejects([&]{program->run();},"failed execution was resubmitted");
    require(injection.executes==1&&!owner.expired(),"failed submission lost ownership");
    if(fault==Fault::quarantine) {
      rejects([&]{program->close();},"unconfirmed completion was released");
      require(injection.destroyed==0&&injection.unbinds==0&&!owner.expired(),"refused drain released resources");
      program.reset(); // Repeated drain failure quarantines the complete owner.
      require(injection.destroyed==0&&injection.unbinds==0&&!owner.expired(),"quarantine lost resources");
      rejects([&]{runtime.close();},"quarantined runtime was finalized");
      rejects([&]{portable_torch::resolve_device(options);},"quarantined runtime accepted new work");
      rejects([&]{portable_torch::RuntimeSession nested;},"quarantined runtime accepted new session");
      require(injection.early_release==0,"quarantine released before drain");
      std::cout<<"device-quarantine: checked retained_owners=true finalize_refused=true worker_exit=86\n"<<std::flush;
      // No actual stuck kernel is created: the injected API merely withholds
      // completion confirmation. Exercise the required failed-worker boundary
      // without calling vendor finalization or destructing retained owners.
      std::_Exit(86);
    }
    program->close();
  }
  require(injection.drained&&injection.early_release==0,"cleanup did not wait for completion");
  require(owner.expired(),"successful cleanup leaked a retained tensor");
  require(value.cpu().sum().item<float>()==8,"accepted asynchronous submission did not execute once");
  program->close();program.reset();
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    bool quarantine=false;std::vector<char*> cli{argv[0]};
    for(int i=1;i<argc;++i)if(std::string(argv[i])=="--quarantine")quarantine=true;else cli.push_back(argv[i]);
    auto args=portable_torch::parse_cli(cli.size(),cli.data(),true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("failure check requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("failure check requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);
    if(quarantine)run_case(device,Fault::quarantine,runtime,args);
    for(auto fault:{Fault::construction,Fault::finish,Fault::submit,Fault::wait,Fault::close})run_case(device,fault,runtime,args);
    runtime.close();
    std::cout<<"device-failure: passed recoverable_cases=5 retained_owners=true early_release=false\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
