#include "../task_services.h"
namespace tide::device_online {
// Numerical phases stay in the same conditional CUDA graph. This does not
// change any online readiness/selection or use a captured real trajectory.
struct TaskServices::Impl {DeviceProgram& program;explicit Impl(DeviceProgram& p):program(p){}};
TaskServices::TaskServices(DeviceProgram& p,at::Device,int64_t):impl_(std::make_unique<Impl>(p)){}
TaskServices::~TaskServices()=default;
DeviceProgram& TaskServices::begin(){return impl_->program;}
void TaskServices::end(DeviceProgram& p){if(&p!=&impl_->program)throw std::logic_error("CUDA phase owner mismatch");}
void TaskServices::append_stop(){}
void TaskServices::finish(){}
void TaskServices::submit(){}
void TaskServices::wait(int32_t){}
void TaskServices::close(){}
int64_t TaskServices::workspace_bytes()const{return impl_->program.workspace_bytes();}
std::vector<const DeviceProgram*> TaskServices::programs()const{return {&impl_->program};}
}
