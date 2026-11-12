#include "../task_services.h"
#include <algorithm>
namespace tide::device_online {
// Numerical phases stay in the same conditional CUDA graph. This does not
// change any online readiness/selection or use a captured real trajectory.
struct TaskServices::Impl {
  DeviceProgram& program;int64_t phases=0,depth=0,maximum_nesting=0;
  explicit Impl(DeviceProgram& p):program(p){}
};
TaskServices::TaskServices(DeviceProgram& p,at::Device,int64_t):impl_(std::make_unique<Impl>(p)){}
TaskServices::~TaskServices()=default;
DeviceProgram& TaskServices::begin(){auto& s=*impl_;++s.phases;++s.depth;s.maximum_nesting=std::max(s.maximum_nesting,s.depth);return s.program;}
void TaskServices::end(DeviceProgram& p){if(&p!=&impl_->program||impl_->depth<1)throw std::logic_error("CUDA phase owner mismatch");--impl_->depth;}
void TaskServices::append_stop(){}
void TaskServices::finish(){}
void TaskServices::submit(){}
void TaskServices::wait(int32_t){}
void TaskServices::close(){}
int64_t TaskServices::workspace_bytes()const{return impl_->program.workspace_bytes();}
TaskServiceStatistics TaskServices::statistics()const{
  const auto used=impl_->program.resources();
  return {impl_->phases,0,impl_->maximum_nesting,workspace_bytes(),used.programs,used.streams,0,used.boundary_events};
}
std::vector<const DeviceProgram*> TaskServices::programs()const{return {&impl_->program};}
}
