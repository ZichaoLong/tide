#pragma once
#include "device_program.h"
namespace tide::device_online {
struct TaskServiceStatistics {
  int64_t phases=0,worker_programs=0,maximum_nesting=0,shared_workspace_bytes=0;
  // Includes the coordinator. Before finish, workspace is the planned bound;
  // after finish it is the retained arena. Counts concern explicit owners only.
  int64_t programs=0,streams=0,notifications=0,boundary_events=0;
};
// Static numerical services under an online coordinator. NPU services execute
// device-commanded loops using local reusable notifications. Nested begin/end
// pairs form serial parent/child calls and close in reverse construction order.
// CUDA keeps the
// same phases in its conditional graph (no fixed CANN stream task buffer).
class TaskServices {
 public:
  TaskServices(DeviceProgram& coordinator,at::Device,int64_t workspace_bytes);
  ~TaskServices();
  DeviceProgram& begin();
  void end(DeviceProgram& worker);
  void append_stop();
  void finish();
  void submit();
  void wait(int32_t timeout_ms);
  void close();
  int64_t workspace_bytes()const;
  TaskServiceStatistics statistics()const;
  std::vector<const DeviceProgram*> programs()const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
}
