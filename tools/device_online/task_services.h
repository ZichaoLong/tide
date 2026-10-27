#pragma once
#include "device_program.h"
namespace tide::device_online {
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
  std::vector<const DeviceProgram*> programs()const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
}
