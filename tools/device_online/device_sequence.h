#pragma once
#include "device_program.h"

namespace tide::device_online {
// A bounded static sequence of programs on one device. Neighbor completion is
// ordered by reusable device notifications; no host wait or branch between
// programs. Each program retains its own device-controlled loops and labels.
// Split only at a semantic boundary whose dependencies are explicitly bridged.
class DeviceSequence {
 public:
  DeviceSequence(at::Device,int64_t max_programs,int64_t per_program_workspace);
  ~DeviceSequence();
  DeviceSequence(const DeviceSequence&)=delete;
  DeviceSequence& operator=(const DeviceSequence&)=delete;
  DeviceProgram& append();
  void finish();
  void submit();
  void wait(int32_t timeout_ms=10000);
  void run(int32_t timeout_ms=10000);
  void close();
  size_t size() const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace tide::device_online
