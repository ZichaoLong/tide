#pragma once
#include "peer_exchange.h"

namespace tide::device_online {
// Global physical projection rows, sorted within each compact bank. Aliased
// parameters may occupy different physical rows; canonical reduction is later.
struct ProjectionBank {std::vector<int64_t> rows;at::Tensor weights,biases;};
struct ProjectionGradient {std::vector<int64_t> rows;at::Tensor weights,biases,connected;};
// A device-controlled service for one bounded batch of actual projection rows.
// The host constructs the protocol once. Payloads and global row IDs travel in
// packets; complete weight/gradient banks never return to the coordinator.
class ProjectionStage {
 public:
  ProjectionStage(std::vector<ProjectionBank>,int64_t parameters,int64_t chunk,
                  bool reverse,int64_t tensor_budget,int64_t operator_budget);
  // Reuse only after an ordered device reduction has consumed the preceding
  // window. append_reset gates peer zeroing with a coordinator start packet.
  ProjectionStage(std::vector<ProjectionBank>,int64_t parameters,int64_t chunk,
                  bool reverse,int64_t tensor_budget,int64_t operator_budget,
                  const std::vector<ProjectionGradient>& reuse);
  ~ProjectionStage();
  at::Tensor append(DeviceProgram&,const at::Tensor& parameter_rows,
      const at::Tensor& values,const at::Tensor& cotangents,const at::Tensor& error);
  void append_reset(DeviceProgram&,at::Device coordinator);
  void append_stop(DeviceProgram&);
  void synchronize_inputs() const;
  void submit();
  void wait();
  void close();
  std::vector<ProjectionGradient> gradients() const;
  int64_t packet_bytes() const;
  int64_t workspace_bytes() const;
  int64_t retained_tensor_bytes() const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace tide::device_online
