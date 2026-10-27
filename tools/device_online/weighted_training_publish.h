#pragma once
// Publish one canonical optimizer decision to this sample's physical banks.
#include "sharded_parameter_banks.h"
#include "device_program.h"
namespace tide::device_online {
class WeightedTrainingPublication {
 public:
  WeightedTrainingPublication(DeviceProgram&,const std::vector<ParameterWrite>&,
    const at::Tensor& masters,const at::Tensor& error,int64_t bytes,int64_t workspace,uint32_t timeout);
  static int64_t required_bytes(const std::vector<ParameterWrite>&,const at::Tensor& masters);
  ~WeightedTrainingPublication();
  void synchronize_inputs();
  void submit();void wait();void close();
  int64_t transferred_bytes()const;
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
}
