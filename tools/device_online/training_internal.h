#pragma once
#include "tide/resident_training.h"
#include "content_flow.h"
#include "device_optimizer.h"
#include "retained_tape.h"

namespace tide {
namespace training_detail {
struct Version {Tensor value;int64_t version;const void* data;};
Model freeze_model(Model,at::Device,std::vector<Version>&);
Continuation freeze_continuation(Continuation);
void restore_parameters(Model&,const ResidentTrainingCheckpoint&);
void no_grad();
}
struct ResidentTrainingSession::Impl {
  struct Saved {
    ResidentToken token;
    device_online::RetainedTape tape;
    Tensor final, present;
  };
  Graph graph;
  Model model;
  at::Device device;
  ResidentTrainingLimits limits;
  ResidentOptimizerKind kind;
  std::vector<training_detail::Version> versions;
  ParameterRegistry registry;
  device_online::ParameterVjp layout, gradient;
  std::unique_ptr<device_online::DeviceOptimizer> optimizer;
  std::unique_ptr<device_online::ContentFlow> flow;
  std::vector<Saved> saved;
  Tensor initial_present;
  uint64_t session;
  Index cut, generation=0, next_token=0, saved_bytes=0, bytes_per_window=0;
  bool gradients_ready=false, failed=false;
  Impl(Graph,Model,const Continuation&,at::Device,ResidentOptimizerKind,
       std::vector<OptimizerGroup>,ResidentTrainingLimits,const ResidentTrainingCheckpoint* = nullptr);
  void check() const;
  void discard();
  ResidentGradients reverse(const std::vector<ResidentCotangents>&);
};
} // namespace tide
