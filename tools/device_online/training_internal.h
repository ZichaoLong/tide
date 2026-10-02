#pragma once
#include "tide/resident_training.h"
#include "content_flow.h"
#include "device_optimizer.h"
#include "retained_tape.h"
#include "training_parameters.h"
#include "sharded_training_owner.h"

namespace tide {
struct ResidentTrainingSession::Impl {
  std::unique_ptr<training_detail::ShardedTrainingOwner> sharded;
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
  device_online::ParameterVjp layout, gradient, accumulated;
  std::unique_ptr<device_online::DeviceOptimizer> optimizer;
  std::unique_ptr<device_online::ContentFlow> flow;
  std::vector<Saved> saved;
  Tensor initial_present;
  uint64_t session;
  Index cut, generation=0, next_token=0, saved_bytes=0, bytes_per_window=0, projection_bytes=0;
  device_online::RetainedProjection projection_snapshot;
  bool gradients_ready=false, failed=false;
  Index accumulated_batches=0;
  Impl(Graph,Model,const Continuation&,at::Device,ResidentOptimizerKind,
       std::vector<OptimizerGroup>,ResidentTrainingLimits,const ResidentTrainingCheckpoint* = nullptr);
  void check() const;
  void discard();
  ResidentGradients reverse(const std::vector<ResidentCotangents>&);
};
} // namespace tide
