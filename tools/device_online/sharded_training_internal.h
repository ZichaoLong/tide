#pragma once
#include "sharded_training_owner.h"
#include "training_parameters.h"
#include "content_flow.h"
#include "sharded_state.h"
#include "sharded_state_vjp.h"
#include "sharded_parameter_reduce.h"
#include "sharded_optimizer.h"
#include "parameter_plan.h"
namespace tide::training_detail {
struct ShardedTrainingOwner::Impl {
  struct Saved {
    ResidentToken token;
    device_online::RetainedShardedTape tape;
    std::vector<ResidentStateWindow> states;
  };
  Graph graph;Model model;at::Device device;
  ResidentTrainingLimits limits;ResidentPlacement placement;
  ResidentOptimizerKind kind;
  std::vector<Version> versions;
  ParameterRegistry registry;
  device_online::ParameterPlan global_layout;
  std::vector<device_online::ParameterVjp> layout,gradient,accumulated;
  std::vector<std::unique_ptr<device_online::DeviceOptimizer>> optimizers;
  std::vector<OptimizerGroup> groups;
  std::unique_ptr<device_online::ContentFlow> flow;
  std::vector<Saved> saved;
  std::vector<Tensor> initial_present;
  uint64_t session;
  Index cut,generation=0,next_token=0,bytes_per_window=0,saved_bytes=0,projection_bytes=0;
  device_online::RetainedProjection projection_snapshot;
  bool gradients_ready=false,failed=false;
  Index accumulated_batches=0;
  Impl(Graph,Model,const Continuation&,at::Device,ResidentOptimizerKind,
      std::vector<OptimizerGroup>,ResidentTrainingLimits,const ResidentTrainingCheckpoint*);
  void check() const;
  void discard();
  ResidentGradients reverse(const std::vector<ResidentCotangents>&);
  ResidentOptimizerState optimizer_state() const;
  void restore_optimizer(const ResidentTrainingCheckpoint&);
};
} // namespace tide::training_detail
