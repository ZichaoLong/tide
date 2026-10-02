#pragma once
#include "tide/resident_training.h"
namespace tide::training_detail {
// Implementation of the public session's explicit compact-owner configuration.
// The public token, root, optimizer and portable checkpoint contracts are shared.
class ShardedTrainingOwner {
 public:
  ShardedTrainingOwner(Graph,Model,const Continuation&,at::Device,ResidentOptimizerKind,
      std::vector<OptimizerGroup>,ResidentTrainingLimits,const ResidentTrainingCheckpoint*);
  ~ShardedTrainingOwner();
  void check() const;
  ResidentTrainingWindow advance(const std::vector<External>&,Index,Index);
  ResidentGradients backward(const std::vector<ResidentCotangents>&);
  void accumulate(Index max_bytes);
  ResidentStep step();
  void detach();
  ResidentTrainingCheckpoint checkpoint() const;
  ResidentContinuation snapshot_device(Index max_bytes) const;
  void restore_device(const ResidentContinuation&);
  Result result() const;
  Index cut() const;
  Index generation() const;
  Index retained_windows() const;
  Index accumulated_batches() const;
  ResidentPlacement placement() const;
  void close();
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace tide::training_detail
