#pragma once
#include "tide/resident_training.h"
#include "retained_fixture.h"

namespace tide::device_online::test {
void train_require(bool,const char*);
void train_reject(const std::function<void()>&,const char*);
Model train_model(Model,at::ScalarType);
Continuation train_boundary(Continuation,at::ScalarType);
ResidentCotangents train_roots(const ResidentTrainingWindow&,int window,int mode);
void train_gradients(const ResidentGradients&,const RetainedReference&,const Fixture&);
void train_checkpoint(const ResidentTrainingCheckpoint&,const Model&,const NamedOptimizer&);
void train_failures(at::Device);
void train_trajectory(at::Device,Fixture,bool prefill,ResidentOptimizerKind,at::ScalarType);
void train_trajectory(at::Device,Fixture,bool prefill,ResidentOptimizerKind,at::ScalarType,double adam_epsilon,bool conditioned_controls=false,Options={});
void train_numerics(at::Device,Fixture,bool prefill);
void train_forward_compare(const Result&,const Result&,const Graph&,bool conditioned_controls);
void train_control_checks();
} // namespace tide::device_online::test
