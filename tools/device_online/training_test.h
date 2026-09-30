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
} // namespace tide::device_online::test
