#pragma once
#include "tide/resident_training.h"
namespace tide::training_detail {
struct Version {Tensor value;int64_t version;const void* data;};
Model freeze_model(Model,at::Device,std::vector<Version>&);
Continuation freeze_continuation(Continuation);
void restore_parameters(Model&,const ResidentTrainingCheckpoint&);
void no_grad();
uint64_t session_id();
} // namespace tide::training_detail
