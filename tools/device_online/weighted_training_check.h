#pragma once
#include "tide/resident_weighted_training.h"
namespace tide::device_online::weighted_training_check {
using Objective=std::pair<std::vector<at::Tensor>,std::vector<at::Tensor>>;
void require(bool,const std::string&);
void gradient(const at::Tensor&,bool,const at::Tensor&,const std::string&);
void equal_window(const WeightedTrainingWindow&,const WeightedTrainingWindow&);
WeightedCotangents seed(const WeightedTrainingWindow&,const std::string&);
Objective objective(const WeightedTrainingWindow&,const WeightedCotangents&,const Result&);
void check_gradients(const Graph&,const ParameterRegistry&,const Continuation&,
  const std::vector<std::vector<External>>&,const std::vector<at::Tensor>&,const WeightedGradients&);
void failures(at::Device);
void accumulation(at::Device);
void trajectory(at::Device,const std::string&,const std::string&,ResidentOptimizerKind,const std::string&);
WeightedTrainingLimits limits();
}
