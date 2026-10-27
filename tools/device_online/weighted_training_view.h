#pragma once
#include "tide/resident_weighted_training.h"
#include "weighted_window_vjp.h"
namespace tide::device_online {
WeightedTrainingWindow weighted_training_view(const WeightedWindowSnapshot&,ResidentToken);
// Boundary admission only. No reference schedule is consulted or constructed.
void validate_weighted_cotangents(const WeightedTrainingWindow&,const WeightedCotangents&);
void seed_weighted_cotangents(const WeightedWindowSnapshot&,const WeightedTrainingWindow&,
  const WeightedCotangents&,WeightedWindowRoots&);
}
