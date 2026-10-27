#pragma once
#include "tide/types.h"
namespace tide::weighted {
// Architecture §2: original quotient, centered exact first-order mass derivative.
Tensor weighted_mean(const Tensor& mass,const Tensor& values);
}
