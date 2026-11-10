#pragma once
// Allocation §3: finite Describe columns, independent of scalar scoring.
#include "weighted_scoring.h"

namespace tide::weighted::query {
TensorFields describe_columns(const NodeWeights&,const RegionalConfig&,
    const Tensor& mass,const Tensor& features,const detail::Groups&,double epsilon);
}
