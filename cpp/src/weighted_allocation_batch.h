#pragma once
// Allocation §6/S.2: complete B x N candidate rows, exact-map identity.
#include "weighted_regularizers.h"

namespace tide::weighted::batch {
enum Branch:Index {Zero=0,Caps=1,Slack=2,Lower=3,Upper=4,Equality=5};
inline const std::vector<std::string> branches{"zero","caps","slack","lower","upper","equality"};
struct NumericAllocation {Tensor controls,multiplier,branch,response,free,residual;};
NumericAllocation solve(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,
                        const detail::Regularizer&,const Tensor& shift={});
}
