#pragma once
#include "flow_config.h"
namespace accelerator_scale::flows {
void check_roots(const Result&,const Result&,const bounded::Program*,const bounded::Window&,
                 const std::vector<Tensor>&,const std::vector<Tensor>&,
                 const std::vector<Tensor>&,double,double);
}
