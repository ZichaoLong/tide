#pragma once
#include "sharded_parameter_sources.h"
#include "parameter_vjp.h"
namespace tide::device_online {
// A construction-time description, with one packed gather/reduce operator per
// group. Tensor retention makes each device address valid through completion.
void append_owner_gradient_pack(CannProgram&,const std::vector<ParameterContribution>&,
    const at::Tensor& values,const at::Tensor& connected,const at::Tensor& error,int64_t budget);
void append_owner_gradient_reduce(CannProgram&,const std::vector<int64_t>& owners,
    const std::vector<int64_t>& references,const std::vector<int64_t>& tiles,
    const at::Tensor& partials,const at::Tensor& partial_connected,const ParameterVjp& output,
    const at::Tensor& error,int64_t budget);
} // namespace tide::device_online
