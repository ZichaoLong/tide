#pragma once
#include "parameter_vjp.h"
namespace tide::device_online {
struct ParameterPlan {
  std::vector<ParameterOwner> owners;
  std::vector<int64_t> offsets,owner_table,references,tiles;
  int64_t elements;
  bool has_tanh,has_lh;
  int64_t swiglu_count;
  int64_t aggregate_slots=0,attention_elements=0;
};
ParameterPlan plan_parameters(const Graph&,const ParameterRegistry&,int64_t width,int64_t tensor_budget_bytes,bool controls=false);
} // namespace tide::device_online
