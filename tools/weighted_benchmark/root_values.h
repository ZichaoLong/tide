#pragma once
// Diagnostic objectives/leaves only; no scheduling or optimizer policy.
#include "tide/types.h"
namespace weighted_benchmark {
struct RootDiagnostic {
  tide::Result result;
  tide::Continuation initial;
  std::map<std::string,at::Tensor> leaves;
  tide::Index start=0,stop=0;
};
std::map<std::string,at::Tensor> diagnostic_roots(const RootDiagnostic&);
std::map<std::string,at::Tensor> diagnostic_vjp(
    const at::Tensor&,const std::map<std::string,at::Tensor>&,bool zero);
}  // namespace weighted_benchmark
