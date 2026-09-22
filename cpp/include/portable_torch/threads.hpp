#pragma once
#include <map>
#include <string>

namespace portable_torch {
// Optional linked-library introspection, never a BLAS control or dependency.
// Missing symbols are reported as -1, not inferred from environment variables.
std::map<std::string, double> thread_metrics();
std::string blas_description();
} // namespace portable_torch
