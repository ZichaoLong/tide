#pragma once
#include "cli.h"
#include "tide/placement.h"
#include "../../cpp/bench/comparison.h"
namespace weighted_benchmark {
void validate_root_budget(const Json&);
Json check_roots(const Json&,const Request&,at::Device,const ExecutionPlacement&,
                 const std::vector<at::Device>& peers,const std::vector<at::Device>& nodes,
                 tide_bench::ComparisonReport&);
}  // namespace weighted_benchmark
