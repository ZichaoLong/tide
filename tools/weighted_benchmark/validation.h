#pragma once
#include "benchmark.h"
#include "../../cpp/bench/comparison.h"
namespace weighted_benchmark {
Json comparison_json(const tide_bench::ComparisonReport&);
void compare_step(tide_bench::ComparisonReport&,const Step&,const Step&,at::ScalarType,Index window);
void compare_training(tide_bench::ComparisonReport&,const TrainingState&,const TrainingState&,at::ScalarType);
void compare_parameters(tide_bench::ComparisonReport&,const std::map<std::string,Tensor>&,
                        const std::map<std::string,Tensor>&,at::ScalarType,const std::string&);
}  // namespace weighted_benchmark
