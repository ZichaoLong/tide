#pragma once
#include "benchmark.h"
namespace weighted_benchmark {
// Runtime/allocator observations are outside all measured step boundaries.
void runtime_probe(at::Device);
void capture_memory(Json& report,at::Device,const std::string& phase,bool reset=false);
}
