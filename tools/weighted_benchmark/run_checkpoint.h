#pragma once
#include "benchmark.h"
namespace weighted_benchmark {
Json checkpoint_manifest(const std::filesystem::path& directory,const Json& identity);
void save_run_checkpoint(const std::filesystem::path&,Resident&,const Json& identity,Index next_window,int64_t max_bytes);
Index restore_run_checkpoint(const std::filesystem::path&,Resident&,const Json& identity,int64_t max_bytes);
}  // namespace weighted_benchmark
