#pragma once
#include "benchmark.h"
namespace weighted_benchmark {
// The early request check ignores only the later static owner selection. Exact
// effective identity is checked again before restore or checkpoint publication.
Json checkpoint_manifest(const std::filesystem::path& directory,const Json& identity,bool request_only=false);
void save_run_checkpoint(const std::filesystem::path&,Resident&,const Json& identity,Index next_window,int64_t max_bytes);
Index restore_run_checkpoint(const std::filesystem::path&,Resident&,const Json& identity,int64_t max_bytes);
}  // namespace weighted_benchmark
