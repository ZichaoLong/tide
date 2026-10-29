#pragma once
#include "benchmark.h"
#include "tide/resident_training.h"
namespace weighted_benchmark {
// Consumer-owned disk schema. GraphCheckpoint handles exact raw state/aliases;
// this codec preserves actual resident optimizer buffers and int64 metadata.
void save_resident(const std::filesystem::path&,Fixture,const ResidentTrainingCheckpoint&,int64_t max_file_bytes);
ResidentTrainingCheckpoint load_resident(const std::filesystem::path&,Fixture&,int64_t max_file_bytes);
}
