#pragma once
#include "benchmark.h"
#include "tide/optimizer.h"
#include "tide/placement.h"
namespace weighted_benchmark {
// Native eager CPU/accelerator owner. Placement precedes optimizer creation;
// detached live state remains on its owners, returned observations are on CPU.
class Eager {
  Fixture data;Model placed;Continuation state;ParameterRegistry registry;
  std::unique_ptr<SGD> optimizer;bool training,prefill;Options options;
  bool storage_reference;std::map<std::string,Tensor> storage_masters;
 public:
  Eager(const Json&,bool training,bool prefill,bool scalar=false,at::ScalarType=at::kFloat,
        at::Device=at::Device(at::kCPU),ExecutionPlacement={},Index workers=1,
        std::vector<at::Device> peers={},bool storage_reference=false);
  Step step(Index);
  std::map<std::string,Tensor> parameters()const;
  TrainingState training_state()const;
  void save(const std::filesystem::path&,int64_t max_file_bytes)const;
  void restore(const std::filesystem::path&,int64_t max_file_bytes);
};
}
