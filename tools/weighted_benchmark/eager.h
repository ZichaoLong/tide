#pragma once
#include "benchmark.h"
#include "tide/optimizer.h"
#include "tide/placement.h"
#include "root_values.h"
namespace weighted_benchmark {
// Native eager CPU/accelerator owner. Placement precedes optimizer creation;
// detached live state remains on its owners, returned observations are on CPU.
class Eager : public Resident {
  Fixture data;Model placed;Continuation state;ParameterRegistry registry;
  std::unique_ptr<SGD> optimizer;bool training,prefill;Options options;
  bool storage_reference,return_packed;std::map<std::string,Tensor> storage_masters;
  std::string return_mode;double loss_scale;
  std::vector<External> placed_inputs(std::vector<External>,std::map<std::string,Tensor>* leaves=nullptr)const;
  Result forward(const Continuation&,const std::vector<External>&,Index stop)const;
 public:
  Eager(const Json&,bool training,bool prefill,bool scalar=false,at::ScalarType=at::kFloat,
        at::Device=at::Device(at::kCPU),ExecutionPlacement={},Index workers=1,
        std::vector<at::Device> peers={},bool storage_reference=false,bool return_packed=false,
        std::vector<at::Device> node_devices={},std::string return_mode="complete");
  Step step(Index);
  RootDiagnostic diagnostic(bool split)const;
  std::map<std::string,Tensor> parameters()const;
  TrainingState training_state()const;
  Continuation checkpoint()const;
  Index cut()const override{return state.cut;}
  void save(const std::filesystem::path&,int64_t max_file_bytes)const;
  void restore(const std::filesystem::path&,int64_t max_file_bytes);
  void close()override{}
};
}
