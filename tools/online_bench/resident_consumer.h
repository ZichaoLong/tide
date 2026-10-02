#pragma once
#include "consumer.h"
#include "head_budget.h"
#include "capacity.h"
#include "memory.h"
#include "phase_timing.h"
#include <tide/resident_training.h>
#include <array>
namespace tide_flow {
struct ConsumerLoss {Tensor value,root,head_gradient;Index count=0,chunks=0;};
ConsumerLoss head_loss(const tide::ResidentWindow&,const Tensor&,const Packet&,Index denominator,bool backward,const HeadBudget&,Index sample_begin=0);
Tensor embedding_gradient(const tide::ResidentGradients&,const Tensor&,Index sample_begin=0);
class ConsumerOptimizer {
 public:
  ConsumerOptimizer(Tensor embedding,Tensor head,std::string kind);
  void prepare(const Tensor& embedding_gradient,const Tensor& head_gradient);
  void commit();
 private:
  struct State {Tensor master,first,second;Index step=0;};
  std::array<Tensor,2> payloads_;
  std::array<State,2> states_,proposal_;
  std::array<Tensor,2> rounded_;
  bool ready_=false;
  std::string kind_;
};
struct ResidentMeasurements {
  std::string memory;
  double construction=0;
  Index cut=0;
  Index sample_rows=0,sample_chunks=1,accumulation_budget=0;
  std::vector<double> seconds,warmup,losses;
  PhaseTiming phases;
  std::vector<Index> outputs;
  std::vector<std::map<std::string,Index>> statistics;
  tide::ResidentTrainingLimits limits;
  tide::ResidentPlacement placement;
  HeadBudget head;
  capacity::Plan capacity;
  std::vector<DeviceMemoryInfo> initial_memory;
  std::vector<Index> peak_growth;
  std::map<Index,Index> context_peaks;
};
void prepare_capacity(const Packet&,const Config&,const std::vector<at::Device>&,ResidentMeasurements&);
std::string capacity_json(const Config&,const ResidentMeasurements&);
tide::ResidentTrainingLimits resident_limits(const Config&,at::Device);
std::string resident_record(const Packet&,const Config&,at::Device,const ResidentMeasurements&,const std::string& error="");
void resident_gradient_add(std::map<std::string,Tensor>&,const Fixture&,const tide::ResidentGradients&);
void resident_gradients_json(std::ostream&,Index,std::map<std::string,Tensor>,const Tensor&,const Tensor&);
void resident_updated_json(std::ostream&,Index,const Fixture&,const tide::ResidentTrainingCheckpoint*,const Tensor&,const Tensor&);
} // namespace tide_flow
