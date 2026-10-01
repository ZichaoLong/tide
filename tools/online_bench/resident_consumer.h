#pragma once
#include "consumer.h"
#include "head_budget.h"
#include <tide/resident_training.h>
#include <array>
namespace tide_flow {
struct ConsumerLoss {Tensor value,root,head_gradient;Index count=0,chunks=0;};
ConsumerLoss head_loss(const tide::ResidentWindow&,const Tensor&,const Packet&,Index denominator,bool backward,const HeadBudget&);
Tensor embedding_gradient(const tide::ResidentGradients&,const Tensor&);
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
  double construction=0;
  Index cut=0;
  std::vector<double> seconds,warmup,losses;
  std::vector<Index> outputs;
  std::vector<std::map<std::string,Index>> statistics;
  tide::ResidentTrainingLimits limits;
  tide::ResidentPlacement placement;
  HeadBudget head;
};
tide::ResidentTrainingLimits resident_limits(const Config&,at::Device);
std::string resident_record(const Packet&,const Config&,at::Device,const ResidentMeasurements&);
void resident_gradients_json(std::ostream&,Index,const Fixture&,const tide::ResidentGradients&,const Tensor&,const Tensor&);
void resident_updated_json(std::ostream&,Index,const Fixture&,const tide::ResidentTrainingCheckpoint*,const Tensor&,const Tensor&);
} // namespace tide_flow
