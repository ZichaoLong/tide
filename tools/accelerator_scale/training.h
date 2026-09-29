#pragma once
#include "execution.h"
#include <tide/optimizer.h>
#include "../../cpp/bench/metrics_jsonl_writer.h"

namespace accelerator_scale {
struct TrainingConfig {
  Index steps=0, warmup=1;
  std::string optimizer="adamw";
  double learning_rate=1e-4, loss_scale=1.;
  int backward_threads=1, optimizer_threads=1;
  void validate(const pdg_scale::Config&) const;
};
struct TrainingWindow { Tensor loss; Result result; };
std::unique_ptr<NamedOptimizer> make_optimizer(ParameterRegistry&,const TrainingConfig&);
TrainingWindow training_window(const pdg_scale::Config&,const pdg_scale::Topology&,
                               pdg_scale::Fixture&,const Placement&,Options);
void train(const pdg_scale::Config&,const pdg_scale::Topology&,pdg_scale::Fixture&,
           const Placement&,Options,const TrainingConfig&,portable_experiment::MetricsJsonlWriter&,double construction);
void check_training(const pdg_scale::Config&,const pdg_scale::Topology&,at::Device,Index,
                    const Placement&,const TrainingConfig&);
}  // namespace accelerator_scale
