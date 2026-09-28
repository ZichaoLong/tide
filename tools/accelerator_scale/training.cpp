#include "training.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace accelerator_scale {
void TrainingConfig::validate(const pdg_scale::Config& c) const {
  if(steps<0 || steps>100 || warmup<0 || (steps && (warmup>=steps || !c.grad || c.warmup!=0))
      || (optimizer!="sgd" && optimizer!="adamw") || !std::isfinite(learning_rate) || learning_rate<=0)
    throw std::invalid_argument("training requires grad1, token warmup0, positive steps/lr and fewer warmup updates");
}
void register_owners(ParameterRegistry& registry,const pdg_scale::Fixture& fixture) {
  for(size_t i=0;i<fixture.owners.size();++i)registry.add("owner."+std::to_string(i),fixture.owners[i]);
  if(registry.owners().size()!=fixture.owners.size())throw std::runtime_error("duplicate training owner");
}
std::unique_ptr<NamedOptimizer> make_optimizer(ParameterRegistry& registry,const TrainingConfig& config) {
  OptimizerGroup group;group.lr=config.learning_rate;group.weight_decay=.01;
  group.eps=1e-5;group.momentum=.9;
  for(const auto& owner:registry.owners())group.parameters.push_back(owner.canonical);
  if(config.optimizer=="adamw")return std::make_unique<AdamW>(registry,std::vector<OptimizerGroup>{group});
  return std::make_unique<SGD>(registry,std::vector<OptimizerGroup>{group});
}
TrainingWindow training_window(const pdg_scale::Config& c,const pdg_scale::Topology& topology,
                               pdg_scale::Fixture& f,const Placement& placement,Options options) {
  // One independent sequence per optimizer update. State starts empty, while
  // every token within the window retains its complete autograd/KV history.
  Execution cursor(f.graph,f.model,options,c.batch,placement);
  TrainingWindow window;std::vector<Tensor> losses;
  for(Index token=0;token<c.steps;++token) {
    auto ids=at::remainder(at::arange(c.batch,at::TensorOptions().dtype(at::kLong))*3+token*7,c.vocab);
    auto embeddings=embed(f.embedding,ids,!placement.resident);
    std::vector<External> inputs;
    for(Index b=0;b<c.batch;++b)inputs.push_back({b,0,token,token*(topology.layers+1),embeddings[b]});
    auto result=cursor.advance(inputs,(token+1)*(topology.layers+1),(token+1)*(topology.layers+1));
    std::vector<Tensor> hidden(c.batch,at::zeros({c.width},embeddings.options()));
    for(const auto& output:result.outputs)hidden.at(output.batch)=output.value;
    auto logits=project(at::stack(hidden),f.head,!placement.resident);
    auto targets=at::remainder(ids+1,c.vocab).to(logits.device());
    losses.push_back(at::cross_entropy_loss(logits,targets));
    for(const auto& [name,value]:result.stats)window.result.stats[name]+=value;
    if(options.trace) {
      window.result.trace.insert(window.result.trace.end(),result.trace.begin(),result.trace.end());
      window.result.messages.insert(window.result.messages.end(),result.messages.begin(),result.messages.end());
      window.result.outputs.insert(window.result.outputs.end(),result.outputs.begin(),result.outputs.end());
    }
  }
  window.loss=at::stack(losses).mean();
  if(options.trace)window.result.continuation=cursor.snapshot();
  return window;
}
void train(const pdg_scale::Config& c,const pdg_scale::Topology& topology,pdg_scale::Fixture& fixture,
           const Placement& placement,Options options,const TrainingConfig& config,
           portable_experiment::MetricsJsonlWriter& writer,double construction) {
  using pdg_scale::Clock;using pdg_scale::seconds;
  at::AutoGradMode grad(true);ParameterRegistry registry;register_owners(registry,fixture);
  auto optimizer=make_optimizer(registry,config);const auto started=Clock::now();
  for(Index step=0;step<config.steps;++step) {
    synchronize(placement);transfers.reset();reset_memory(placement);
    const auto begin=Clock::now();optimizer->zero_grad(true);synchronize(placement);
    const auto forward_begin=Clock::now();
    auto window=training_window(c,topology,fixture,placement,options);
    synchronize(placement);const auto forward=seconds(forward_begin);
    const auto backward_begin=Clock::now();window.loss.backward();
    synchronize(placement);const auto backward=seconds(backward_begin);
    const auto optimizer_begin=Clock::now();optimizer->step();synchronize(placement);
    const auto update=seconds(optimizer_begin);const auto total=seconds(begin);
    const auto loss=window.loss.item<double>();
    if(!std::isfinite(loss))throw std::runtime_error("nonfinite training loss");
    Index gradients=0;for(const auto& p:fixture.owners)gradients+=p.grad().defined();
    if(!gradients)throw std::runtime_error("training objective disconnected from all owners");
    auto metrics=transfers.metrics();auto mem=memory(placement);metrics.insert(mem.begin(),mem.end());
    metrics.insert({{"perf/train_step_seconds",total},{"perf/forward_seconds",forward},
      {"perf/backward_seconds",backward},{"perf/optimizer_seconds",update},
      {"perf/ms_per_sample_token",total*1000/(c.batch*c.steps)},
      {"perf/sample_tokens_per_second",c.batch*c.steps/total},{"perf/construction_seconds",construction},
      {"train/loss",loss},{"train/gradient_owners",double(gradients)},
      {"train/optimizer_state_owners",double(optimizer->state().size())},
      {"train/window_tokens",double(c.steps)},{"runtime/devices",double(placement.devices.size())}});
    for(const auto& [key,value]:fixture.inventory)metrics["model/"+key]=value;
    for(const auto& [key,value]:window.result.stats)metrics["work/"+key]=value;
    writer.Write(step,metrics,seconds(started),{{"phase",std::string(step<config.warmup?"warmup":"measure")},
      {"mode",std::string("training")},{"optimizer",config.optimizer},{"window_boundary",std::string("reset")},
      {"memory",c.memory},{"ranking_device",placement.ranking_device},{"event_device",placement.event_device}});
    std::cout<<"TRAIN "<<step<<" seconds="<<total<<" forward="<<forward<<" backward="<<backward
             <<" optimizer="<<update<<" loss="<<loss<<" gradient_owners="<<gradients<<'\n'<<std::flush;
  }
}
}  // namespace accelerator_scale
