#include "training.h"
#include "precision.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace accelerator_scale {
void TrainingConfig::validate(const pdg_scale::Config& c) const {
  if(backward_threads<1 || backward_threads>160 || optimizer_threads<1 || optimizer_threads>160
      || (!steps && (backward_threads!=1 || optimizer_threads!=1)))
    throw std::invalid_argument("phase threads require training and1..160 threads");
  if(steps<0 || steps>100 || warmup<0 || (steps && (warmup>=steps || !c.grad || c.warmup!=0))
      || (optimizer!="sgd" && optimizer!="adamw") || !std::isfinite(learning_rate) || learning_rate<=0 || !std::isfinite(loss_scale) || loss_scale<=0)
    throw std::invalid_argument("training requires grad1, token warmup0, positive steps/lr and fewer warmup updates");
}
TrainingWindow training_window(const pdg_scale::Config& c,const pdg_scale::Topology& topology,
                               pdg_scale::Fixture& f,const Placement& placement,Options options) {
  // One independent sequence per optimizer update. State starts empty, while
  // every token within the window retains its complete autograd/KV history.
  Execution cursor(f.graph,f.model,options,c.batch,placement);
  DenseLinear head(c.head_workers);
  TrainingWindow window;std::vector<Tensor> losses;
  for(Index token=0;token<c.steps;++token) {
    auto ids=at::remainder(at::arange(c.batch,at::TensorOptions().dtype(at::kLong))*3+token*7,c.vocab);
    auto embeddings=embed(f.embedding,ids,!placement.resident);
    std::vector<External> inputs;
    for(Index b=0;b<c.batch;++b)inputs.push_back({b,0,token,token*(topology.layers+1),embeddings[b]});
    auto result=cursor.advance(inputs,(token+1)*(topology.layers+1),(token+1)*(topology.layers+1));
    std::vector<Tensor> hidden(c.batch,at::zeros({c.width},embeddings.options()));
    for(const auto& output:result.outputs)hidden.at(output.batch)=output.value;
    auto logits=project(at::stack(hidden),f.head,!placement.resident,&head);
    auto targets=at::remainder(ids+1,c.vocab).to(logits.device());
    losses.push_back(at::cross_entropy_loss(logits.to(at::kFloat),targets));
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
  at::AutoGradMode grad(true);const auto setup_begin=Clock::now();
  TrainingOwners owners(fixture,config);synchronize(placement);
  const auto optimizer_setup=seconds(setup_begin);const auto started=Clock::now();
  for(Index step=0;step<config.steps;++step) {
    synchronize(placement);transfers.reset();reset_memory(placement);
    const auto begin=Clock::now();owners.zero_grad();synchronize(placement);
    ProfileScope forward_trace(placement.profile,placement.devices,step,"forward");
    const auto forward_begin=Clock::now();
    auto window=training_window(c,topology,fixture,placement,options);
    synchronize(placement);const auto forward=seconds(forward_begin);
    forward_trace.finish();
    ProfileScope backward_trace(placement.profile,placement.devices,step,"backward");
    const auto backward_begin=Clock::now();owners.backward(window.loss);
    synchronize(placement);const auto backward=seconds(backward_begin);
    backward_trace.finish();
    ProfileScope optimizer_trace(placement.profile,placement.devices,step,"optimizer");
    const auto optimizer_begin=Clock::now();owners.step();synchronize(placement);
    const auto update=seconds(optimizer_begin);const auto total=seconds(begin);
    optimizer_trace.finish();
    const auto loss=window.loss.item<double>();
    if(!std::isfinite(loss))throw std::runtime_error("nonfinite training loss");
    Index gradients=0;for(const auto& p:fixture.owners)gradients+=p.grad().defined();
    if(!gradients)throw std::runtime_error("training objective disconnected from all owners");
    auto metrics=transfers.metrics();auto mem=memory(placement);metrics.insert(mem.begin(),mem.end());
    metrics.insert({{"perf/train_step_seconds",total},{"perf/forward_seconds",forward},
      {"perf/backward_seconds",backward},{"perf/optimizer_seconds",update},
      {"perf/ms_per_sample_token",total*1000/(c.batch*c.steps)},
      {"perf/sample_tokens_per_second",c.batch*c.steps/total},{"perf/construction_seconds",construction},{"perf/optimizer_setup_seconds",optimizer_setup},
      {"train/loss",loss},{"train/loss_scale",config.loss_scale},{"memory/fp32_master_bytes",owners.master_bytes()},{"train/gradient_owners",double(gradients)},
      {"train/optimizer_state_owners",double(owners.optimizer().state().size())},
      {"train/window_tokens",double(c.steps)},{"runtime/devices",double(placement.devices.size())}});
    metrics.insert({{"runtime/workers",double(c.workers)}, {"runtime/head_workers",double(c.head_workers)},
                    {"runtime/aten_threads",double(c.threads)},
                    {"runtime/backward_aten_threads",double(owners.backward_threads().aten)},
                    {"runtime/backward_openblas_threads",double(owners.backward_threads().openblas)},
                    {"runtime/optimizer_aten_threads",double(owners.optimizer_threads().aten)},
                    {"runtime/optimizer_openblas_threads",double(owners.optimizer_threads().openblas)}});
    for(const auto& [key,value]:fixture.inventory)metrics["model/"+key]=value;
    for(const auto& [key,value]:window.result.stats)metrics["work/"+key]=value;
    writer.Write(step,metrics,seconds(started),{{"phase",std::string(step<config.warmup?"warmup":"measure")},
      {"mode",std::string("training")},{"optimizer",config.optimizer},{"window_boundary",std::string("reset")},
      {"dtype",portable_torch::dtype_name(c.runtime.dtype)},{"optimizer_dtype",std::string("float32")},{"memory",c.memory},{"ranking_device",placement.ranking_device},{"event_device",placement.event_device}});
    std::cout<<"TRAIN "<<step<<" seconds="<<total<<" forward="<<forward<<" backward="<<backward
             <<" optimizer="<<update<<" loss="<<loss<<" gradient_owners="<<gradients<<'\n'<<std::flush;
  }
}
}  // namespace accelerator_scale
