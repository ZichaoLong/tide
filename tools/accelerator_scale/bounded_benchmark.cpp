#include "bounded.h"
#include "bounded_optimizer.h"
#include "graph_replay.h"
#include "../../cpp/bench/metrics_jsonl_writer.h"
#include <ATen/Parallel.h>
#include <filesystem>
#include <iostream>
#include <set>

namespace {
bool used_npu=false;
int run(int argc,char** argv) {
  using namespace accelerator_scale;using namespace bounded;
  Index devices=1,iterations=4,iteration_warmup=1,workspace_gib=16;
  bool capture=true,training=false;std::string kind="adamw";
  std::vector<char*> common{argv[0]};std::set<std::string> seen;
  for(int i=1;i<argc;++i) {
    const std::string key=argv[i];
    if(key!="--devices" && key!="--iterations" && key!="--iteration-warmup" && key!="--workspace-gib"
        && key!="--replay" && key!="--training" && key!="--optimizer"){common.push_back(argv[i]);continue;}
    if(!seen.insert(key).second || ++i==argc)throw std::invalid_argument("duplicate/missing "+key);
    const std::string value=argv[i];if(key=="--optimizer"){kind=value;continue;}
    if(value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("integer required: "+key);
    const auto n=std::stoll(value);
    if(key=="--devices")devices=n;else if(key=="--iterations")iterations=n;
    else if(key=="--iteration-warmup")iteration_warmup=n;else if(key=="--workspace-gib")workspace_gib=n;
    else {if(n>1)throw std::invalid_argument("boolean requires0|1");if(key=="--replay")capture=n;else training=n;}
  }
  auto c=pdg_scale::parse(common.size(),common.data());
  if(c.runtime.help) {
    portable_torch::print_usage(std::cout,argv[0]);
    std::cout<<"Bounded historical window: --steps TOKENS --replay 0|1 --training 0|1 --devices 1..16\n"
      "--iterations N --iteration-warmup N --workspace-gib N --optimizer sgd|adamw.\n"
      "Each observation is a complete reset-state window; training owners persist. FP16 static loss scale128.\n";
    return 0;
  }
  if(devices<1 || devices>16 || iterations<2 || iterations>100 || iteration_warmup>=iterations
      || workspace_gib<1 || workspace_gib>32768 || (kind!="sgd" && kind!="adamw")
      || c.threads!=1 || c.check || c.runtime.device_spec=="auto")throw std::invalid_argument("invalid bounded benchmark configuration");
  if(c.runtime.dtype!=at::kFloat && c.runtime.dtype!=at::kHalf)throw std::invalid_argument("bounded payload requires FP32/FP16");
  c.runtime.allow_npu_float16=c.runtime.dtype==at::kHalf;c.grad=training;
  auto device=portable_torch::resolve_device(c.runtime);used_npu=device.type()==c10::DeviceType::PrivateUse1;
  if((!device.is_cpu() && !used_npu) || (capture && !used_npu))throw std::invalid_argument("bounded replay requires NPU; eager requires CPU/NPU");
  at::set_num_threads(1);at::set_num_interop_threads(1);at::AutoGradMode grad(training);
  auto topology=pdg_scale::read_topology(c.topology);
  const Limits limits{c.steps,c.batch,workspace_gib*(int64_t(1)<<30),false,training};
  const auto bound=estimate_workspace(topology,c.memory,c.width,c.runtime.dtype,limits,training);
  if(bound>limits.max_workspace_bytes)
    throw std::invalid_argument("bounded workspace estimate exceeds explicit capacity before weight allocation: "+std::to_string(bound));
  auto contexts=initialize_devices(device,devices);
  if(!std::filesystem::create_directories(c.runtime.output_dir))throw std::invalid_argument("new output directory required");
  portable_experiment::MetricsJsonlWriter writer(std::filesystem::path(c.runtime.output_dir)/"metrics.jsonl",c.run_id);
  auto construction_start=pdg_scale::Clock::now();std::unique_ptr<GraphReplay> replay;
  if(capture){std::vector<at::Device> targets;for(Index i=0;i<devices;++i)targets.emplace_back(device.type(),device.index()+i);
    replay=std::make_unique<GraphReplay>(targets);}
  portable_torch::seed_runtime(at::Device(at::kCPU),c.runtime.seed);
  auto fixture=pdg_scale::fixture(c,topology);Scoring scoring{"model","model",at::kFloat};configure_scoring(fixture,scoring);
  auto placement=place(fixture,device,devices,"locality",true);placement.scoring=scoring;
  Program program(fixture,topology,placement,limits);
  auto input=(at::arange(c.steps).unsqueeze(1)*7+at::arange(c.batch).unsqueeze(0)*3).remainder(c.vocab).to(at::kLong);
  auto ids=input.to(fixture.embedding.device());std::unique_ptr<bounded::Optimizer> optimizer;
  if(training)optimizer=std::make_unique<bounded::Optimizer>(fixture.owners,kind,1e-4,c.runtime.dtype==at::kHalf?128.:1.,iterations);
  synchronize(placement);const auto construction=pdg_scale::seconds(construction_start);
  Window window;Value loss;Tensor finite;
  auto execute=[&] {
    window=program.run(ids);
    if(training){loss=program.loss(window,ids);auto gradients=program.vjp(loss,at::ones_like(loss.data)*(c.runtime.dtype==at::kHalf?128.:1.),false);
      finite=optimizer->step(loss,gradients);}
  };
  const auto prepare_start=pdg_scale::Clock::now();execute();synchronize(placement);
  if(optimizer)optimizer->reset();
  window=Window{};loss=Value{};finite=Tensor{};
  if(capture)replay->capture(execute);
  if(optimizer)optimizer->reset();synchronize(placement);
  const auto prepare=pdg_scale::seconds(prepare_start);const auto start=pdg_scale::Clock::now();
  for(Index iteration=0;iteration<iterations;++iteration) {
    synchronize(placement);reset_memory(placement);const auto begin=pdg_scale::Clock::now();
    ids.copy_(input);if(capture)replay->replay();else execute();synchronize(placement);
    const auto elapsed=pdg_scale::seconds(begin);
    if(training && !finite.item<bool>())throw std::runtime_error("nonfinite update or bounded optimizer capacity exceeded");
    for(const auto& logits:window.logits)if(!at::isfinite(logits.data).all().item<bool>())throw std::runtime_error("nonfinite bounded logits");
    auto metrics=memory(placement);
    const auto peer=replay?replay->peer_inventory():std::pair<size_t,int64_t>{0,0};
    metrics.insert({{"perf/window_seconds",elapsed},{"perf/ms_per_sample_token",elapsed*1000/(c.batch*c.steps)},
      {"perf/sample_tokens_per_second",c.batch*c.steps/elapsed},{"perf/construction_seconds",construction},
      {"perf/prepare_seconds",prepare},{"memory/workspace_bound_bytes",double(program.workspace_bound())},
      {"runtime/devices",double(devices)},{"runtime/graph_replay",double(capture)},
      {"runtime/peer_channels",double(peer.first)},{"memory/retained_peer_buffer_bytes",double(peer.second)},
      {"placement/cut_edges",double(placement.cut_edges)},
      {"placement/cut_fraction",placement.edges?double(placement.cut_edges)/placement.edges:0.},
      {"placement/node_load_limit_bytes",double(placement.node_load_limit)},
      {"work/window_tokens",double(c.steps)},{"work/batch",double(c.batch)}});
    if(training)metrics["train/loss"]=loss.data.item<double>();
    for(const auto& [name,value]:fixture.inventory)metrics["model/"+name]=value;
    writer.Write(iteration,metrics,pdg_scale::seconds(start),{{"phase",std::string(iteration<iteration_warmup?"warmup":"measure")},
      {"dtype",portable_torch::dtype_name(c.runtime.dtype)},{"mode",std::string(training?"training":"inference")},
      {"scheduler",std::string(capture?"bounded-replay":"bounded-eager")},{"window_boundary",std::string("reset")}});
    std::cout<<"WINDOW "<<iteration<<" seconds="<<elapsed<<" training="<<training<<" replay="<<capture<<'\n'<<std::flush;
  }
  return 0;
}
}
int main(int argc,char** argv) {
  int result=0;try{result=run(argc,argv);}catch(const std::exception& e){std::cerr<<e.what()<<'\n';result=1;}
  if(used_npu)try{accelerator_scale::finalize();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';result=1;}
  return result;
}
