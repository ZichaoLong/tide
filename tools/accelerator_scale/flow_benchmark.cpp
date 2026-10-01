#include "flow_config.h"
#include "bounded_optimizer.h"
#include "graph_replay.h"
#include "precision.h"
#include "peer_transport.h"
#include "../../cpp/bench/metrics_jsonl_writer.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/resource.h>

namespace accelerator_scale::flows {
namespace {
double count_masks(const std::vector<Tensor>& values) {
  std::map<std::string,std::vector<Tensor>> groups;
  for(const auto& v:values)groups[v.device().str()].push_back(v);
  double count=0.;for(const auto& [device,rows]:groups)count+=at::stack(rows).sum(at::kLong).item<int64_t>();
  return count;
}
double rss() {rusage r{};if(getrusage(RUSAGE_SELF,&r))throw std::runtime_error("getrusage failed");return double(r.ru_maxrss)*1024.;}
void publish_placement(const Config& c,const Fixture& f,const Placement& p) {
  std::ofstream out(std::filesystem::path(c.model.runtime.output_dir)/"placement.json");
  out<<"{\"schema\":\"tide-complete-flow-placement-v1\",\"family\":\""<<c.family<<"\",\"flow\":\""<<c.flow
    <<"\",\"scheduler\":\""<<c.scheduler<<"\",\"period\":"<<f.period<<",\"node_shards\":[";
  for(size_t i=0;i<p.node_device.size();++i)out<<(i?",":"")<<p.node_device[i];
  out<<"],\"parameter_bytes\":[";for(size_t i=0;i<p.parameter_bytes.size();++i)out<<(i?",":"")<<p.parameter_bytes[i];
  out<<"],\"cut_edges\":"<<p.cut_edges<<",\"physical_edges\":"<<p.edges<<"}\n";
  out.close();if(!out)throw std::runtime_error("complete-flow placement publication failed");
}
}
void benchmark(const Config& c,const Topology& topology,at::Device device) {
  using pdg_scale::Clock;using pdg_scale::seconds;
  at::AutoGradMode grad(c.training);
  const bool bounded=c.scheduler.rfind("bounded-",0)==0,capture=c.scheduler=="bounded-replay";
  const auto construction_start=Clock::now();const auto contexts=initialize_devices(device,c.devices);
  std::unique_ptr<GraphReplay> replay;
  if(capture){std::vector<at::Device> targets;for(Index i=0;i<c.devices;++i)targets.emplace_back(device.type(),device.index()+i);
    replay=std::make_unique<GraphReplay>(targets);}
  auto f=seeded(c,topology);auto placement=placed(f,c,device);publish_placement(c,f,placement);
  std::unique_ptr<PeerTransport> eager_peer;
  if(bounded && !capture && device.type()==c10::DeviceType::PrivateUse1 && c.devices>1)
    eager_peer=std::make_unique<PeerTransport>(placement.devices);
  const bounded::Limits limits{c.model.steps,c.model.batch,c.workspace_gib*(int64_t(1)<<30),false,c.training};
  std::unique_ptr<bounded::Program> program;
  if(bounded)program=std::make_unique<bounded::Program>(f.values,schedule(f,limits,c.training),placement,limits);
  TrainingConfig train;train.optimizer=c.optimizer;train.loss_scale=c.model.runtime.dtype==at::kHalf?128.:1.;
  train.backward_threads=c.backward_threads;train.optimizer_threads=c.optimizer_threads;
  std::unique_ptr<TrainingOwners> owners;std::unique_ptr<bounded::Optimizer> optimizer;
  if(c.training) {
    if(bounded)optimizer=std::make_unique<bounded::Optimizer>(f.values.owners,c.optimizer,1e-4,train.loss_scale,c.iterations+4);
    else owners=std::make_unique<TrainingOwners>(f.values,train);
  }
  auto ids=at::zeros({c.model.steps,c.model.batch},f.values.embedding.options().dtype(at::kLong));
  bounded::Window window;bounded::Value loss;HostWindow host;Tensor finite;
  auto execute=[&] {
    if(program) {
      window=program->run(ids);
      if(optimizer){loss=program->loss(window,ids);auto gradients=program->vjp(loss,at::ones_like(loss.data)*train.loss_scale,false);
        finite=optimizer->step(loss,gradients);}
    } else {
      if(owners)owners->zero_grad();host=host_window(c.model,f,placement,c.scheduler,options(c,false),ids);
      if(owners){owners->backward(host.loss);owners->step();}
    }
  };
  synchronize(placement);const auto construction=seconds(construction_start)+c.runtime_setup_seconds;
  const auto prepare_start=Clock::now();
  if(capture) {
    execute();synchronize(placement);if(optimizer)optimizer->reset();
    window={};loss={};finite=Tensor();
    replay->capture(execute);if(optimizer)optimizer->reset();synchronize(placement);
  }
  const auto prepare=seconds(prepare_start);const auto start=Clock::now();double cumulative=0.;
  portable_experiment::MetricsJsonlWriter writer(std::filesystem::path(c.model.runtime.output_dir)/"metrics.jsonl",c.model.run_id);
  for(Index iteration=0;iteration<c.iterations;++iteration) {
    synchronize(placement);reset_memory(placement);transfers.reset();
    ProfileScope profile(c.profile,placement.devices,iteration,"window");
    const auto begin=Clock::now();
    // Input generation/upload is part of every independent request. Captured
    // execution receives new values, never a CPU oracle's decisions or answers.
    auto input=(at::arange(c.model.steps).unsqueeze(1)*7+at::arange(c.model.batch).unsqueeze(0)*3+iteration*5)
      .remainder(c.model.vocab).to(at::kLong);
    ids.copy_(input);
    if(capture)replay->replay();else execute();
    synchronize(placement);
    double value=0.;
    if(c.training) {
      value=(program?loss.data:host.loss).item<double>();
      if(!std::isfinite(value) || (optimizer && !finite.item<bool>()))throw std::runtime_error("nonfinite complete training update");
    }
    std::vector<Tensor> checks;
    if(program)for(const auto& x:window.logits)checks.push_back(at::isfinite(x.data).all());
    else for(const auto& x:host.logits)checks.push_back(at::isfinite(x).all());
    if(!at::stack(checks).all().item<bool>())throw std::runtime_error("nonfinite complete-flow logits");
    // Retain only small diagnostic masks until after timing. Eager request
    // activation/gradient destruction and device completion belong to latency.
    auto work=host.result.stats;std::vector<Tensor> candidates,selected,edges,outputs;
    if(program){candidates=window.candidates;selected=window.selected;edges=window.edge_presence;outputs=window.output_present;}
    if(!capture){window={};loss={};finite=Tensor();host={};}if(owners)owners->zero_grad();
    input=Tensor();checks.clear();synchronize(placement);
    const auto elapsed=seconds(begin);cumulative+=elapsed;
    profile.finish();
    auto metrics=memory(placement);auto traffic=transfers.metrics();metrics.insert(traffic.begin(),traffic.end());
    if(program){work["candidate_events"]=count_masks(candidates);work["selected_events"]=count_masks(selected);
      work["visited_edges"]=count_masks(edges);work["output_rows"]=count_masks(outputs);}
    const auto peer=replay?replay->peer_inventory():std::pair<size_t,int64_t>{0,0};
    metrics.insert({{"perf/window_seconds",elapsed},{"perf/ms_per_sample_token",elapsed*1000/(c.model.batch*c.model.steps)},
      {"perf/sample_tokens_per_second",c.model.batch*c.model.steps/elapsed},{"perf/construction_seconds",construction},
      {"perf/prepare_seconds",prepare},{"perf/cold_through_current_seconds",construction+prepare+cumulative},
      {"perf/amortized_window_seconds",(construction+prepare+cumulative)/(iteration+1)},
      {"memory/process_peak_rss_bytes",rss()},{"memory/workspace_bound_bytes",program?double(program->workspace_bound()):0.},
      {"runtime/devices",double(c.devices)},{"runtime/graph_replay",double(capture)},
      {"runtime/peer_notifications",double(peer.first)},{"memory/retained_peer_buffer_bytes",double(peer.second)},
      {"runtime/workers",double(c.model.workers)},{"runtime/head_workers",double(c.model.head_workers)},
      {"runtime/aten_threads",double(c.model.threads)},{"runtime/backward_threads",double(c.backward_threads)},
      {"runtime/optimizer_threads",double(c.optimizer_threads)},{"placement/cut_edges",double(placement.cut_edges)},
      {"placement/cut_fraction",placement.edges?double(placement.cut_edges)/placement.edges:0.},
      {"work/window_tokens",double(c.model.steps)},{"work/batch",double(c.model.batch)}});
    if(c.training){metrics["train/loss"]=value;metrics["train/loss_scale"]=train.loss_scale;}
    for(const auto& [key,v]:f.values.inventory)metrics["model/"+key]=v;
    for(const auto& [key,v]:work)metrics["work/"+key]=v;
    writer.Write(iteration,metrics,seconds(start),{{"phase",std::string(iteration<c.iteration_warmup?"warmup":"measure")},
      {"dtype",portable_torch::dtype_name(c.model.runtime.dtype)},{"mode",std::string(c.training?"training":"inference")},
      {"family",c.family},{"flow",c.flow},{"scheduler",c.scheduler},{"memory",c.model.memory},{"optimizer",c.optimizer},
      {"window_boundary",std::string("reset")},{"read_device",c.scoring.read_device},{"read_dtype",c.scoring.dtype_name()},
      {"control_device",c.scoring.control_device},{"ranking_device",c.ranking},{"event_device",c.events}});
    std::cout<<"WINDOW "<<iteration<<" seconds="<<elapsed<<" training="<<c.training<<" scheduler="<<c.scheduler<<'\n'<<std::flush;
  }
}
}  // namespace accelerator_scale::flows
