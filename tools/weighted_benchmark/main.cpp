// Standalone LibTorch timing. Python is an optional static-fixture author,
// never a runtime scheduler, loss callback or source of selected actions.
#include "benchmark.h"
#include "observation.h"
#include "tide/greedy.h"
#include "tide/stream.h"
#include "tide/optimizer.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <cmath>
#include <torch/csrc/autograd/autograd.h>
#include <filesystem>
#include <fstream>
#include <iostream>
namespace {
using namespace weighted_benchmark;
struct Eager {
  Fixture data;Continuation state;ParameterRegistry registry;std::unique_ptr<SGD> optimizer;
  bool training,prefill;Options options;
  Eager(const Json& j,bool train,bool batch,bool scalar=false):data(fixture(j)),registry(data.model.parameters()),training(train),prefill(batch){
    state.identity=data.graph.identity;state.batch_size=data.samples;
    options.packed=!scalar;options.full_autograd=options.aggregate_autograd=scalar?"replay":"batched";
    OptimizerGroup group;for(const auto& p:registry.owners())group.parameters.push_back(p.canonical);
    group.lr=.001;group.momentum=.3;group.weight_decay=.01;
    optimizer=std::make_unique<SGD>(registry,std::vector<OptimizerGroup>{group});
  }
  Step step(Index index){
    at::AutoGradMode grad(training);optimizer->zero_grad();auto xs=inputs(data,index);
    auto r=prefill?Greedy(data.graph,data.model,options).run(state,xs,(index+1)*data.span,(index+1)*data.span)
                  :Streaming(data.graph,data.model,options).run(state,xs,(index+1)*data.span,(index+1)*data.span);
    auto value=loss(r);
    if(training){torch::autograd::backward({value});bool connected=false;
      for(const auto& p:registry.owners())if(p.value.grad().defined()){
        connected=true;if(!at::isfinite(p.value.grad()).all().item<bool>())throw std::runtime_error("nonfinite benchmark gradient");}
      if(!connected)throw std::runtime_error("benchmark has no connected parameter gradient");optimizer->step();}
    state=detached(r.continuation);return {detached(r),value.item<double>()};
  }
  std::map<std::string,Tensor> parameters()const{
    std::map<std::string,Tensor> out;for(const auto& p:registry.owners())out[p.canonical]=p.value.detach().clone();return out;
  }
};
}
int main(int argc,char** argv){
  Json report{{"state","starting"}};std::filesystem::path output;bool created=false;
  try{
    std::string input,device,implementation="native",mode="training",schedule="prefill";Index steps=3;
    for(int i=1;i<argc;++i){std::string key=argv[i];if(key=="--help"){
        std::cout<<"--fixture JSON --device cpu|npu:N|cuda:N --implementation native|resident --mode inference|training --schedule streaming|prefill --steps 1..8 --output-dir NEW\n";return 0;}
      if(i+1==argc)throw std::invalid_argument("missing benchmark option value");std::string value=argv[++i];
      if(key=="--fixture")input=value;else if(key=="--device")device=value;else if(key=="--implementation")implementation=value;
      else if(key=="--mode")mode=value;else if(key=="--schedule")schedule=value;else if(key=="--steps")steps=std::stoll(value);
      else if(key=="--output-dir")output=value;else throw std::invalid_argument("unknown benchmark option "+key);
    }
    if(input.empty()||device.empty()||output.empty()||steps<1||steps>8||(implementation!="native"&&implementation!="resident")
        ||(mode!="inference"&&mode!="training")||(schedule!="streaming"&&schedule!="prefill"))throw std::invalid_argument("invalid bounded benchmark request");
    if(implementation=="native"&&device!="cpu")throw std::invalid_argument("standalone eager measurement requires explicit CPU; resident measures accelerator");
    if(implementation=="resident"&&(device=="cpu"||device=="auto"))throw std::invalid_argument("resident measurement requires an indexed accelerator");
    input=std::filesystem::absolute(input);output=std::filesystem::absolute(output);
    if(!std::filesystem::create_directory(output))throw std::invalid_argument("benchmark output directory must be new");
    created=true;
    std::filesystem::current_path(output);
    auto runtime_start=tide_bench::Clock::now();portable_torch::RuntimeSession runtime;
    portable_torch::RuntimeOptions runtime_options;runtime_options.device_spec=device;runtime_options.dtype=at::kFloat;
    const auto target=portable_torch::resolve_device(runtime_options);at::set_num_threads(1);at::set_num_interop_threads(1);
    runtime_probe(target);const auto runtime_seconds=tide_bench::seconds(runtime_start);
    report={{"state","running"},{"schema","tide-weighted-standalone-benchmark-v1"},
      {"device",device},{"backend",portable_torch::compiled_backend()},{"dtype","float32"},
      {"runtime_initialization_seconds",runtime_seconds},
      {"implementation",implementation},{"mode",mode},{"schedule",schedule},{"warmup_windows",1},{"measured_windows",steps},
      {"timing","input clone, schedule/packing, complete CPU return, loss, backward, finite checks, SGD, detach, sync"},
      {"excluded","static fixture/owner/optimizer construction, warmup, post-measurement independent check, final parameter export"},
      {"metrics",Json::array()}};write(output/"result.json",report);
    capture_memory(report,target,"runtime_ready",true);
    auto fixture_start=tide_bench::Clock::now();
    std::ifstream file(input);if(!file)throw std::invalid_argument("missing static fixture");Json data=Json::parse(file);
    data["__fixture_root"]=std::filesystem::path(input).parent_path().string();
    report["fixture_sha256"]=data.at("fixture_sha256");report["config"]=data.at("config");
    if(data.contains("workload")){
      report["parameter_inventory"]=data["workload"]["inventory"];report["memory_floor"]=data["workload"]["memory_floor"];
      if(data["workload"].contains("parameter_match"))report["parameter_match"]=data["workload"]["parameter_match"];
    }
    report["fixture_preparation_seconds"]=tide_bench::seconds(fixture_start);
    std::unique_ptr<Eager> eager;std::unique_ptr<Resident> remote;
    const bool training=mode=="training",prefill=schedule=="prefill";auto start=tide_bench::Clock::now();
    if(implementation=="native")eager=std::make_unique<Eager>(data,training,prefill);
    else {
#ifdef TIDE_WEIGHTED_RESIDENT
      remote=resident(data,target,training,prefill);
#else
      throw std::invalid_argument("this standalone build has no resident consumer");
#endif
    }
    portable_torch::synchronize(target);report["construction_seconds"]=tide_bench::seconds(start);
    capture_memory(report,target,"construction",true);
    std::vector<Step> actual;auto execute=[&](Index i){return eager?eager->step(i):remote->step(i);};
    start=tide_bench::Clock::now();actual.push_back(execute(0));portable_torch::synchronize(target);report["warmup_seconds"]=tide_bench::seconds(start);
    capture_memory(report,target,"warmup",true);
    for(Index step=1;step<=steps;++step){portable_torch::synchronize(target);start=tide_bench::Clock::now();actual.push_back(execute(step));
      portable_torch::synchronize(target);const auto seconds=tide_bench::seconds(start);
      auto event=work(actual.back().result);event["step"]=step;event["seconds"]=seconds;event["loss"]=actual.back().loss;
      report["metrics"].push_back(event);write(output/"result.json",report);std::cout<<"step="<<step<<" seconds="<<seconds<<'\n'<<std::flush;
    }
    capture_memory(report,target,"measurement");
    const auto parameters=eager?eager->parameters():remote->parameters();
    // A separate scalar schedule runs only after timing and never guides the candidate.
    Eager reference(data,training,!prefill,true);
    for(Index i=0;i<=steps;++i){const auto expected=reference.step(i);tide_bench::compare(actual[i].result,expected.result,true,at::kFloat);
      if(std::abs(actual[i].loss-expected.loss)>1e-6+1e-5*std::abs(expected.loss))throw std::runtime_error("benchmark objective differs");}
    if(!parameters.empty())for(const auto& [name,value]:reference.parameters())
      if(!at::allclose(parameters.at(name),value,1e-5,1e-6))throw std::runtime_error("benchmark final optimizer parameter differs: "+name);
    if(remote)remote->close();report["state"]="passed";report["post_measurement_complete_records"]=true;
    report["parameter_values_checked"]=!parameters.empty();report["optimizer_values_checked"]=training&&!parameters.empty();write(output/"result.json",report);return 0;
  }catch(const std::exception& e){report["state"]="failed";report["error"]=e.what();
    if(created)try{write(output/"result.json",report);}catch(const std::exception& write_error){std::cerr<<write_error.what()<<'\n';}
    std::cerr<<e.what()<<'\n';return 2;}
}
