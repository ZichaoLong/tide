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
#include <set>
#include "eager.h"
#include "../device_online/weighted_precision_reference.h"
using namespace weighted_benchmark;
int main(int argc,char** argv){
  Json report{{"state","starting"}};std::filesystem::path output;bool created=false;
  try{
    std::string input,device,implementation="native",mode="training",schedule="prefill",precision="float32",
      placement_name="native",scoring,packing="auto",limits_path;Index steps=3,threads=1,interop=1,workers=1;bool checkpoint=false;
    std::vector<std::string> peer_specs;
    for(int i=1;i<argc;++i){std::string key=argv[i];if(key=="--help"){
        std::cout<<"--fixture JSON --device cpu|npu:N|cuda:N --dtype float16|float32|float64 --scoring-dtype float32|float64 --placement native|mixed-a|mixed-b|mixed-c --intraop-threads N --interop-threads N --workers N --packing auto|packed|scalar --peer-device DEVICE (repeatable) --checkpoint-check true|false --implementation native|resident --mode inference|training --schedule streaming|prefill --steps 1..8 --output-dir NEW\n";return 0;}
      if(i+1==argc)throw std::invalid_argument("missing benchmark option value");std::string value=argv[++i];
      if(key=="--fixture")input=value;else if(key=="--device")device=value;else if(key=="--implementation")implementation=value;
      else if(key=="--mode")mode=value;else if(key=="--schedule")schedule=value;else if(key=="--steps")steps=std::stoll(value);
      else if(key=="--dtype")precision=value;
      else if(key=="--placement")placement_name=value;else if(key=="--scoring-dtype")scoring=value;
      else if(key=="--intraop-threads")threads=std::stoll(value);else if(key=="--interop-threads")interop=std::stoll(value);
      else if(key=="--workers")workers=std::stoll(value);else if(key=="--packing")packing=value;
      else if(key=="--peer-device")peer_specs.push_back(value);
      else if(key=="--resident-limits")limits_path=value;
      else if(key=="--checkpoint-check"){if(value!="true"&&value!="false")throw std::invalid_argument("checkpoint flag must be true/false");checkpoint=value=="true";}
      else if(key=="--output-dir")output=value;else throw std::invalid_argument("unknown benchmark option "+key);
    }
    if(input.empty()||device.empty()||output.empty()||steps<1||steps>8||(implementation!="native"&&implementation!="resident")
        ||(mode!="inference"&&mode!="training")||(schedule!="streaming"&&schedule!="prefill")
        ||(precision!="float16"&&precision!="float32"&&precision!="float64")||threads<1||interop<1||workers<1)throw std::invalid_argument("invalid bounded benchmark request");
    if(precision=="float64"&&(device!="cpu"||implementation!="native"))throw std::invalid_argument("FP64 benchmark requires CPU eager execution; no implicit substitution");
    const auto dtype=precision=="float64"?at::kDouble:precision=="float16"?at::kHalf:at::kFloat;
    if(scoring.empty())scoring=dtype==at::kDouble?"float64":"float32";
    if(implementation=="resident"&&scoring!="float32")throw std::invalid_argument("resident scoring requires float32");
    if(scoring!="float32"&&scoring!="float64")throw std::invalid_argument("invalid scoring dtype");
    if(packing!="auto"&&packing!="packed"&&packing!="scalar")throw std::invalid_argument("invalid packing");
    if(checkpoint&&mode!="training")throw std::invalid_argument("disk checkpoint check requires training");
    if(implementation=="resident"&&(placement_name!="native"||packing=="scalar"||workers!=1))throw std::invalid_argument("resident does not accept eager placement/scalar/host workers");
    if(implementation=="resident"&&(device=="cpu"||device=="auto"))throw std::invalid_argument("resident measurement requires an indexed accelerator");
    input=std::filesystem::absolute(input);output=std::filesystem::absolute(output);
    if(!limits_path.empty())limits_path=std::filesystem::absolute(limits_path);
    if(!std::filesystem::create_directory(output))throw std::invalid_argument("benchmark output directory must be new");
    created=true;
    std::filesystem::current_path(output);
    auto runtime_start=tide_bench::Clock::now();portable_torch::RuntimeSession runtime;
    portable_torch::RuntimeOptions runtime_options;runtime_options.device_spec=device;runtime_options.dtype=dtype;
    runtime_options.allow_npu_float16=true;
    const auto target=portable_torch::resolve_device(runtime_options);at::set_num_threads(threads);at::set_num_interop_threads(interop);
    ExecutionPlacement placement;placement.preset=implementation=="resident"?"resident":placement_name;placement.scoring_dtype=scoring;
    auto effective=resolve_placement(placement,target).record();
    std::vector<at::Device> peers;for(const auto& spec:peer_specs){auto opt=runtime_options;opt.device_spec=spec;peers.push_back(portable_torch::resolve_device(opt));}
    if(!peers.empty()){std::set<std::string> names;for(auto d:peers){names.insert(d.str());if(d.type()!=target.type())throw std::invalid_argument("peer backend mismatch");}
      if(target.is_cpu()||peers.size()<2||peers.front()!=target||names.size()!=peers.size())throw std::invalid_argument("invalid distinct peers/coordinator");}
    runtime_probe(target,dtype);for(auto d:peers)if(d!=target)runtime_probe(d,dtype);const auto runtime_seconds=tide_bench::seconds(runtime_start);
    report={{"state","running"},{"schema","tide-weighted-standalone-benchmark-v1"},
      {"device",device},{"backend",portable_torch::compiled_backend()},{"dtype",precision},{"requested_dtype",precision},
      {"fixture_dtype","float32"},{"input_policy","same hashed FP32 material; explicit promotion before both schedules"},
      { "runtime_initialization_seconds",runtime_seconds},{"placement",placement_name},{"effective_placement",effective},
      {"scoring_dtype",scoring},{"intraop_threads",threads},{"interop_threads",interop},{"workers",workers},{"packing",packing},{"peer_devices",peer_specs},
      {"implementation",implementation},{"mode",mode},{"schedule",schedule},{"warmup_windows",1},{"measured_windows",steps},
      {"timing","input clone, schedule/packing, complete CPU return, loss, backward, finite checks, SGD, detach, sync"},
      {"excluded","static fixture/owner/optimizer construction, warmup, post-measurement independent check, final parameter/gradient/optimizer export"},
      {"phase","fixture-preparation"},{"metrics",Json::array()}};write(output/"result.json",report);
    capture_memory(report,target,"runtime_ready",true,peers);
    auto fixture_start=tide_bench::Clock::now();
    std::ifstream file(input);if(!file)throw std::invalid_argument("missing static fixture");Json data=Json::parse(file);
    if(!limits_path.empty()){
      if(implementation!="resident")throw std::invalid_argument("capacity overrides require resident");
      std::ifstream limits_file(limits_path);if(!limits_file)throw std::invalid_argument("missing limits file");
      auto limits=Json::parse(limits_file);if(!limits.is_object())throw std::invalid_argument("limits must be object");
      for(const auto& [key,value]:limits.items()){
        if(!data.at("limits").contains(key)||!value.is_number_integer()||value.is_boolean()||value.get<int64_t>()<1)
          throw std::invalid_argument("invalid capacity override: "+key);
        if(mode!="training"&&(key=="retained_bytes"||key=="backward_bytes"||key=="optimizer_bytes"||key=="total_bytes"))
          throw std::invalid_argument("training limit on inference");
        data["limits"][key]=value;
      }
      report["resident_limit_overrides"]=limits;
    }
    report["resident_limits"]=data.at("limits");
    data["__fixture_root"]=std::filesystem::path(input).parent_path().string();
    report["fixture_sha256"]=data.at("fixture_sha256");report["config"]=data.at("config");
    if(data.contains("workload")){
      report["model_id"]=data["workload"].value("model_id","");report["workload_id"]=data["workload"]["sha256"];
      report["parameter_inventory"]=data["workload"]["inventory"];report["memory_floor"]=data["workload"]["memory_floor"];
      const auto& floor=data["workload"]["memory_floor"];
      report["effective_static_tensor_bytes"]=(floor["logical_parameter_bytes"].get<int64_t>()+
        floor["legacy_inert_buffer_bytes"].get<int64_t>())*(dtype==at::kDouble?2.:dtype==at::kHalf?.5:1.);
      if(data["workload"].contains("parameter_match"))report["parameter_match"]=data["workload"]["parameter_match"];
    }
    report["fixture_preparation_seconds"]=tide_bench::seconds(fixture_start);
    std::unique_ptr<Eager> eager;std::unique_ptr<Resident> remote;
    const bool training=mode=="training",prefill=schedule=="prefill";
    report["phase"]="construction";write(output/"result.json",report);
    auto start=tide_bench::Clock::now();
    if(implementation=="native")eager=std::make_unique<Eager>(data,training,prefill,packing=="scalar",dtype,target,placement,workers,peers);
    else {
#ifdef TIDE_WEIGHTED_RESIDENT
      remote=resident(data,target,training,prefill,dtype,peers);
#else
      throw std::invalid_argument("this standalone build has no resident consumer");
#endif
    }
    portable_torch::synchronize(target);for(auto d:peers)if(d!=target)portable_torch::synchronize(d);report["construction_seconds"]=tide_bench::seconds(start);
    capture_memory(report,target,"construction",true,peers);
    std::vector<Step> actual;auto execute=[&](Index i){report["phase"]=i?"measurement":"warmup";report["window"]=i;
      return eager?eager->step(i):remote->step(i);};
    report["phase"]="warmup";report["window"]=0;write(output/"result.json",report);
    start=tide_bench::Clock::now();actual.push_back(execute(0));portable_torch::synchronize(target);for(auto d:peers)if(d!=target)portable_torch::synchronize(d);report["warmup_seconds"]=tide_bench::seconds(start);
    capture_memory(report,target,"warmup",true,peers);
    for(Index step=1;step<=steps;++step){
      // Persist progress before synchronization/timing, never inside a step.
      report["phase"]="measurement";report["window"]=step;write(output/"result.json",report);
      portable_torch::synchronize(target);for(auto d:peers)if(d!=target)portable_torch::synchronize(d);start=tide_bench::Clock::now();actual.push_back(execute(step));
      portable_torch::synchronize(target);for(auto d:peers)if(d!=target)portable_torch::synchronize(d);const auto seconds=tide_bench::seconds(start);
      auto event=work(actual.back().result);event["step"]=step;event["seconds"]=seconds;event["loss"]=actual.back().loss;
      report["metrics"].push_back(event);write(output/"result.json",report);std::cout<<"step="<<step<<" seconds="<<seconds<<'\n'<<std::flush;
    }
    capture_memory(report,target,"measurement",false,peers);
    report["phase"]="validation-preparation";write(output/"result.json",report);
    // Inference does not update parameters. Avoid exporting a second complete
    // bank, and release the candidate before constructing the independent run.
    const auto parameters=training?(eager?eager->parameters():remote->parameters()):std::map<std::string,Tensor>{};
    const bool checked_training=training;
    const auto trained=training?(eager?eager->training_state():remote->training_state()):TrainingState{};
    if(checkpoint){
      const auto path=output/"resume.pt";Step a,b;TrainingState x,y;std::map<std::string,Tensor> px,py;
      if(eager){
        eager->save(path);Eager restored(data,training,prefill,packing=="scalar",dtype,target,placement,workers,peers);restored.restore(path);
        a=eager->step(steps+1);b=restored.step(steps+1);x=eager->training_state();y=restored.training_state();
        px=eager->parameters();py=restored.parameters();
      }else{
#ifdef TIDE_WEIGHTED_RESIDENT
        remote->save(path);a=remote->step(steps+1);x=remote->training_state();px=remote->parameters();
        remote->close();remote.reset();
        auto restored=resident(data,target,training,prefill,dtype,peers);restored->restore(path);
        b=restored->step(steps+1);y=restored->training_state();py=restored->parameters();restored->close();
#endif
      }
      tide_bench::compare(a.result,b.result,true,dtype,{},0.,0.);
      if(a.loss!=b.loss)throw std::runtime_error("restored objective differs");
      compare_tensors(px,py,dtype,"restored parameters",true);
      compare_tensors(x.gradients,y.gradients,dtype,"restored gradient",true);
      compare_tensors(x.momentum,y.momentum,dtype,"restored momentum",true);
      compare_tensors(x.masters,y.masters,dtype,"restored master",true);
      report["exact_resume"]={{"state","passed"},{"checkpoint","resume.pt"},{"next_window",steps+1},{"exact",true}};
    }
    if(remote)remote->close();eager.reset();remote.reset();
    // A separate scalar schedule runs only after timing and never guides the candidate.
    ExecutionPlacement cpu;cpu.scoring_dtype=scoring;
    const bool storage_reference=implementation=="resident"&&dtype==at::kHalf;
    Eager reference(data,training,!prefill,true,dtype,at::Device(at::kCPU),cpu,1,{},storage_reference);
    report["reference_precision"]=storage_reference?"fp16-storage-fp32-arithmetic":"same-dtype-eager";
    for(Index i=0;i<=steps;++i){report["phase"]="independent-check";report["window"]=i;
      write(output/"result.json",report);
      const auto expected=reference.step(i);
      if(storage_reference)tide::device_online::weighted_precision_check::validate_storage(actual[i].result,dtype);
      tide_bench::compare(actual[i].result,expected.result,true,dtype,{},dtype==at::kHalf?2e-2:-1.,dtype==at::kHalf?1e-3:-1.);
      if(std::abs(actual[i].loss-expected.loss)>(dtype==at::kDouble?1e-10:dtype==at::kHalf?1e-3:1e-6)+(dtype==at::kDouble?1e-8:dtype==at::kHalf?2e-2:1e-5)*std::abs(expected.loss))throw std::runtime_error("benchmark objective differs");}
    if(!parameters.empty())compare_tensors(parameters,reference.parameters(),dtype,"final parameter");
    if(checked_training){const auto expected=reference.training_state();
      compare_tensors(trained.gradients,expected.gradients,dtype,"final parameter gradient");
      compare_tensors(trained.momentum,expected.momentum,dtype,"final SGD momentum");
      compare_tensors(trained.masters,expected.masters,dtype,"final FP32 master");}
    report["state"]="passed";report["phase"]="complete";report["post_measurement_complete_records"]=true;
    report["parameter_values_checked"]=!parameters.empty();report["optimizer_values_checked"]=checked_training;
    report["final_parameter_gradients_checked"]=checked_training;write(output/"result.json",report);return 0;
  }catch(const std::exception& e){report["state"]="failed";report["error"]=e.what();
    if(created)if(const auto* failure=dynamic_cast<const tide::weighted::AllocationFailure*>(&e))try{
      auto record=allocation_failure(*failure);record["execution"]={{"phase",report.value("phase","")},{"window",report.value("window",Index(-1))}};
      write(output/"allocation-failure.json",record);report["allocation_failure"]="allocation-failure.json";
    }catch(const std::exception& write_error){report["allocation_failure_write_error"]=write_error.what();}
    if(created)try{write(output/"result.json",report);}catch(const std::exception& write_error){std::cerr<<write_error.what()<<'\n';}
    std::cerr<<e.what()<<'\n';return 2;}
}
