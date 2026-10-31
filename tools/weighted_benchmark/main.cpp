// Standalone LibTorch experiment. Static fixture authors never schedule a run.
#include "cli.h"
#include "eager.h"
#include "observation.h"
#include "work_observation.h"
#include "run_checkpoint.h"
#include "validation.h"
#include "roots.h"
#include "tide/communication.h"
#include "../../cpp/bench/streaming.h"
#include "../device_online/weighted_precision_reference.h"
#include <ATen/Parallel.h>
#include <ATen/Context.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
using namespace weighted_benchmark;
namespace {
Json read_json(const std::filesystem::path& path) {
  std::ifstream stream(path);if(!stream)throw std::invalid_argument("cannot read "+path.string());return Json::parse(stream);
}
void limits(Json& data,const Request& request,Json& report) {
  if(request.limits.empty())return;
  if(request.implementation!="resident")throw std::invalid_argument("capacity overrides require resident");
  const auto values=read_json(request.limits);
  apply_resident_limits(data.at("limits"),values,request.mode=="training",request.schedule=="prefill");
  report["resident_limit_overrides"]=values;
}
}
int main(int argc,char** argv) {
  Json report{{"state","starting"}};std::filesystem::path output;bool created=false;
  try {
    const auto request=parse_request(argc,argv);if(request.help){help();return 0;}
    const auto& r=request;output=r.output;const auto dtype=r.dtype();
    const bool training=r.mode=="training",prefill=r.schedule=="prefill",validate=r.validation!="none";
    const Index windows=r.warmup+r.steps;
    // Scope the explicit copy policy to this process before any model transfer.
    if(setenv("TIDE_COMMUNICATION_POLICY",r.communication.c_str(),1))throw std::runtime_error("cannot set process communication policy");
    const auto fixture_start=tide_bench::Clock::now();
    auto data=read_json(r.input);data["__fixture_root"]=r.input.parent_path().string();data["__task"]=r.task();
    if(r.check_roots)validate_root_budget(data);
    const auto fixture_seconds=tide_bench::seconds(fixture_start);
    const Json identity={{"fixture_sha256",data.at("fixture_sha256")},{"mode",r.mode},{"dtype",r.precision},
      {"scoring_dtype",r.scoring},{"implementation",r.implementation},{"span",data.at("span")},{"task",r.task()}};
    if(!r.resume.empty())checkpoint_manifest(r.resume,identity); // Reject mismatched input before allocating a model.
    if(!std::filesystem::create_directory(output))throw std::invalid_argument("output directory must be new");created=true;
    std::filesystem::current_path(output);
    auto begin=tide_bench::Clock::now();portable_torch::RuntimeSession runtime;
    portable_torch::RuntimeOptions runtime_options;runtime_options.device_spec=r.device;runtime_options.dtype=dtype;runtime_options.allow_npu_float16=true;
    const auto target=portable_torch::resolve_device(runtime_options);at::set_num_threads(r.threads);at::set_num_interop_threads(r.interop);
    if(target.is_cuda()){at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);}
    ExecutionPlacement placement;placement.preset=r.implementation=="resident"?"resident":r.placement;placement.scoring_dtype=r.scoring;
    const auto effective=resolve_placement(placement,target).record();
    std::vector<at::Device> peers,nodes,devices{target};
    auto resolve=[&](const std::string& name){auto opt=runtime_options;opt.device_spec=name;auto d=portable_torch::resolve_device(opt);
      if(d.type()!=target.type()||(!d.is_cpu()&&d.index()<0))throw std::invalid_argument("node/peer devices require one indexed backend");
      if(std::find(devices.begin(),devices.end(),d)==devices.end())devices.push_back(d);return d;};
    for(const auto& name:r.peers)peers.push_back(resolve(name));for(const auto& name:r.nodes)nodes.push_back(resolve(name));
    if(!peers.empty()){std::set<std::string> names;for(auto d:peers)names.insert(d.str());
      if(target.is_cpu()||peers.size()<2||peers.front()!=target||names.size()!=peers.size())throw std::invalid_argument("invalid distinct peers/coordinator");}
    if(!nodes.empty()&&nodes.size()!=data.at("graph").at("nodes").size())throw std::invalid_argument("node-devices must cover every physical node");
    if(nodes.empty())for(const auto& node:data.at("graph").at("nodes"))
      nodes.push_back(peers.empty()?target:peers.at(node.at("region").get<size_t>()%peers.size()));
    if(dtype==at::kDouble&&devices.size()>1)throw std::invalid_argument("FP64 multi-device qualification is not declared");
    portable_torch::resolve_device(runtime_options);for(auto d:devices)runtime_probe(d,dtype);
    auto synchronize=[&]{for(auto d:devices)portable_torch::synchronize(d);};
    std::vector<std::string> synchronized;for(auto d:devices)synchronized.push_back(d.str());
    std::vector<std::string> effective_nodes;for(auto d:nodes)effective_nodes.push_back(d.str());
    report={{"state","running"},{"schema","tide-weighted-standalone-benchmark-v2"},
      {"kind",r.observe_work?"work-observation":"benchmark"},{"throughput_sample",!r.observe_work},{"throughput_qualified",false},
      {"device",r.device},{"backend",portable_torch::compiled_backend()},{"dtype",r.precision},{"requested_dtype",r.precision},
      {"fixture_dtype","float32"},{"input_policy","same hashed FP32 material; explicit promotion before both schedules"},
      {"runtime_initialization_seconds",tide_bench::seconds(begin)},{"placement",r.placement},{"effective_placement",effective},
      {"fixture_preparation_seconds",fixture_seconds},
      {"scoring_dtype",r.scoring},{"intraop_threads",r.threads},{"interop_threads",r.interop},{"workers",r.workers},
      {"packing",r.packing},{"peer_devices",r.peers},{"node_devices",r.nodes},{"effective_node_devices",effective_nodes},
      {"placement_rule",r.nodes.empty()?"region modulo ordered peers, or single coordinator":"explicit node map"},{"synchronized_devices",synchronized},
      {"communication",{{"requested",r.communication},{"effective",tide::communication_policy()}}},
      {"communication_counter_scope","instrumented explicit tensor copies; tensor endpoint bytes, not PCIe traffic or proof of P2P; resident internal decode/protocol traffic is separate"},
      {"return_packing",r.return_packing},{"return_mode",r.return_mode},{"task",r.task()},
      {"validation_mode",r.validation},{"exploratory_enabled",r.validation=="explore"},{"exploratory_accepted",false},
      {"execution_completed",false},{"execution_finite","not_checked"},
      {"independent_comparison",{{"numeric","not_checked"},{"discrete","not_checked"},{"strict_passed",false}}},
      {"exact_resume",{{"state","not_checked"}}},{"implementation",r.implementation},{"mode",r.mode},{"schedule",r.schedule},
      {"independent_roots",{{"state","not_checked"},{"requested",r.check_roots}}},
      {"warmup_windows",r.warmup},{"measured_windows",r.steps},{"checkpoint_max_bytes",r.checkpoint_bytes},
      {"timing_contract",r.return_mode=="complete"?"complete-diagnostic-v2":"selected-return-v1"},
      {"timing","input clone/transfer, scheduling, packing, requested return, finite guards, task loss when requested, backward/SGD when training, detach and synchronization"},
      {"autograd_enabled",training},{"optimizer_constructed",training},
      {"excluded","fixture/owner construction, warmup, checkpoint I/O, independent reference and prefix replay, final parameter/optimizer export"},
      {"phase","fixture-preparation"},{"metrics",Json::array()},{"warmup_metrics",Json::array()},{"warmup_seconds",0.0},
      {"checkpoint_total_seconds",0.0}};
    if(target.is_cuda())report["cuda_precision"]={{"matmul_allow_tf32",at::globalContext().allowTF32CuBLAS()},
      {"cudnn_allow_tf32",at::globalContext().allowTF32CuDNN()}};
    report["finite_guard"]=r.implementation=="resident"?"public device transaction guards, task/gradient/optimizer checks; returned fields scanned":"all actual result fields, gradients, updated parameter/master/optimizer owners";
    limits(data,r,report);report["resident_limits"]=data.at("limits");
    report["fixture_sha256"]=data.at("fixture_sha256");report["config"]=data.at("config");
    if(data.contains("workload")) {
      const auto& w=data.at("workload");report["model_id"]=w.value("model_id","");report["workload_id"]=w.at("sha256");
      report["parameter_inventory"]=w.at("inventory");report["memory_floor"]=w.at("memory_floor");
      const auto& floor=w.at("memory_floor");report["effective_static_tensor_bytes"]=(floor.at("logical_parameter_bytes").get<int64_t>()+
        floor.at("legacy_inert_buffer_bytes").get<int64_t>())*(dtype==at::kDouble?2.:dtype==at::kHalf?.5:1.);
      if(w.contains("parameter_match"))report["parameter_match"]=w.at("parameter_match");
    }
    write(output/"result.json",report);capture_memory(report,target,"runtime_ready",true,devices);
    auto make_owner=[&]()->std::unique_ptr<Resident> {
      if(r.implementation=="native")return std::make_unique<Eager>(data,training,prefill,r.packing=="scalar",dtype,target,
        placement,r.workers,peers,false,r.return_packing=="packed",nodes,r.return_mode);
#ifdef TIDE_WEIGHTED_RESIDENT
      return resident(data,target,training,prefill,dtype,peers,nodes,r.return_mode);
#else
      throw std::invalid_argument("this standalone build has no resident consumer");
#endif
    };
    report["phase"]="construction";write(output/"result.json",report);begin=tide_bench::Clock::now();
    auto owner=make_owner();Index first=0;
    if(!r.resume.empty())first=restore_run_checkpoint(r.resume,*owner,identity,r.checkpoint_bytes);
    const auto span=data.at("span").get<Index>();
    if(windows+2>std::numeric_limits<Index>::max()/span||first>std::numeric_limits<Index>::max()/span-windows-2)
      throw std::invalid_argument("resumed window exceeds int64 time domain");
    report["start_window"]=first;report["resume_source"]=r.resume.empty()?Json(nullptr):Json(r.resume.string());
    synchronize();report["construction_seconds"]=tide_bench::seconds(begin);capture_memory(report,target,"construction",true,devices);
    std::vector<Step> actual;
    auto execute=[&](Index window,bool warmup) {
      report["phase"]=warmup?"warmup":"measurement";report["window"]=window;write(output/"result.json",report);
      if(r.observe_work&&!warmup)tide::work::reset(true);synchronize();tide::communication_statistics(true);
      begin=tide_bench::Clock::now();auto step=owner->step(window);
      synchronize();const auto seconds=tide_bench::seconds(begin);
      if(window==first+windows-1){report["execution_completed"]=true;report["execution_finite"]="passed";}
      const auto communication=tide::communication_statistics();
      capture_memory(report,target,"window-"+std::to_string(window)+"-compute",true,devices);
      auto event=step.observation.is_null()?weighted_benchmark::work(step.result):step.observation;
      event["step"]=window;event["seconds"]=seconds;event["loss"]=!training&&r.return_mode=="metrics"?Json(nullptr):Json(step.loss);
      event["comm"]=communication;
      if(r.observe_work&&!warmup){event["packing_work"]=packing_work();tide::work::reset(false);}
      report[warmup?"warmup_metrics":"metrics"].push_back(event);
      if(warmup)report["warmup_seconds"]=report["warmup_seconds"].get<double>()+seconds;
      report["phase"]="checkpoint-save";report["checkpoint_write_state"]="running";report["checkpoint_requested_window"]=window+1;
      write(output/"result.json",report);begin=tide_bench::Clock::now();
      save_run_checkpoint(output/"checkpoint",*owner,identity,window+1,r.checkpoint_bytes);
      const auto checkpoint_seconds=tide_bench::seconds(begin);
      report[warmup?"warmup_metrics":"metrics"].back()["checkpoint_seconds"]=checkpoint_seconds;
      report["checkpoint_total_seconds"]=report["checkpoint_total_seconds"].get<double>()+checkpoint_seconds;
      report["checkpoint"]={{"state","saved"},{"path","checkpoint"},{"next_window",window+1},
        {"generation","window-"+std::to_string(window+1)},{"seconds",checkpoint_seconds}};
      report["checkpoint_write_state"]="passed";
      capture_memory(report,target,"window-"+std::to_string(window)+"-checkpoint",true,devices);
      if(validate)actual.push_back(std::move(step));
      write(output/"result.json",report);std::cout<<(warmup?"warmup=":"step=")<<window<<" seconds="<<seconds<<'\n'<<std::flush;
    };
    for(Index i=0;i<r.warmup;++i)execute(first+i,true);
    capture_memory(report,target,"warmup",true,devices);
    for(Index i=0;i<r.steps;++i)execute(first+r.warmup+i,false);
    report["execution_completed"]=true;capture_memory(report,target,"measurement",false,devices);
    const Index next=first+windows;
    const auto parameters=training&&validate?owner->parameters():std::map<std::string,Tensor>{};
    const auto trained=training&&validate?owner->training_state():TrainingState{};
    if(r.checkpoint) {
      report["phase"]="exact-resume";report["window"]=next;write(output/"result.json",report);
      const auto saved=owner->checkpoint();const auto a=owner->step(next);
      const auto x=training?owner->training_state():TrainingState{};
      const auto px=training?owner->parameters():std::map<std::string,Tensor>{};
      owner->close();owner.reset();auto restored=make_owner();restore_run_checkpoint(output/"checkpoint",*restored,identity,r.checkpoint_bytes);
      tide_bench::compare(Result{saved},Result{restored->checkpoint()},true,dtype,{},0.,0.);
      const auto b=restored->step(next);tide_bench::compare(a.result,b.result,true,dtype,{},0.,0.);
      if(a.loss!=b.loss)throw tide_bench::ComparisonFailure({tide_bench::DifferenceKind::numeric,"restored objective","exact objective differs"});
      const auto y=training?restored->training_state():TrainingState{};
      compare_tensors(px,training?restored->parameters():std::map<std::string,Tensor>{},dtype,"restored parameters",true);
      compare_tensors(x.gradients,y.gradients,dtype,"restored gradient",true);compare_tensors(x.momentum,y.momentum,dtype,"restored momentum",true);
      compare_tensors(x.masters,y.masters,dtype,"restored master",true);restored->close();
      report["exact_resume"]={{"state","passed"},{"checkpoint","checkpoint"},{"next_window",next},{"exact",true},
        {"scope",training?"raw decode, complete next-window, parameters, gradients, SGD/master; serial new owner":"raw decode and complete next-window; serial new owner"}};
      capture_memory(report,target,"exact_resume",false,devices);
    }
    if(owner){owner->close();owner.reset();}
    tide_bench::ComparisonReport comparison;
    if(validate) {
      // Replay from original material after measurement. The candidate's
      // checkpoint is never the independent reference's initial state.
      ExecutionPlacement cpu;cpu.scoring_dtype=r.scoring;const bool storage=r.implementation=="resident"&&dtype==at::kHalf;
      Eager reference(data,training,!prefill,true,dtype,at::Device(at::kCPU),cpu,1,{},storage);
      report["reference_precision"]=storage?"fp16-storage-fp32-arithmetic":"same-dtype-eager";
      report["reference_origin"]="original fixture; independent prefix replay";
      for(Index i=0;i<first;++i){report["phase"]="reference-prefix";report["window"]=i;write(output/"result.json",report);reference.step(i);}
      for(Index i=0;i<windows;++i) {
        report["phase"]="independent-check";report["window"]=first+i;write(output/"result.json",report);
        const auto expected=reference.step(first+i);
        if(storage)tide::device_online::weighted_precision_check::validate_storage(actual[i].result,dtype);
        compare_step(comparison,actual[i],expected,dtype,first+i);report["independent_comparison"]=comparison_json(comparison);
      }
      if(training){compare_parameters(comparison,parameters,reference.parameters(),dtype,"final parameter");compare_training(comparison,trained,reference.training_state(),dtype);}
      report["independent_comparison"]=comparison_json(comparison);
    }
    if(r.check_roots){
      report["phase"]="independent-roots";write(output/"result.json",report);begin=tide_bench::Clock::now();
      report["independent_roots"]=check_roots(data,r,target,placement,peers,nodes,comparison);
      synchronize();report["independent_roots_seconds"]=tide_bench::seconds(begin);
      report["independent_comparison"]=comparison_json(comparison);
    }
    const bool strict=validate&&comparison.passed();
    const bool exploratory=r.validation=="explore"&&comparison.numeric>0&&comparison.discrete==0;
    report["exploratory_accepted"]=exploratory;report["state"]=strict?"passed":exploratory?"exploratory":validate?"failed":"completed";
    report["phase"]="complete";report["post_measurement_complete_records"]=strict;
    report["independent_complete_records_checked"]=validate;report["parameter_values_checked"]=training&&validate;
    report["optimizer_values_checked"]=training&&validate;report["final_parameter_gradients_checked"]=training&&validate;
    report["throughput_qualified"]=strict&&r.checkpoint&&!r.observe_work;write(output/"result.json",report);
    return validate&&!strict&&!exploratory?2:0;
  }catch(const std::exception& e) {
    report["state"]="failed";report["error"]=e.what();
    if(report.value("phase","")=="checkpoint-save")report["checkpoint_write_state"]="failed";
    if(const auto* mismatch=dynamic_cast<const tide_bench::ComparisonFailure*>(&e))report["comparison_failure"]={
      {"kind",tide_bench::difference_name(mismatch->difference.kind)},{"path",mismatch->difference.path},{"message",mismatch->difference.message}};
    if(tide::work::packing::enabled()){report["partial_packing_work"]=packing_work();tide::work::reset(false);}
    if(created)if(const auto* failure=dynamic_cast<const tide::weighted::AllocationFailure*>(&e))try {
      auto record=allocation_failure(*failure);record["execution"]={{"phase",report.value("phase","")},{"window",report.value("window",Index(-1))}};
      write(output/"allocation-failure.json",record);report["allocation_failure"]="allocation-failure.json";
    }catch(const std::exception& error){report["allocation_failure_write_error"]=error.what();}
    if(created)try{write(output/"result.json",report);}catch(const std::exception& error){std::cerr<<error.what()<<'\n';}
    std::cerr<<e.what()<<'\n';return 2;
  }
}
