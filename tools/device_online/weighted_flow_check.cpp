// Complete graph schedule/record anchor. Local math is independently qualified
// by the weighted_*_formula/check sources; this reference has its own scheduler.
#include "weighted_flow.h"
#include "weighted_precision_reference.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "../../cpp/bench/streaming.h"
#include "tide/greedy.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
#include <filesystem>
namespace tide::device_online {void check_weighted_flow_boundaries(at::Device,at::ScalarType,const std::filesystem::path&);}
namespace {
using namespace tide;using namespace tide::device_online;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void compare(const Result& actual,const Result& expected,at::ScalarType dtype){
  weighted_precision_check::validate_storage(actual,dtype);
  require(actual.continuation.identity==expected.continuation.identity,"weighted graph identity differs");
  tide_bench::compare(actual,expected,true,dtype,std::nullopt,dtype==at::kHalf?2e-2:-1.,dtype==at::kHalf?1e-3:-1.);
}
void run(at::Device device,at::ScalarType dtype,const std::string& family,const std::string& topology,const std::string& read,bool prefill){
  auto g=weighted_check::graph(topology,family);for(auto& r:g.regions)r.read_mode=read;g.compile();
  auto m=weighted_check::model(g,family,at::kCPU,dtype);auto inputs=weighted_check::inputs(g,at::kCPU,dtype);
  // The second sample deliberately misses one port and includes zero reception.
  inputs.erase(inputs.begin()+4);for(auto& x:inputs)if(x.batch==1&&x.position==0)x.value=at::zeros_like(x.value);
  // Repair per-port positions after removing its first occurrence.
  for(auto& x:inputs)if(x.batch==1&&x.port==0)x.position=0;
  Continuation q;q.identity=g.identity;q.batch_size=2;
  auto ref_model=dtype==at::kHalf?weighted_precision_check::reference_model(m):m;
  auto ref_inputs=weighted_precision_check::reference_inputs(inputs);
  Options options;options.packed=false;Greedy reference(g,ref_model,options);WeightedFlowLimits limits;limits.prefill=prefill;limits.cache=16;limits.stages=64;
  at::NoGradGuard no_grad;
  WeightedFlow flow(g,m,q,device,limits);auto actual=flow.advance(inputs,10);auto expected=reference.run(q,ref_inputs,10,10);compare(actual,expected,dtype);
  for(const auto& out:actual.outputs)require(out.value.scalar_type()==dtype,"weighted output storage dtype differs");
  for(const auto& atom:actual.messages)require(atom.value.scalar_type()==dtype,"weighted message storage dtype differs");
  for(const auto& [key,state]:actual.continuation.states){require(state.value.scalar_type()==dtype,"weighted raw state dtype differs");
    for(const auto& [name,value]:state.slots)require(value.scalar_type()==(value.is_floating_point()?dtype:at::kLong),"weighted raw field storage dtype differs");}
  require(actual.stats.at("candidate_updates")==int64_t(actual.trace.size()),"weighted graph actual work count differs");
  require(actual.stats.at("resident_stages")>0,"weighted graph did not use resident progress");
  auto checkpoint=flow.checkpoint();compare(Result{checkpoint},Result{actual.continuation},dtype);flow.close();
  WeightedFlow split(g,m,q,device,limits);std::vector<External> head,tail;
  for(const auto& x:inputs)(x.time<4?head:tail).push_back(x);
  auto first=split.advance(head,4);auto first_ref=reference.run(q,weighted_precision_check::reference_inputs(head),4,4);compare(first,first_ref,dtype);
  auto saved=split.checkpoint();split.close();WeightedFlow restored(g,m,saved,device,limits);
  auto continued=restored.advance(tail,10);auto tail_ref=reference.run(first_ref.continuation,weighted_precision_check::reference_inputs(tail),10,10);compare(continued,tail_ref,dtype);
  compare(Result{continued.continuation},Result{actual.continuation},dtype);
  // Empty later windows retain expired raw tuples, and are independent of the
  // bounded stage journal used for the preceding call.
  auto empty=restored.advance({},10);require(empty.trace.empty(),"empty weighted window fabricated events");
  compare(Result{empty.continuation},Result{continued.continuation},dtype);restored.close();
  std::cout<<"weighted-flow case "<<family<<' '<<topology<<' '<<read<<' '<<(prefill?"prefill":"streaming")<<" passed stages="
    <<actual.stats.at("resident_stages")<<" candidates="<<actual.trace.size()<<" full="<<actual.stats.at("full_evaluations")<<'\n'<<std::flush;
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("weighted flow requires explicit accelerator FP32/FP16");
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(const std::string family:{"kv","affine","delta","gated_delta","ssd","heterogeneous"})
      for(bool prefill:{false,true}){run(device,args.dtype,family,"fc","proposal",prefill);++cases;}
    for(const std::string topology:{"hb","nonaligned","feedback"})for(const std::string read:{"content","old","proposal"}){
      run(device,args.dtype,"heterogeneous",topology,read,true);++cases;}
    check_weighted_flow_boundaries(device,args.dtype,args.output_dir.empty()?std::filesystem::path("artifacts/weighted-flow-check"):std::filesystem::path(args.output_dir));
    std::cout<<"weighted-flow: passed cases="<<cases<<" complete_records=true exact_raw_checkpoint=true online_device_progress=true capacity_refusal=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
