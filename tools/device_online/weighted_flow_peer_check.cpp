// Fixed upstream core-5 complete fibers/regions/cuts, with real split ownership.
#include "weighted_flow_internal.h"
#include "weighted_precision_reference.h"
#include "weighted_flow_fixture.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "../../cpp/bench/streaming.h"
#include "tide/graph_checkpoint.h"
#include "tide/greedy.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
#include <filesystem>
namespace tide::device_online {
struct WeightedFlowPeerTestAccess {
  static void rejected_first_stage(const WeightedFlow& flow){
    const auto& s=*flow.impl_;
    for(const auto& n:s.nodes)
      if(n->state_present.any().cpu().item<bool>()||!at::equal(n->state.floating,n->initial.floating)||!at::equal(n->state.integer,n->initial.integer))
        throw std::runtime_error("a node adopted a rejected common stage");
    for(const auto& r:s.regions)if(r->history.select(1,1).any().cpu().item<bool>())throw std::runtime_error("history adopted a rejected common stage");
    if(!s.failed)throw std::runtime_error("rejected weighted flow remained reusable");
  }
};
}
namespace {
using namespace tide;using namespace tide::device_online;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void compare(const Result& a,const Result& b,at::ScalarType dtype,bool exact=false){
  weighted_precision_check::validate_storage(a,dtype);
  require(a.continuation.identity==b.continuation.identity,"placed graph identity differs");
  tide_bench::compare(a,b,true,dtype,std::nullopt,exact?0.:dtype==at::kHalf?2e-2:-1.,exact?0.:dtype==at::kHalf?1e-3:-1.);
}
std::vector<at::Device> placement(const Graph& g,at::Device coordinator,bool swap=false){
  std::vector<at::Device> out;for(size_t v=0;v<g.nodes.size();++v)out.emplace_back(coordinator.type(),(v+swap)%2);return out;
}
void run(at::Device device,at::ScalarType dtype,Graph g,Model m,const std::string& name,const std::filesystem::path& output,bool prefill){
  at::NoGradGuard ng;auto xs=weighted_check::inputs(g,at::kCPU,dtype);xs.erase(xs.begin()+4);
  for(auto& x:xs)if(x.batch==1){if(x.port==0)x.position=0;if(x.position==0)x.value=at::zeros_like(x.value);}
  auto rm=dtype==at::kHalf?weighted_precision_check::reference_model(m):m;
  Options opts;opts.packed=false;Greedy reference(g,rm,opts);Continuation q;q.identity=g.identity;q.batch_size=2;
  WeightedFlowLimits limits;limits.prefill=prefill;limits.cache=16;limits.stages=64;
  const auto owners=placement(g,device);
  WeightedFlow flow(g,m,q,device,limits,owners);require(flow.node_devices()==owners,"effective node placement differs");
  auto actual=flow.advance(xs,10),expected=reference.run(q,weighted_precision_check::reference_inputs(xs),10,10);compare(actual,expected,dtype);
  require(actual.stats.at("node_peer_devices")==1&&actual.stats.at("node_peer_packet_bytes")>0,"node peer execution absent");
  compare(Result{flow.checkpoint()},Result{actual.continuation},dtype,true);flow.close();
  std::vector<External> head,tail;for(const auto& x:xs)(x.time<4?head:tail).push_back(x);
  WeightedFlow split(g,m,q,device,limits,owners);auto first=split.advance(head,4);
  auto left=reference.run(q,weighted_precision_check::reference_inputs(head),4,4);compare(first,left,dtype);
  const auto path=output/(name+(prefill?"-prefill":"-streaming")+".pt");
  GraphCheckpoint::save(path,g,m,split.checkpoint());split.close();auto restored_model=m;
  auto saved=GraphCheckpoint::load(path,g,restored_model);compare(Result{saved},Result{first.continuation},dtype,true);
  WeightedFlow restored(g,restored_model,saved,device,limits,placement(g,device,true));
  auto last=restored.advance(tail,10);compare(last,reference.run(left.continuation,weighted_precision_check::reference_inputs(tail),10,10),dtype);
  compare(Result{last.continuation},Result{actual.continuation},dtype);
  auto empty=restored.advance({},10);require(empty.trace.empty(),"empty peer window fabricated actions");
  compare(Result{empty.continuation},Result{last.continuation},dtype,true);restored.close();
  std::cout<<"weighted-node-peer case="<<name<<" prefill="<<prefill<<" passed candidates="<<actual.trace.size()<<'\n'<<std::flush;
}
void refusals(at::Device device,at::ScalarType dtype){
  auto g=weighted_check::graph("fc","kv");auto m=weighted_check::model(g,"kv",at::kCPU,dtype);auto xs=weighted_check::inputs(g,at::kCPU,dtype);
  Continuation q;q.identity=g.identity;q.batch_size=2;auto owners=placement(g,device);WeightedFlowLimits limits;limits.cache=8;limits.stages=32;
  at::NoGradGuard ng;
  for(const auto& invalid:std::vector<std::vector<at::Device>>{{device},{at::Device(at::kCPU),device,device,device}}){
    bool refused=false;try{WeightedFlow owner(g,m,q,device,limits,invalid);}catch(const std::invalid_argument&){refused=true;}
    require(refused,"invalid physical node placement accepted");
  }
  limits.queue=2;WeightedFlow blocked(g,m,q,device,limits,owners);bool refused=false;
  try{blocked.advance({xs[0]},10);}catch(const std::exception& e){refused=std::string(e.what()).find("device error=1")!=std::string::npos;}
  require(refused,"cross-owner queue overflow was not refused");WeightedFlowPeerTestAccess::rejected_first_stage(blocked);blocked.close();
  limits.queue=64;WeightedFlow retry(g,m,q,device,limits,owners);auto bad=xs;bad[0].position=9;refused=false;
  try{retry.advance(bad,10);}catch(const std::invalid_argument&){refused=true;}require(refused,"bad boundary was accepted");
  compare(Result{retry.checkpoint()},Result{q},dtype,true);
  auto actual=retry.advance(xs,10);Options opts;opts.packed=false;
  auto rm=dtype==at::kHalf?weighted_precision_check::reference_model(m):m;
  compare(actual,Greedy(g,rm,opts).run(q,weighted_precision_check::reference_inputs(xs),10,10),dtype);retry.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf)||args.output_dir.empty())throw std::invalid_argument("node peers require explicit accelerator, FP32/FP16 and new output directory");
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);
    if(device.index()!=0)throw std::invalid_argument("node peer check requires remapped logical devices 0 and 1");
    at::set_num_threads(1);at::set_num_interop_threads(1);std::filesystem::create_directories(std::filesystem::path(args.output_dir).parent_path());
    if(!std::filesystem::create_directory(args.output_dir))throw std::invalid_argument("node peer output already exists");int count=0;
    for(const std::string family:{"kv","affine","delta","gated_delta","ssd"}){
      auto g=weighted_check::graph("fc",family);auto m=weighted_check::model(g,family,at::kCPU,args.dtype);
      run(device,args.dtype,g,m,family,args.output_dir,true);++count;
    }
    for(const std::string topology:{"hb","nonaligned","feedback"}){
      auto g=weighted_check::graph(topology,"heterogeneous");g.regions[0].read_mode="content";g.regions[1].read_mode="old";g.compile();
      auto m=weighted_check::model(g,"heterogeneous",at::kCPU,args.dtype);
      run(device,args.dtype,g,m,topology,args.output_dir,false);++count;
    }
    auto config=weighted_flow_check::configurations()[0];auto pair=weighted_flow_check::configured(config,true,true,true,args.dtype);
    run(device,args.dtype,pair.first,pair.second,"global-attention",args.output_dir,true);++count;
    refusals(device,args.dtype);
    std::cout<<"weighted-node-peer: passed cases="<<count<<" complete_records=true relocated_checkpoint=true common_refusal=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
