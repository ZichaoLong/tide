// Standalone graph training/continuation; local formulas have separate literal anchors.
#include "weighted_graph_fixture.h"
#include "tide/greedy.h"
#include "tide/graph_checkpoint.h"
#include "tide/placement.h"
#include "../bench/streaming.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <fstream>
#include <iostream>
void check_native_graph_records(const std::filesystem::path&,at::Device,at::ScalarType);
namespace {
using namespace tide;
void require(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
void close(const Tensor& a,const Tensor& b){
  require(a.defined()==b.defined(),"native training gradient connection mismatch");if(!a.defined())return;
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"native training tensor metadata mismatch");
  auto x=a.detach().to(at::kCPU),y=b.detach().to(at::kCPU);
  if(!x.is_floating_point()){require(at::equal(x,y),"native training integer mismatch");return;}
  require(at::allclose(x,y,x.scalar_type()==at::kDouble?1e-8:1e-5,x.scalar_type()==at::kDouble?1e-10:1e-6),"native training tensor mismatch");
}
void owners(const ParameterRegistry& a,const ParameterRegistry& b,bool grad){
  require(a.alias_partitions()==b.alias_partitions(),"native training aliases differ");
  for(const auto& o:a.owners())close(grad?o.value.grad():o.value,grad?b.value(o.canonical).grad():b.value(o.canonical));
}
void optimizer(const NamedOptimizer& a,const NamedOptimizer& b){
  require(a.state().size()==b.state().size(),"native optimizer state count differs");
  for(const auto& [name,s]:a.state()){const auto& t=b.state().at(name);require(s.step==t.step,"native optimizer step differs");
    close(s.momentum_buffer,t.momentum_buffer);close(s.exp_avg,t.exp_avg);close(s.exp_avg_sq,t.exp_avg_sq);close(s.max_exp_avg_sq,t.max_exp_avg_sq);}
}
Tensor loss(const Result& r){
  std::vector<Tensor> terms;for(const auto& o:r.outputs)terms.push_back(o.value.square().sum()*.7);
  for(const auto& [o,s]:r.continuation.states){terms.push_back(s.value.square().sum()*.3);
    for(const auto& [k,t]:s.slots)if(t.is_floating_point())terms.push_back(t.square().sum()*.11);}
  for(const auto& [o,h]:r.continuation.history)for(const auto& [k,t]:h.tensors)terms.push_back(t.square().sum()*.13);
  for(const auto& a:r.continuation.pending)terms.push_back(a.value.square().sum()*.2);
  return at::stack(terms).sum();
}
Continuation detach(Continuation q){
  for(auto& [o,s]:q.states){s.value=s.value.detach();for(auto& [k,t]:s.slots)t=t.detach();}
  for(auto& [o,h]:q.history)for(auto& [k,t]:h.tensors)t=t.detach();
  for(auto& a:q.pending)a.value=a.value.detach();return q;
}
void compare(const Result& a,const Result& b,at::ScalarType dtype){
  require(a.continuation.identity==b.continuation.identity,"native graph identity mismatch");
  tide_bench::compare(a,b,true,dtype);
}
void case_check(const std::string& topology,const std::string& family,at::Device device,at::ScalarType dtype,
                const std::filesystem::path& dir,const ExecutionPlacement& placement){
  auto g=weighted_check::graph(topology,family);
  auto a=weighted_check::model(g,family,at::Device(at::kCPU),dtype),b=weighted_check::model(g,family,device,dtype);
  auto pa=a.parameters(),pb=b.parameters();OptimizerGroup group;group.lr=.001;group.weight_decay=.01;
  for(const auto& o:pa.owners())group.parameters.push_back(o.canonical);
  AdamW oa(pa,{group}),ob(pb,{group});Options options;options.packed=true;options.full_autograd=options.aggregate_autograd="batched";
  auto placed=place_model(g,b,placement);
  Streaming scalar(g,a,Options{});Greedy candidate(g,placed,options);
  const Index stop=10,cut=2;Continuation q;q.identity=g.identity;q.batch_size=2;
  for(Index step=0;step<3;++step){
    oa.zero_grad();ob.zero_grad();auto xa=weighted_check::inputs(g,at::Device(at::kCPU),dtype),xb=weighted_check::inputs(g,device,dtype);
    auto expected=scalar.run(q,xa,stop,stop),actual=candidate.run(q,xb,stop,stop);compare(expected,actual,dtype);
    loss(expected).backward();loss(actual).backward();owners(pa,pb,true);
    for(size_t i=0;i<xa.size();++i)close(xa[i].value.grad(),xb[i].value.grad());
    oa.step();ob.step();owners(pa,pb,false);optimizer(oa,ob);
  }
  auto xs=weighted_check::inputs(g,device,dtype);std::vector<External> head,tail;
  for(const auto& x:xs)(x.time<cut?head:tail).push_back(x);
  auto first=candidate.run(q,head,cut,cut);require(!first.continuation.pending.empty(),"native checkpoint case has no pending");
  auto file=dir/(topology+"-"+family+".tidegraph");GraphCheckpoint::save(file,g,b,first.continuation,&ob);
  auto restored=weighted_check::model(g,family,device,dtype);auto pr=restored.parameters();AdamW restored_opt(pr,{group});
  auto state=GraphCheckpoint::load(file,g,restored,&restored_opt);owners(pb,pr,false);optimizer(ob,restored_opt);
  Result saved,loaded;saved.continuation=detach(first.continuation);loaded.continuation=state;compare(saved,loaded,dtype);
  auto whole=candidate.run(q,xs,stop,stop),retained=candidate.run(first.continuation,tail,stop,stop);
  Result joined=retained;joined.outputs.insert(joined.outputs.begin(),first.outputs.begin(),first.outputs.end());
  auto roots=pb.owners();std::vector<Tensor> params;for(const auto& p:roots)params.push_back(p.value);
  auto ga=torch::autograd::grad({loss(whole)},params,{},false,false,true);
  auto gb=torch::autograd::grad({loss(joined)},params,{},false,false,true);
  for(size_t i=0;i<ga.size();++i)close(ga[i],gb[i]);
  Greedy resume(g,place_model(g,restored,placement),options);ob.zero_grad();restored_opt.zero_grad();
  auto expected=candidate.run(detach(first.continuation),tail,stop,stop),actual=resume.run(state,tail,stop,stop);compare(expected,actual,dtype);
  loss(expected).backward();loss(actual).backward();owners(pb,pr,true);ob.step();restored_opt.step();owners(pb,pr,false);optimizer(ob,restored_opt);
  // Validation precedes publication/mutation, and publication never overwrites.
  bool rejected=false;try{GraphCheckpoint::save(file,g,b,first.continuation,&ob);}catch(const std::exception&){rejected=true;}
  require(rejected,"native graph checkpoint overwrote an existing file");
  auto bad=first.continuation;bad.pending[0].time=0;rejected=false;
  try{GraphCheckpoint::save(dir/(topology+"-"+family+"-bad"),g,b,bad,&ob);}catch(const std::exception&){rejected=true;}
  require(rejected,"native graph checkpoint accepted invalid pending");
  if(topology=="fc"&&family=="kv"){
    auto target=weighted_check::model(g,family,device,dtype);target.nodes[1].extra["kernel.base"]=target.nodes[1].extra.at("kernel.base").detach().clone().set_requires_grad(true);
    auto before=target.nodes[0].extra.at("kernel.base").clone();rejected=false;
    try{GraphCheckpoint::load(file,g,target);}catch(const std::exception&){rejected=true;}
    require(rejected,"native graph checkpoint accepted changed alias topology");close(before,target.nodes[0].extra.at("kernel.base"));
    std::ifstream in(file,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(in)),{});bytes[17]^=1;
    auto corrupt=dir/"corrupt.tidegraph";{std::ofstream out(corrupt,std::ios::binary);out.write(bytes.data(),bytes.size());}
    rejected=false;try{GraphCheckpoint::load(corrupt,g,b);}catch(const std::exception&){rejected=true;}
    require(rejected,"native graph checkpoint accepted checksum corruption");
    auto large=q;large.cut=(Index{1}<<54)+17;auto lx=weighted_check::inputs(g,device,dtype,large.cut);
    auto one=candidate.run(large,{lx[0],lx[1],lx[4],lx[5]},large.cut+2,large.cut+2);
    auto clock_file=dir/"int64-clock.tidegraph";GraphCheckpoint::save(clock_file,g,b,one.continuation);
    Result clock_read;clock_read.continuation=GraphCheckpoint::load(clock_file,g,b);Result clock_saved;clock_saved.continuation=one.continuation;
    compare(clock_saved,clock_read,dtype);
  }
}
}
int main(int argc,char** argv){
  portable_torch::RuntimeSession runtime;
  try{
    tide::ExecutionPlacement placement;std::vector<std::string> cases;
    std::vector<char*> remaining{argv[0]};
    for(int i=1;i<argc;++i){
      std::string arg=argv[i];
      if(arg=="--placement"||arg=="--case"){
        if(++i>=argc)throw std::invalid_argument(arg+" requires a value");
        if(arg=="--placement")placement.preset=argv[i];else cases.push_back(argv[i]);
      }else remaining.push_back(argv[i]);
    }
    auto args=portable_torch::parse_cli(remaining.size(),remaining.data(),true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);
      std::cout<<"  --placement cpu|mixed-a|mixed-b|mixed-c  --case TOPOLOGY/FAMILY (repeatable)\n";return 0;}
    auto device=portable_torch::resolve_device(args);
    auto resolved=tide::resolve_placement(placement,device);tide::Index checked=0;
    if(args.dtype!=at::kFloat&&args.dtype!=at::kDouble)throw std::invalid_argument("FP32/FP64 required");
    if(args.output_dir.empty()||std::filesystem::exists(args.output_dir))throw std::invalid_argument("new output directory required");
    std::filesystem::create_directories(args.output_dir);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(const std::string topology:{"fc","hb","nonaligned","feedback"})
      for(const std::string family:{"kv","affine","delta","gated_delta","ssd","heterogeneous"}){
        const auto key=topology+"/"+family;
        if(!cases.empty()&&std::find(cases.begin(),cases.end(),key)==cases.end())continue;
        try{case_check(topology,family,device,args.dtype,args.output_dir,placement);++checked;}
        catch(const std::exception& e){throw std::runtime_error(topology+"/"+family+": "+e.what());}
      }
    if(!checked)throw std::invalid_argument("no matching weighted graph cases");
    check_native_graph_records(args.output_dir,device,args.dtype);
    portable_torch::synchronize(device);
    std::cout<<"{\"state\":\"passed\",\"cases\":"<<checked<<",\"updates_per_case\":4,\"complete_native_checkpoint\":true,"
      <<"\"cpu_reference\":true,\"placement\":\""<<placement.preset<<"\"}\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
