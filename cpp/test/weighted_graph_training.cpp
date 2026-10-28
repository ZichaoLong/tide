// Standalone graph training/continuation; local formulas have separate literal anchors.
#include "weighted_graph_fixture.h"
#include "tide/greedy.h"
#include "tide/graph_checkpoint.h"
#include "tide/placement.h"
#include "../bench/streaming.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
void check_native_graph_records(const std::filesystem::path&,at::Device,at::ScalarType);
namespace {
using namespace tide;
void require(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
void close(const Tensor& a,const Tensor& b,bool half=false,bool exact=false){
  require(a.defined()==b.defined(),"native training gradient connection mismatch");if(!a.defined())return;
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"native training tensor metadata mismatch");
  auto x=a.detach().to(at::kCPU),y=b.detach().to(at::kCPU);
  if(!x.is_floating_point()){require(at::equal(x,y),"native training integer mismatch");return;}
  if(exact){require(at::equal(x,y),"native checkpoint exact tensor mismatch");return;}
  half=half||x.scalar_type()==at::kHalf;
  require(at::allclose(x,y,half?2e-2:x.scalar_type()==at::kDouble?1e-8:1e-5,half?1e-3:x.scalar_type()==at::kDouble?1e-10:1e-6),"native training tensor mismatch");
}
void owners(const ParameterRegistry& a,const ParameterRegistry& b,bool grad,double scale=1.,bool exact=false){
  require(a.alias_partitions()==b.alias_partitions(),"native training aliases differ");
  for(const auto& o:a.owners()){
    auto x=grad?o.value.grad():o.value,y=grad?b.value(o.canonical).grad():b.value(o.canonical);
    if(grad&&scale!=1.){if(x.defined())x=x.to(at::kFloat)/scale;if(y.defined())y=y.to(at::kFloat)/scale;}
    close(x,y,o.value.scalar_type()==at::kHalf,exact);
  }
}
void optimizer(const NamedOptimizer& a,const NamedOptimizer& b,bool exact=false){
  const bool half=a.precision().fp32_masters;
  require(half==b.precision().fp32_masters&&a.precision().loss_scale==b.precision().loss_scale,"native optimizer precision differs");
  require(a.masters().size()==b.masters().size(),"native master inventory differs");
  for(const auto& [name,m]:a.masters())close(m,b.masters().at(name),half,exact);
  require(a.state().size()==b.state().size(),"native optimizer state count differs");
  for(const auto& [name,s]:a.state()){const auto& t=b.state().at(name);require(s.step==t.step,"native optimizer step differs");
    close(s.momentum_buffer,t.momentum_buffer,half,exact);close(s.exp_avg,t.exp_avg,half,exact);close(s.exp_avg_sq,t.exp_avg_sq,half,exact);close(s.max_exp_avg_sq,t.max_exp_avg_sq,half,exact);}
}
Tensor loss(const Result& r){
  auto square=[](const Tensor& x){return (x.scalar_type()==at::kHalf?x.to(at::kFloat):x).square().sum();};
  std::vector<Tensor> terms;for(const auto& o:r.outputs)terms.push_back(square(o.value)*.7);
  for(const auto& [o,s]:r.continuation.states){terms.push_back(square(s.value)*.3);
    for(const auto& [k,t]:s.slots)if(t.is_floating_point())terms.push_back(square(t)*.11);}
  for(const auto& [o,h]:r.continuation.history)for(const auto& [k,t]:h.tensors)terms.push_back(square(t)*.13);
  for(const auto& a:r.continuation.pending)terms.push_back(square(a.value)*.2);
  return at::stack(terms).sum();
}
Continuation detach(Continuation q){
  for(auto& [o,s]:q.states){s.value=s.value.detach();for(auto& [k,t]:s.slots)t=t.detach();}
  for(auto& [o,h]:q.history)for(auto& [k,t]:h.tensors)t=t.detach();
  for(auto& a:q.pending)a.value=a.value.detach();return q;
}
void compare(const Result& a,const Result& b,at::ScalarType dtype,bool exact=false){
  require(a.continuation.identity==b.continuation.identity,"native graph identity mismatch");
  tide_bench::compare(a,b,true,dtype,std::nullopt,exact?0.:dtype==at::kHalf?2e-2:-1.,exact?0.:dtype==at::kHalf?1e-3:-1.);
}
std::unique_ptr<NamedOptimizer> make_optimizer(ParameterRegistry& p,const OptimizerGroup& group,
    const std::string& kind,OptimizerPrecision precision){
  if(kind=="sgd")return std::make_unique<SGD>(p,std::vector<OptimizerGroup>{group},precision);
  return std::make_unique<AdamW>(p,std::vector<OptimizerGroup>{group},precision);
}
void case_check(const std::string& topology,const std::string& family,at::Device device,at::ScalarType dtype,
                const std::filesystem::path& dir,const ExecutionPlacement& placement,
                const std::string& kind,OptimizerPrecision precision){
  auto g=weighted_check::graph(topology,family);
  auto a=weighted_check::model(g,family,at::Device(at::kCPU),dtype),b=weighted_check::model(g,family,device,dtype);
  auto pa=a.parameters(),pb=b.parameters();OptimizerGroup group;group.lr=.001;group.weight_decay=.01;
  for(const auto& o:pa.owners())group.parameters.push_back(o.canonical);
  auto oa_owner=make_optimizer(pa,group,kind,precision),ob_owner=make_optimizer(pb,group,kind,precision);
  auto& oa=*oa_owner;auto& ob=*ob_owner;
  Options options;options.packed=true;options.full_autograd=options.aggregate_autograd="batched";
  auto placed=place_model(g,b,placement);
  ExecutionPlacement cpu;cpu.scoring_dtype=placement.scoring_dtype;
  Streaming scalar(g,place_model(g,a,cpu),Options{});Greedy candidate(g,placed,options);
  const Index stop=10,cut=2;Continuation q;q.identity=g.identity;q.batch_size=2;
  for(Index step=0;step<3;++step){
    oa.zero_grad();ob.zero_grad();auto xa=weighted_check::inputs(g,at::Device(at::kCPU),dtype),xb=weighted_check::inputs(g,device,dtype);
    auto expected=scalar.run(q,xa,stop,stop),actual=candidate.run(q,xb,stop,stop);
    try{compare(expected,actual,dtype);}catch(const std::exception& e){throw std::runtime_error("train forward step="+std::to_string(step)+": "+e.what());}
    oa.backward(loss(expected));ob.backward(loss(actual));owners(pa,pb,true,precision.loss_scale);
    for(size_t i=0;i<xa.size();++i)close(xa[i].value.grad()/precision.loss_scale,xb[i].value.grad()/precision.loss_scale);
    oa.step();ob.step();owners(pa,pb,false);optimizer(oa,ob);
  }
  auto xs=weighted_check::inputs(g,device,dtype);std::vector<External> head,tail;
  for(const auto& x:xs)(x.time<cut?head:tail).push_back(x);
  auto first=candidate.run(q,head,cut,cut);require(!first.continuation.pending.empty(),"native checkpoint case has no pending");
  auto file=dir/(topology+"-"+family+".tidegraph");GraphCheckpoint::save(file,g,b,first.continuation,&ob);
  auto restored=weighted_check::model(g,family,device,dtype);auto pr=restored.parameters();
  auto restored_owner=make_optimizer(pr,group,kind,precision);auto& restored_opt=*restored_owner;
  auto state=GraphCheckpoint::load(file,g,restored,&restored_opt);owners(pb,pr,false,1.,true);optimizer(ob,restored_opt,true);
  Result saved,loaded;saved.continuation=detach(first.continuation);loaded.continuation=state;compare(saved,loaded,dtype,true);
  auto whole=candidate.run(q,xs,stop,stop),retained=candidate.run(first.continuation,tail,stop,stop);
  Result joined=retained;joined.outputs.insert(joined.outputs.begin(),first.outputs.begin(),first.outputs.end());
  auto roots=pb.owners();std::vector<Tensor> params;for(const auto& p:roots)params.push_back(p.value);
  auto ga=torch::autograd::grad({loss(whole)},params,{},false,false,true);
  auto gb=torch::autograd::grad({loss(joined)},params,{},false,false,true);
  for(size_t i=0;i<ga.size();++i)close(ga[i],gb[i]);
  Greedy resume(g,place_model(g,restored,placement),options);ob.zero_grad();restored_opt.zero_grad();
  auto expected=candidate.run(detach(first.continuation),tail,stop,stop),actual=resume.run(state,tail,stop,stop);compare(expected,actual,dtype);
  ob.backward(loss(expected));restored_opt.backward(loss(actual));owners(pb,pr,true,precision.loss_scale);ob.step();restored_opt.step();owners(pb,pr,false);optimizer(ob,restored_opt);
  if(!device.is_cpu()){
    // A portable handoff starts from the actual serialized third-step record,
    // with a second device owner to avoid comparing different optimizer steps.
    auto host=weighted_check::model(g,family,at::Device(at::kCPU),dtype);
    auto peer=weighted_check::model(g,family,device,dtype);
    auto hp=host.parameters(),dp=peer.parameters();
    auto ho=make_optimizer(hp,group,kind,precision),dopt=make_optimizer(dp,group,kind,precision);
    auto hq=GraphCheckpoint::load(file,g,host,ho.get()),dq=GraphCheckpoint::load(file,g,peer,dopt.get());
    Result hs,ds;hs.continuation=hq;ds.continuation=dq;compare(hs,ds,dtype,true);
    owners(hp,dp,false,1.,true);optimizer(*ho,*dopt,true);
    std::vector<External> hx;for(const auto& x:weighted_check::inputs(g,at::Device(at::kCPU),dtype))if(x.time>=cut)hx.push_back(x);
    Greedy hrun(g,place_model(g,host,cpu),options),drun(g,place_model(g,peer,placement),options);
    auto hr=hrun.run(hq,hx,stop,stop),dr=drun.run(dq,tail,stop,stop);compare(hr,dr,dtype);
    ho->backward(loss(hr));dopt->backward(loss(dr));owners(hp,dp,true,precision.loss_scale);
    ho->step();dopt->step();owners(hp,dp,false);optimizer(*ho,*dopt);
  }
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
    compare(clock_saved,clock_read,dtype,true);
  }
}
}
int main(int argc,char** argv){
  portable_torch::RuntimeSession runtime;
  try{
    tide::ExecutionPlacement placement;std::vector<std::string> cases;std::string kind="adamw";double loss_scale=16.;bool explicit_scale=false;
    std::vector<char*> remaining{argv[0]};
    for(int i=1;i<argc;++i){
      std::string arg=argv[i];
      if(arg=="--placement"||arg=="--case"||arg=="--scoring-dtype"||arg=="--optimizer"||arg=="--loss-scale"){
        if(++i>=argc)throw std::invalid_argument(arg+" requires a value");
        if(arg=="--placement")placement.preset=argv[i];
        else if(arg=="--case")cases.push_back(argv[i]);
        else if(arg=="--scoring-dtype")placement.scoring_dtype=argv[i];
        else if(arg=="--optimizer")kind=argv[i];
        else {size_t used=0;std::string value=argv[i];loss_scale=std::stod(value,&used);
          if(used!=value.size())throw std::invalid_argument("invalid loss scale");explicit_scale=true;}
      }else remaining.push_back(argv[i]);
    }
    auto args=portable_torch::parse_cli(remaining.size(),remaining.data(),true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);
      std::cout<<"  --placement cpu|mixed-a|mixed-b|mixed-c  --case TOPOLOGY/FAMILY (repeatable)\n"
        <<"  --scoring-dtype profile|float32|float64  --optimizer adamw|sgd  --loss-scale SCALE\n";return 0;}
    if(args.dtype!=at::kHalf&&args.dtype!=at::kFloat&&args.dtype!=at::kDouble)throw std::invalid_argument("FP16/FP32/FP64 required");
    if(kind!="sgd"&&kind!="adamw")throw std::invalid_argument("optimizer must be adamw or sgd");
    if(args.dtype==at::kHalf&&placement.scoring_dtype=="profile")throw std::invalid_argument("FP16 requires explicit scoring dtype");
    if(args.dtype!=at::kHalf&&explicit_scale&&loss_scale!=1.)throw std::invalid_argument("loss scaling requires FP16");
    if(!std::isfinite(loss_scale)||loss_scale<=0)throw std::invalid_argument("loss scale must be positive and finite");
    OptimizerPrecision precision{args.dtype==at::kHalf,args.dtype==at::kHalf?loss_scale:1.};
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);
    auto resolved=tide::resolve_placement(placement,device);tide::Index checked=0;
    if(args.output_dir.empty()||std::filesystem::exists(args.output_dir))throw std::invalid_argument("new output directory required");
    std::filesystem::create_directories(args.output_dir);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(const std::string topology:{"fc","hb","nonaligned","feedback"})
      for(const std::string family:{"kv","affine","delta","gated_delta","ssd","heterogeneous"}){
        const auto key=topology+"/"+family;
        if(!cases.empty()&&std::find(cases.begin(),cases.end(),key)==cases.end())continue;
        try{case_check(topology,family,device,args.dtype,args.output_dir,placement,kind,precision);++checked;}
        catch(const std::exception& e){throw std::runtime_error(topology+"/"+family+": "+e.what());}
      }
    if(!checked)throw std::invalid_argument("no matching weighted graph cases");
    check_native_graph_records(args.output_dir,device,args.dtype);
    portable_torch::synchronize(device);
    std::cout<<"{\"state\":\"passed\",\"cases\":"<<checked<<",\"updates_per_case\":4,\"complete_native_checkpoint\":true,"
      <<"\"cpu_reference\":true,\"placement\":\""<<placement.preset<<"\",\"payload_dtype\":\""
      <<(args.dtype==at::kHalf?"float16":args.dtype==at::kFloat?"float32":"float64")
      <<"\",\"scoring_dtype\":\""<<placement.scoring_dtype<<"\",\"optimizer\":\""<<kind
      <<"\",\"loss_scale\":"<<precision.loss_scale<<",\"fp32_masters\":"<<(precision.fp32_masters?"true":"false")<<"}\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
