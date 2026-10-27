// Independent scalar schedules, private roots and NamedOptimizer algebra.
#include "weighted_training_check.h"
#include "weighted_precision_reference.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "../../cpp/bench/streaming.h"
#include "tide/greedy.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
namespace {
using namespace tide;using namespace tide::device_online;using namespace weighted_training_check;
void leaf(at::Tensor& x){x=x.detach().clone().set_requires_grad(true);}
void independent(Continuation& q){for(auto& a:q.pending)leaf(a.value);
  for(auto& [_,s]:q.states){leaf(s.value);for(auto& [name,x]:s.slots)if(x.is_floating_point())leaf(x);}}
void same_slots(const ResidentTrainingCheckpoint& a,const ResidentTrainingCheckpoint& b){
  require(a.generation==b.generation&&a.aliases==b.aliases,"replica generation/aliases differ");
  for(const auto& [name,x]:a.parameters)require(at::equal(x,b.parameters.at(name)),"replica named parameters differ");
  for(const auto& pair:std::vector<std::pair<Tensor,Tensor>>{{a.state.values,b.state.values},{a.state.first,b.state.first},
      {a.state.second,b.state.second},{a.state.maximum,b.state.maximum},{a.state.steps,b.state.steps},{a.state.corrections,b.state.corrections}})
    require(at::equal(pair.first,pair.second),"replica optimizer slots differ");
}
void run(at::Device device,at::ScalarType dtype,const std::string& topology,const std::string& family,ResidentOptimizerKind kind){
  const at::Device other(device.type(),1);auto graph=weighted_check::graph(topology,family);
  auto candidate=weighted_check::model(graph,family,at::kCPU,dtype);
  auto model=dtype==at::kHalf?weighted_precision_check::reference_model(candidate):weighted_check::model(graph,family,at::kCPU,at::kFloat);
  auto masters=weighted_precision_check::reference_model(model);auto registry=model.parameters(),master_registry=masters.parameters();
  OptimizerGroup group;for(const auto& p:registry.owners())group.parameters.push_back(p.canonical);
  group.lr=.001;group.weight_decay=.01;group.momentum=kind==ResidentOptimizerKind::sgd?.3:0.;group.amsgrad=kind==ResidentOptimizerKind::adamw;
  std::unique_ptr<NamedOptimizer> optimizer;
  if(kind==ResidentOptimizerKind::sgd)optimizer=std::make_unique<SGD>(master_registry,std::vector<OptimizerGroup>{group});
  else optimizer=std::make_unique<AdamW>(master_registry,std::vector<OptimizerGroup>{group});
  at::NoGradGuard ng;auto l=limits();l.forward.stages=32;
  std::vector<Continuation> qs(2);std::vector<std::unique_ptr<WeightedTrainingSession>> owners;
  for(auto d:{device,other}){Continuation q;q.identity=graph.identity;q.batch_size=1;qs[owners.size()]=q;
    owners.push_back(std::make_unique<WeightedTrainingSession>(graph,candidate,q,d,kind,std::vector<OptimizerGroup>{group},l));}
  for(int update=0;update<3;++update){
    const auto input=weighted_check::inputs(graph,at::kCPU,dtype,10*update);std::vector<at::Tensor> total(registry.owners().size());
    for(size_t replica=0;replica<2;++replica){
      independent(qs[replica]);const auto initial=qs[replica];std::vector<External> xs;
      for(auto x:input)if(x.batch==int64_t(replica)){x.batch=0;x.position+=2*update;xs.push_back(x);}
      auto ref_xs=weighted_precision_check::reference_inputs(xs);for(auto& x:ref_xs)leaf(x.value);
      auto w=owners[replica]->advance(xs,10*(update+1));auto actual=owners[replica]->result();weighted_precision_check::validate_storage(actual,dtype);
      auto roots=seed(w,update==0?"outputs":update==1?"all":"zero");
      Result reference;std::vector<at::Tensor> expected,leaves;
      {at::AutoGradMode grad(true);Options options;options.packed=false;
        reference=Greedy(graph,model,options).run(qs[replica],ref_xs,10*(update+1),10*(update+1));qs[replica]=reference.continuation;
        auto objective_roots=objective(w,roots,reference,model);
        for(const auto& p:registry.owners())leaves.push_back(p.value);for(const auto& x:ref_xs)leaves.push_back(x.value);
        for(const auto& a:initial.pending)leaves.push_back(a.value);
        for(const auto& [_,s]:initial.states){leaves.push_back(s.value);for(const auto& [name,x]:s.slots)if(x.is_floating_point())leaves.push_back(x);}
        expected=torch::autograd::grad(objective_roots.first,leaves,objective_roots.second,false,false,true);}
      tide_bench::compare(actual,reference,true,dtype,std::nullopt,dtype==at::kHalf?2e-2:-1.,dtype==at::kHalf?1e-3:-1.);
      auto gradient=owners[replica]->backward({roots});check_gradients(graph,registry,initial,{ref_xs},expected,gradient);
      for(size_t j=0;j<total.size();++j)if(expected[j].defined())total[j]=total[j].defined()?total[j]+expected[j]:expected[j].clone();
      if(update==1)owners[replica]->accumulate(); // Collective also accepts detached accumulated groups.
    }
    std::vector<WeightedTrainingSession*> group_owners{owners[0].get(),owners[1].get()};
    if(!update){bool refused=false;try{WeightedTrainingSession::step_replicas(group_owners,1);}catch(const std::invalid_argument&){refused=true;}
      require(refused,"weighted peer budget not refused before allocation");}
    auto result=WeightedTrainingSession::step_replicas(group_owners);
    require(result.step.applied&&result.parameters.size()==2&&result.transferred_bytes>0,"weighted peer step did not transfer/commit");
    for(const auto& g:result.parameters)for(size_t j=0;j<total.size();++j){const auto& p=registry.owners()[j];
      gradient(g.values.narrow(0,g.offsets[j],p.value.numel()).reshape(p.value.sizes()),g.connected[j].item<bool>(),total[j],"peer canonical sum "+p.canonical);}
    optimizer->zero_grad();for(size_t j=0;j<total.size();++j)master_registry.value(registry.owners()[j].canonical).mutable_grad()=total[j];optimizer->step();
    for(const auto& p:registry.owners())p.value.copy_(master_registry.value(p.canonical).to(dtype).to(at::kFloat));
    auto a=owners[0]->checkpoint(),b=owners[1]->checkpoint();same_slots(a,b);
    for(size_t j=0;j<registry.owners().size();++j){const auto& p=registry.owners()[j];
      require(at::allclose(a.parameters.at(p.canonical).to(at::kFloat),p.value,1e-5,1e-6),"peer parameter update differs from independent optimizer");
      require(at::allclose(a.state.values.narrow(0,a.offsets[j],p.value.numel()).reshape(p.value.sizes()),master_registry.value(p.canonical),3e-5,3e-6),"peer FP32 master differs");}
    if(!update)for(size_t j=0;j<2;++j){auto checkpoint=owners[j]->checkpoint();owners[j]->close();
      owners[j]=std::make_unique<WeightedTrainingSession>(graph,candidate,checkpoint,j?other:device,l);same_slots(checkpoint,owners[j]->checkpoint());}
    std::cout<<"weighted-peer update="<<update<<" topology="<<topology<<" bytes="<<result.transferred_bytes<<" passed\n"<<std::flush;
  }
  // A same-generation slot mismatch must reject every commit, including a None-only step.
  auto a=owners[0]->checkpoint(),b=owners[1]->checkpoint();for(auto& owner:owners)owner->close();
  b.state.first=b.state.first+.001;
  owners[0]=std::make_unique<WeightedTrainingSession>(graph,candidate,a,device,l);
  owners[1]=std::make_unique<WeightedTrainingSession>(graph,candidate,b,other,l);
  for(auto& owner:owners){auto w=owner->advance({},owner->cut());owner->backward({seed(w,"none")});}
  auto refused=WeightedTrainingSession::step_replicas({owners[0].get(),owners[1].get()});
  require(!refused.step.applied&&refused.step.refusal_code==25,"weighted replica slot mismatch was committed");
  for(size_t j=0;j<2;++j){owners[j]->detach();same_slots(j?b:a,owners[j]->checkpoint());owners[j]->close();}
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("weighted peers require explicit FP32/FP16 accelerator");
    args.allow_npu_float16=true;auto d=portable_torch::resolve_device(args);
    if(d.index()!=0)throw std::invalid_argument("weighted peer check requires remapped logical devices 0 and 1");
    at::set_num_threads(1);at::set_num_interop_threads(1);
    run(d,args.dtype,"nonaligned","kv",ResidentOptimizerKind::sgd);run(d,args.dtype,"feedback","heterogeneous",ResidentOptimizerKind::adamw);
    std::cout<<"weighted-peer: passed configurations=2 updates=6 replicas=2 complete_roots=true both_optimizers=true resume=true consensus_refusal=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
