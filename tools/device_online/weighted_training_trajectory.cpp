#include "weighted_training_check.h"
#include "weighted_precision_reference.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "../../cpp/bench/streaming.h"
#include "tide/greedy.h"
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
namespace tide::device_online::weighted_training_check {
WeightedTrainingLimits limits(){
  WeightedTrainingLimits l;l.forward.cache=16;l.forward.stages=32;l.forward.queue=64;
  l.forward.tensor_bytes=768*1024*1024;l.forward.operator_bytes=128*1024*1024;l.forward.timeout_ms=60000;
  l.total_bytes=2304LL*1024*1024;return l;
}
namespace {
void freeze(Model& model,const std::string& mode){size_t j=0;for(const auto& p:model.parameters(false).owners()){
  if(mode=="all"||(mode=="partial"&&j++%3==0))p.value.set_requires_grad(false);}}
void leaf(at::Tensor& x){x=x.detach().clone().set_requires_grad(true);}
void independent_left(Continuation& q){for(auto& a:q.pending)leaf(a.value);
  for(auto& [key,s]:q.states){leaf(s.value);for(auto& [name,x]:s.slots)if(x.is_floating_point())leaf(x);}}
void equal_checkpoint(const ResidentTrainingCheckpoint& a,const ResidentTrainingCheckpoint& b){
  require(a.aliases==b.aliases&&a.trainable==b.trainable&&a.offsets==b.offsets&&a.generation==b.generation&&a.next_token==b.next_token,"restored metadata differs");
  for(const auto& [name,x]:a.parameters)require(at::equal(x,b.parameters.at(name)),"restored parameter differs: "+name);
  for(const auto& pair:std::vector<std::pair<at::Tensor,at::Tensor>>{{a.state.values,b.state.values},{a.state.first,b.state.first},{a.state.second,b.state.second},
      {a.state.maximum,b.state.maximum},{a.state.steps,b.state.steps},{a.state.corrections,b.state.corrections}})require(at::equal(pair.first,pair.second),"restored optimizer slot differs");
  tide_bench::compare(Result{a.continuation},Result{b.continuation},true,at::kFloat);
}
}
void trajectory(at::Device device,const std::string& topology,const std::string& family,ResidentOptimizerKind kind,const std::string& frozen,at::ScalarType dtype){
  auto graph=weighted_check::graph(topology,family);
  graph.outputs.push_back(graph.outputs.front());
  graph.layout.reset();graph.compile(); // New physical port needs a new default slot layout.
  auto candidate=weighted_check::model(graph,family,at::kCPU,dtype);freeze(candidate,frozen);
  auto model=dtype==at::kHalf?weighted_precision_check::reference_model(candidate):weighted_check::model(graph,family,at::kCPU,at::kFloat);freeze(model,frozen);
  auto masters=dtype==at::kHalf?weighted_precision_check::reference_model(model):model;
  auto master_registry=masters.parameters();
  const auto master_owners=master_registry.owners(); // Own returned metadata while borrowing Tensor fields.
  auto registry=model.parameters();Continuation q;q.identity=graph.identity;q.batch_size=2;
  OptimizerGroup group;for(const auto& owner:registry.owners())group.parameters.push_back(owner.canonical);
  group.lr=.001;group.weight_decay=.01;if(kind==ResidentOptimizerKind::sgd)group.momentum=.3;else group.amsgrad=true;
  std::unique_ptr<NamedOptimizer> optimizer;
  if(kind==ResidentOptimizerKind::sgd)optimizer=std::make_unique<SGD>(master_registry,std::vector<OptimizerGroup>{group});
  else optimizer=std::make_unique<AdamW>(master_registry,std::vector<OptimizerGroup>{group});
  at::NoGradGuard ng;auto l=limits();l.forward.prefill=topology!="nonaligned";
  auto session=std::make_unique<WeightedTrainingSession>(graph,candidate,q,device,kind,std::vector<OptimizerGroup>{group},l);
  const auto original=session->checkpoint().state.values;
  const std::vector<std::string> modes{"outputs","zero","all","outputs"};
  for(size_t update=0;update<modes.size();++update){
    std::cout<<"weighted-training phase "<<topology<<'/'<<family<<'/'<<frozen<<" update="<<update<<' '<<modes[update]<<'\n'<<std::flush;
    independent_left(q);auto initial=q;
    std::unique_ptr<WeightedTrainingSession> restored;
    if(update==1){std::cout<<"  restore checkpoint\n"<<std::flush;
      const auto checkpoint=session->checkpoint();restored=std::make_unique<WeightedTrainingSession>(graph,candidate,checkpoint,device,l);equal_checkpoint(checkpoint,restored->checkpoint());}
    auto inputs=weighted_check::inputs(graph,at::kCPU,dtype,10*update);
    std::vector<std::vector<External>> windows(3);for(auto x:inputs){leaf(x.value);x.position+=2*update;windows[x.time<10*update+4?0:2].push_back(x);}
    auto reference_windows=windows;
    if(dtype==at::kHalf)for(auto& window:reference_windows)for(auto& x:window)x.value=x.value.detach().to(at::kFloat).set_requires_grad(true);
    std::vector<WeightedCotangents> roots,other_roots;std::vector<at::Tensor> outputs,bars,leaves;
    for(size_t i=0;i<windows.size();++i){const auto stop=10*update+(i<2?4:10);
      std::cout<<"  window="<<i<<" stop="<<stop<<" forward\n"<<std::flush;
      auto actual=session->advance(windows[i],stop);auto direct=session->result();
      weighted_precision_check::validate_storage(direct,dtype);
      Result reference;{at::AutoGradMode grad(true);Options options;options.packed=false;reference=Greedy(graph,model,options).run(q,reference_windows[i],stop,stop);q=reference.continuation;}
      tide_bench::compare(direct,reference,true,dtype,std::nullopt,dtype==at::kHalf?2e-2:-1.,dtype==at::kHalf?1e-3:-1.);roots.push_back(seed(actual,modes[update]));
      {at::AutoGradMode grad(true);auto objective_roots=objective(actual,roots.back(),reference,model);
        outputs.insert(outputs.end(),objective_roots.first.begin(),objective_roots.first.end());bars.insert(bars.end(),objective_roots.second.begin(),objective_roots.second.end());}
      if(restored){std::cout<<"  window="<<i<<" restored forward\n"<<std::flush;
        auto other=restored->advance(windows[i],stop);other_roots.push_back(seed(other,modes[update]));
        equal_window(actual,other);tide_bench::compare(restored->result(),direct,true,at::kFloat);}
    }
    for(const auto& owner:registry.owners())leaves.push_back(owner.value);
    for(const auto& xs:reference_windows)for(const auto& x:xs)leaves.push_back(x.value);
    for(const auto& a:initial.pending)leaves.push_back(a.value);
    for(const auto& [key,s]:initial.states){leaves.push_back(s.value);for(const auto& [name,x]:s.slots)if(x.is_floating_point())leaves.push_back(x);}
    std::vector<at::Tensor> expected(leaves.size());if(!outputs.empty()){at::AutoGradMode grad(true);expected=torch::autograd::grad(outputs,leaves,bars,false,false,true);}
    std::cout<<"  backward\n"<<std::flush;
    auto actual=session->backward(roots);check_gradients(graph,registry,initial,reference_windows,expected,actual);
    if(restored){std::cout<<"  restored backward\n"<<std::flush;
      auto other=restored->backward(other_roots);require(at::equal(actual.parameters.values,other.parameters.values)&&at::equal(actual.parameters.connected,other.parameters.connected),"restored parameter adjoints differ");}
    std::cout<<"  optimizer\n"<<std::flush;
    optimizer->zero_grad();for(size_t j=0;j<registry.owners().size();++j)master_registry.value(registry.owners()[j].canonical).mutable_grad()=expected[j];
    optimizer->step();
    if(dtype==at::kHalf)for(const auto& owner:registry.owners())owner.value.copy_(master_registry.value(owner.canonical).to(at::kHalf).to(at::kFloat));
    require(session->step().applied,"weighted optimizer rejected finite update");
    if(restored){require(restored->step().applied,"restored weighted step refused");equal_checkpoint(session->checkpoint(),restored->checkpoint());restored->close();}
    const auto checkpoint=session->checkpoint();
    if(update==0&&frozen!="all")require(!at::equal(original,checkpoint.state.values)&&checkpoint.state.steps.gt(0).any().item<bool>(),"training check performed no parameter update");
    for(const auto& owner:model.parameters(false).owners())require(at::allclose(checkpoint.parameters.at(owner.canonical).to(at::kFloat),owner.value,1e-5,1e-6),"updated parameter differs: "+owner.canonical);
    for(size_t j=0;j<master_owners.size();++j){const auto& x=master_owners[j].value;
      require(at::allclose(checkpoint.state.values.narrow(0,checkpoint.offsets[j],x.numel()).reshape(x.sizes()),x,3e-5,3e-6),"independent FP32 optimizer master differs");}
    require(session->generation()==int64_t(update+1),"parameter generation did not advance");
  }
  session->close();std::cout<<"weighted-training case "<<topology<<'/'<<family<<'/'<<frozen<<" passed updates=4 windows=3 duplicate_ports=true\n"<<std::flush;
}
}
