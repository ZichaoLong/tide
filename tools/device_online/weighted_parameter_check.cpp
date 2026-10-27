// Canonical aliases across banks/roles, complete local training and owner resume.
#include "weighted_parameter_vjp.h"
#include "weighted_head.h"
#include "device_optimizer.h"
#include "sharded_parameter_banks.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <iostream>
namespace {
using namespace tide;using namespace tide::device_online;
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b){
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"canonical parameter comparison metadata");
  if(!at::allclose(a.cpu(),b,1e-5,1e-6)){std::cerr<<"actual="<<a.cpu()<<" expected="<<b<<'\n';throw std::runtime_error("canonical parameter numeric difference");}
}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto d:shape)n*=d;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape)*.2;}
std::vector<Tensor> tensors(const WeightedHeadParameters& w){return {w.matrix,w.hidden_bias,w.projection,w.bias};}
std::string name(int bank,int role){return "bank"+std::to_string(bank)+"."+std::vector<std::string>{"matrix","hidden","projection","bias"}[role];}
ParameterRegistry fixture(){
  auto a=values({2,2},.1),b=values({2},.2),c=values({2,2},.3);
  auto u=values({2,2},.4);ParameterRegistry r;
  std::vector<std::vector<Tensor>> banks{{a,b,c,values({2},.5)},{c,b,a,values({2},.6)},
    {u,values({2},.7),u.clone(),values({2},.8)}};
  for(int bank=0;bank<3;++bank)for(int role=0;role<4;++role)r.add(name(bank,role),banks[bank][role]);
  r.add("dead",at::ones({1},at::kFloat));return r;
}
WeightedHeadParameters pack(const ParameterRegistry& registry,at::Device device){
  std::vector<Tensor> fields;for(int role=0;role<4;++role){std::vector<Tensor> rows;
    for(int bank=0;bank<3;++bank)rows.push_back(registry.value(name(bank,role)));fields.push_back(at::stack(rows).to(device));}
  return {fields[0],fields[1],fields[2],fields[3]};
}
WeightedHeadParameters clone(const WeightedHeadParameters& w){return {w.matrix.clone(),w.hidden_bias.clone(),w.projection.clone(),w.bias.clone()};}
std::vector<Tensor> state(const ResidentOptimizerState& s){return {s.values,s.first,s.second,s.maximum,s.steps,s.corrections};}
void exact(const ResidentOptimizerState& a,const ResidentOptimizerState& b){auto x=state(a),y=state(b);
  for(size_t i=0;i<x.size();++i)require(at::equal(x[i].view(at::kByte),y[i].view(at::kByte)),"optimizer resume/failed commit changed exact state");}
struct Pipeline {
  WeightedHeadParameters weights;
  std::unique_ptr<DeviceProgram> program;
  std::unique_ptr<DeviceOptimizer> optimizer;
  WeightedHeadTape tape;ParameterVjp gradient;std::vector<WeightedParameterPartial> partials;
  Pipeline(const ParameterRegistry& registry,WeightedHeadParameters w,const Tensor& x,const Tensor& owners,
      const Tensor& bar,const Tensor& roots,const Tensor& error,DeviceOptimizerKind kind,const OptimizerGroup& group):weights(std::move(w)){
    program=std::make_unique<DeviceProgram>(x.device());program->limit_workspace(64*1024*1024);
    tape=append_weighted_head(*program,x,owners,weights,error);auto g=append_weighted_head_vjp(*program,tape,bar,roots,error);
    auto fields=tensors(g.parameters);
    for(int bank=0;bank<3;++bank)for(int role=0;role<4;++role)partials.push_back({name(bank,role),fields[role][bank],g.owner_connected[bank]});
    gradient=append_weighted_parameter_vjp(*program,registry,partials,error);
    optimizer=std::make_unique<DeviceOptimizer>(gradient,kind,std::vector<OptimizerGroup>{group},16*1024*1024);
    optimizer->append_step(*program,gradient,error);
    std::map<std::string,int64_t> offsets;for(size_t i=0;i<gradient.owners.size();++i)offsets[gradient.owners[i].canonical]=gradient.offsets[i];
    std::vector<ParameterWrite> writes;auto destinations=tensors(weights);
    for(int bank=0;bank<3;++bank)for(int role=0;role<4;++role)
      writes.push_back({offsets.at(registry.canonical_name(name(bank,role))),{destinations[role][bank],at::kFloat}});
    append_owner_parameter_publish(*program,writes,optimizer->values(),error,1024*1024);program->finish();
  }
  ~Pipeline(){program->close();}
};
std::vector<Tensor> reference(ParameterRegistry& registry,const Tensor& x,const Tensor& owners,
    const Tensor& bar,const Tensor& roots,const Pipeline& candidate){
  at::AutoGradMode enable(true);std::vector<Tensor> outputs,ys,bars,leaves;
  for(const auto& owner:registry.owners())leaves.push_back(owner.value);
  for(int row=0;row<4;++row){auto bank=owners[row].item<int64_t>();auto input=at::mv(registry.value(name(bank,0)),x[row])+registry.value(name(bank,1));
    auto y=at::mv(registry.value(name(bank,2)),at::silu(input))+registry.value(name(bank,3));outputs.push_back(y);
    if(roots[row].item<bool>()){ys.push_back(y);bars.push_back(bar[row]);}}
  close(candidate.tape.output,at::stack(outputs).detach());
  return torch::autograd::grad(ys,leaves,bars,false,false,true);
}
void trajectory(at::Device device,DeviceOptimizerKind kind){
  at::NoGradGuard no_grad;auto registry=fixture();ParameterRegistry cpu;
  for(const auto& owner:registry.owners()){auto x=owner.value.clone().set_requires_grad(true);for(const auto& alias:owner.aliases)cpu.add(alias,x);}
  require(registry.owners().size()==10,"fixture alias partition changed");
  require(registry.canonical_name(name(2,0))!=registry.canonical_name(name(2,2)),"equal independent parameters collapsed");
  OptimizerGroup group;group.lr=.01;group.weight_decay=.03;group.momentum=.75;group.nesterov=true;
  group.beta1=.8;group.beta2=.95;group.eps=1e-5;group.amsgrad=true;
  for(const auto& owner:registry.owners())group.parameters.push_back(owner.canonical);
  std::unique_ptr<NamedOptimizer> reference_optimizer;
  if(kind==DeviceOptimizerKind::sgd)reference_optimizer=std::make_unique<SGD>(cpu,std::vector<OptimizerGroup>{group});
  else reference_optimizer=std::make_unique<AdamW>(cpu,std::vector<OptimizerGroup>{group});
  auto x=at::zeros({4,2},at::TensorOptions().device(device).dtype(at::kFloat)),bar=at::zeros_like(x);
  auto owners=at::tensor(std::vector<int64_t>{0,1,0,2},at::kLong).to(device),roots=at::zeros({4},x.options().dtype(at::kBool));
  auto error=at::zeros({1},x.options().dtype(at::kInt));Pipeline live(registry,pack(registry,device),x,owners,bar,roots,error,kind,group);
  std::unique_ptr<Pipeline> restored;std::vector<int64_t> counts(10,0);
  for(int step=0;step<4;++step){
    auto input=values({4,2},.3+step*.1),cotangent=values({4,2},.8+step*.1);
    if(step==2)cotangent.zero_();cotangent[3].fill_(NAN);
    auto flags=at::tensor({int(step!=1),1,int(step%2==0),0},at::kInt).to(at::kBool);
    auto ids=at::tensor(std::vector<int64_t>{0,1,step%2,2},at::kLong);
    x.copy_(input);bar.copy_(cotangent);roots.copy_(flags);owners.copy_(ids);portable_torch::synchronize(device);live.program->run();
    require(error.cpu().item<int>()==0,"weighted canonical training refused valid step");
    auto grads=reference(cpu,input,ids,cotangent,flags,live);auto actual=live.gradient.values.cpu(),connected=live.gradient.connected.cpu();
    for(size_t i=0;i<grads.size();++i){const auto& owner=live.gradient.owners[i];
      require(connected[i].item<bool>()==grads[i].defined(),"canonical alias None/zero differs");
      cpu.value(owner.canonical).mutable_grad()=grads[i];
      if(grads[i].defined()){++counts[i];close(actual.narrow(0,live.gradient.offsets[i],owner.value.numel()).reshape(owner.value.sizes()),grads[i]);}}
    reference_optimizer->step();const auto snapshot=live.optimizer->snapshot();
    for(size_t i=0;i<grads.size();++i){const auto& owner=live.gradient.owners[i];const auto offset=live.gradient.offsets[i],size=owner.value.numel();
      require(snapshot.steps[i].item<int64_t>()==counts[i],"canonical None/zero optimizer counter");
      if(offset<0)continue;
      close(snapshot.values.narrow(0,offset,size).reshape(owner.value.sizes()),cpu.value(owner.canonical));
      auto found=reference_optimizer->state().find(owner.canonical);if(found==reference_optimizer->state().end())continue;
      const auto& s=found->second;
      for(const auto& pair:std::vector<std::pair<Tensor,Tensor>>{{snapshot.first,kind==DeviceOptimizerKind::sgd?s.momentum_buffer:s.exp_avg},
          {snapshot.second,s.exp_avg_sq},{snapshot.maximum,s.max_exp_avg_sq}})
        if(pair.second.defined())close(pair.first.narrow(0,offset,size).reshape(owner.value.sizes()),pair.second);
    }
    if(restored){portable_torch::synchronize(device);restored->program->run();
      require(error.cpu().item<int>()==0,"restored canonical training failed");exact(snapshot,restored->optimizer->snapshot());
      auto a=tensors(live.weights),b=tensors(restored->weights);for(int i=0;i<4;++i)close(a[i],b[i].cpu());}
    if(step==1){restored=std::make_unique<Pipeline>(registry,clone(live.weights),x,owners,bar,roots,error,kind,group);
      restored->optimizer->restore(snapshot);exact(snapshot,restored->optimizer->snapshot());}
  }
  const auto before=live.optimizer->snapshot();auto published=tensors(clone(live.weights));x.fill_(NAN);
  portable_torch::synchronize(device);live.program->run();require(error.cpu().item<int>()==20,"nonfinite input reached optimizer");
  exact(before,live.optimizer->snapshot());auto after=tensors(live.weights);for(int i=0;i<4;++i)close(after[i],published[i].cpu());
  DeviceProgram refusal(device);bool rejected=false;
  try{append_weighted_parameter_vjp(refusal,registry,live.partials,error,1);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"weighted parameter scratch budget omitted");refusal.close();
}
void overflow(at::Device device){
  at::NoGradGuard no_grad;ParameterRegistry registry;registry.add("a",at::ones({513},at::kFloat));registry.add("alias",registry.value("a"));
  auto opts=at::TensorOptions().device(device).dtype(at::kFloat);auto root=at::ones({1},opts.dtype(at::kBool));
  auto values=at::full({513},3e38,opts),error=at::zeros({1},opts.dtype(at::kInt));DeviceProgram p(device);
  auto result=append_weighted_parameter_vjp(p,registry,{{"a",values,root},{"alias",values,root}},error);p.finish();
  portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==23,"canonical tiled sum overflow accepted");
  error.zero_();root.zero_();values.fill_(NAN);portable_torch::synchronize(device);p.run();
  require(error.cpu().item<int>()==0&&!result.connected.cpu().any().item<bool>()&&result.values.cpu().eq(0).all().item<bool>(),"canonical absent poison read");p.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted canonical parameters require explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    trajectory(device,DeviceOptimizerKind::sgd);trajectory(device,DeviceOptimizerKind::adamw);overflow(device);
    std::cout<<"weighted-parameters: passed optimizers=2 updates=8 canonical_owners=10 cross_role_aliases=true resume_next_steps=true finite_transaction=true\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
