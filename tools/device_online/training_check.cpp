#include "training_test.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>

namespace {
using namespace tide;using namespace tide::device_online;
void trajectory(at::Device device,int shape,int variant,int64_t width,bool prefill,ResidentOptimizerKind kind,at::ScalarType dtype) {
  at::NoGradGuard guard;auto f=test::retained_fixture(shape,variant,width);f.model=test::train_model(f.model,at::kFloat);
  auto cpu=f;cpu.model=test::train_model(f.model,dtype);cpu.initial=test::train_boundary(f.initial,dtype);
  auto registry=cpu.model.parameters(true);OptimizerGroup group;
  group.lr=.001;group.momentum=.875;group.nesterov=true;group.weight_decay=.0125;group.amsgrad=true;
  for(const auto& owner:registry.owners())group.parameters.push_back(owner.canonical);
  std::unique_ptr<NamedOptimizer> optimizer;
  if(kind==ResidentOptimizerKind::sgd)optimizer=std::make_unique<SGD>(registry,std::vector<OptimizerGroup>{group});
  else optimizer=std::make_unique<AdamW>(registry,std::vector<OptimizerGroup>{group});
  ResidentTrainingLimits limits;limits.forward.prefill=prefill;limits.forward.trace=512;limits.forward.full_chunk_rows=3;
  limits.reverse_chunk_rows=3;if(width>3)limits.forward.workspace_bytes=512*1024*1024;
  auto session=std::make_unique<ResidentTrainingSession>(f.graph,f.model,f.initial,device,kind,std::vector<OptimizerGroup>{group},limits);
  const auto original=session->checkpoint();
  for(int step=0;step<4;++step) {
    const auto start=session->cut();const int mode=step==1?5:step==2?0:4;
    std::vector<External> all=f.input;for(auto& x:all){x.time+=step*11;x.position+=step*(x.batch==0?3:2);}
    cpu.input=all;auto ref=test::retained_reference(cpu,mode,dtype);
    std::vector<ResidentCotangents> roots;int w=0;Index cut=start;
    for(auto stop:test::retained_stops(start)) {
      std::vector<External> input;for(auto x:all)if(cut<=x.time&&x.time<stop){if(step==1)x.value=x.value.to(device);input.push_back(x);}
      auto window=session->advance(input,stop,stop);roots.push_back(test::train_roots(window,w,mode));
      if(dtype==at::kFloat)tide_bench::compare(session->result(),ref.windows[w],true,at::kFloat);
      ++w;cut=stop;
    }
    test::train_reject([&]{session->checkpoint();},"checkpoint accepted unconsumed windows");
    test::train_reject([&]{session->step();},"optimizer crossed outstanding windows");
    auto bad=roots;std::swap(bad[0],bad[1]);test::train_reject([&]{session->backward(bad);},"out-of-order roots accepted");
    auto gradient=session->backward(roots);test::train_gradients(gradient,ref,cpu);
    test::train_reject([&]{session->backward(roots);},"backward reused consumed roots");
    test::train_reject([&]{session->advance({},cut,cut);},"advance bypassed unconsumed gradients");
    for(const auto& o:registry.owners())o.value.mutable_grad()=ref.gradients.at(o.canonical);
    optimizer->step();const auto updated=session->step();test::train_require(updated.applied&&updated.generation==step+1,"public optimizer did not update");
    auto checkpoint=session->checkpoint();test::train_checkpoint(checkpoint,cpu.model,*optimizer);
    if(step==0)test::train_require(!at::equal(checkpoint.state.values,original.state.values)&&checkpoint.state.steps.gt(0).any().item<bool>(),"trajectory performed no real parameter updates");
    cpu.initial=test::train_boundary(ref.windows.back().continuation,dtype);
    if(step==1) {
      session->close();session=std::make_unique<ResidentTrainingSession>(f.graph,f.model,checkpoint,device,limits);
      auto restored=session->checkpoint();test::train_require(at::equal(checkpoint.state.corrections,restored.state.corrections),"Adam correction restart changed");
      for(auto& [_,p]:checkpoint.parameters)p.fill_(123); // Exports must own storage.
    }
  }
  session->close();session->close();test::train_reject([&]{session->checkpoint();},"closed training owner exported state");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("public training check requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int shape:{0,3})for(bool prefill:{false,true})for(auto kind:{ResidentOptimizerKind::sgd,ResidentOptimizerKind::adamw})for(auto dtype:{at::kFloat,at::kDouble}) {
      try{trajectory(device,shape,0,3,prefill,kind,dtype);++cases;}
      catch(...){std::cerr<<"public training shape="<<shape<<" prefill="<<prefill<<" kind="<<int(kind)<<" dtype="<<dtype<<'\n';throw;}
    }
    trajectory(device,0,0,257,true,ResidentOptimizerKind::adamw,at::kFloat);++cases;
    trajectory(device,3,1,3,true,ResidentOptimizerKind::sgd,at::kDouble);++cases;
    test::train_failures(device);
    std::cout<<"public-resident-training: passed trajectories="<<cases<<" windows="<<cases*16<<" updates="<<cases*4
      <<" CPU=FP32_FP64 checkpoint_resume=true lifecycle=true scope=single_NPU_HARD_subset\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
