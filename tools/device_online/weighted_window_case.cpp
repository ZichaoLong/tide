// Whole-window chain rule against an independently scheduled CPU graph.
#include "weighted_window_check.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/core/grad_mode.h>
#include <iostream>
namespace tide::device_online::window_check {
namespace {
void leaf(at::Tensor& x){x=x.detach().clone().set_requires_grad(true);}
void leaf(Continuation& q){for(auto& a:q.pending)leaf(a.value);for(auto& [key,s]:q.states){leaf(s.value);
  for(auto& [name,x]:s.slots)if(x.scalar_type()==at::kFloat)leaf(x);}}
}
void run_window(at::Device device,Graph g,Model m,std::vector<External> xs,
    const std::string& label,bool split,bool prefill,bool retained){
  auto mark=[&](const std::string& phase){std::cout<<"weighted-window phase "<<label<<' '<<phase<<'\n'<<std::flush;};
  for(auto& x:xs)leaf(x.value);
  Continuation q;q.identity=g.identity;q.batch_size=2;WeightedFlowLimits limits;limits.cache=16;limits.stages=32;limits.queue=64;
  limits.tensor_bytes=768*1024*1024;limits.operator_bytes=128*1024*1024;limits.timeout_ms=60000;limits.prefill=prefill;
  at::NoGradGuard ng;mark("construct-forward");WeightedFlow forward(g,m,q,device,limits);
  if(split){std::vector<External> first,tail;for(const auto& x:xs)(x.time<4?first:tail).push_back(x);
    mark("advance-head");q=forward.advance(first,4).continuation;leaf(q);xs=tail;}
  mark("advance-tail");auto actual=forward.advance(xs,10);
  mark("budget-refusal");bool refused=false;
  try{WeightedWindowAdjoint too_small(forward,forward.reserved_bytes());}catch(const std::invalid_argument&){refused=true;}
  if(!refused)throw std::runtime_error("window VJP admitted roots without room beyond the forward owner");
  std::shared_ptr<WeightedWindowSnapshot> saved;
  if(retained){
    bool refused=false;try{WeightedWindowSnapshot::retain(forward,1);}catch(const std::invalid_argument&){refused=true;}
    if(!refused)throw std::runtime_error("retained window omitted capacity preflight");
    mark("retain");saved=WeightedWindowSnapshot::retain(forward,limits.tensor_bytes);
    mark("overwrite-forward");forward.advance({},12);forward.close();saved->validate();
  }
  mark("construct-adjoint");auto owner=retained?std::make_unique<WeightedWindowAdjoint>(saved):std::make_unique<WeightedWindowAdjoint>(forward);auto& vjp=*owner;
  mark("decode-adjoint");const auto records=vjp.records();
  Result expected;{at::AutoGradMode enabled(true);Options options;options.packed=false;expected=Greedy(g,m,options).run(q,xs,10,10);}
  tide_bench::compare(actual,expected,true,at::kFloat);
  for(const std::string mode:{"none","full","content","control","describe","old","proposal","final","messages","fibers","pending","all"}){
    mark("seed-"+mode);seed(vjp,records,actual,mode);mark("run-"+mode);vjp.run();mark("verify-"+mode);
    verify(g,m,q,xs,expected,vjp,records,mode);}
  if(retained){
    // Even an in-place write of the same exact integer is a saved-version edit.
    saved->stages.add_(0);bool refused=false;try{vjp.run();}catch(const std::logic_error&){refused=true;}
    if(!refused)throw std::runtime_error("modified saved window metadata accepted");
    vjp.close();std::cout<<"weighted-retained case "<<label<<" passed modes=12 closed_forward=true mutation_refused=true\n"<<std::flush;return;
  }
  mark("close-adjoint");vjp.close();
  // An empty new window is an identity on exact checkpoint leaves and pending.
  mark("empty-advance");q=forward.checkpoint();leaf(q);auto empty=forward.advance({},10);mark("empty-adjoint");WeightedWindowAdjoint identity(forward);const auto empty_records=identity.records();
  seed(identity,empty_records,empty,"all");mark("empty-run");identity.run();mark("empty-verify");verify(g,m,q,{},Result{q},identity,empty_records,"empty");
  mark("stale-generation");forward.advance({},10);bool stale=false;try{identity.run();}catch(const std::logic_error&){stale=true;}
  if(!stale)throw std::runtime_error("adjoint silently reused a different forward generation");identity.close();forward.close();
  std::cout<<"weighted-window case "<<label<<" split="<<split<<" prefill="<<prefill<<" passed modes=12 stages="<<actual.stats.at("resident_stages")<<'\n'<<std::flush;
}
}
