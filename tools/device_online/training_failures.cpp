#include "training_test.h"
#include <ATen/core/grad_mode.h>
#include <limits>

namespace tide::device_online::test {
void train_failures(at::Device device) {
  at::NoGradGuard guard;auto f=retained_fixture(0,0,3);f.model=train_model(f.model,at::kFloat);
  ResidentTrainingLimits limits;limits.forward.trace=512;limits.windows=1;
  {at::AutoGradMode enabled(true);train_reject([&]{ResidentTrainingSession s(f.graph,f.model,f.initial,device);},"implicit autograd accepted");}
  auto tiny=limits;tiny.retained_bytes=1;
  train_reject([&]{ResidentTrainingSession s(f.graph,f.model,f.initial,device,ResidentOptimizerKind::sgd,{},tiny);},"retained byte budget ignored");
  auto wrong=f;wrong.graph.nodes[0].full="relu";wrong.graph.compile();wrong.initial.identity=wrong.graph.identity;
  train_reject([&]{ResidentTrainingSession s(wrong.graph,wrong.model,wrong.initial,device);},"unavailable Full adjoint accepted");
  auto overlap=f;overlap.model.nodes[1].weight=overlap.model.nodes[0].weight.view({3,3});
  train_reject([&]{ResidentTrainingSession s(overlap.graph,overlap.model,overlap.initial,device);},"overlapping distinct parameter owners accepted");
  ResidentTrainingSession s(f.graph,f.model,f.initial,device,ResidentOptimizerKind::adamw,{},limits);
  const auto initial=s.checkpoint();auto bad=initial;bad.schema=0;
  train_reject([&]{ResidentTrainingSession r(f.graph,f.model,bad,device,limits);},"checkpoint schema ignored");
  bad=initial;bad.aliases.pop_back();train_reject([&]{ResidentTrainingSession r(f.graph,f.model,bad,device,limits);},"checkpoint aliases ignored");
  bad=initial;bad.state.values=bad.state.values.clone().add_(1);
  train_reject([&]{ResidentTrainingSession r(f.graph,f.model,bad,device,limits);},"packed/named checkpoint conflict ignored");
  bad=initial;bad.state.steps=bad.state.steps.clone().fill_(-1);
  train_reject([&]{ResidentTrainingSession r(f.graph,f.model,bad,device,limits);},"negative restored counter accepted");
  train_reject([&]{s.advance(f.input,11,10);},"unsealed training input accepted");
  auto invalid=f.input;invalid[0].position=9;
  train_reject([&]{s.advance(invalid,11,11);},"invalid input ledger accepted");train_require(s.cut()==0,"rejected input changed cut");
  auto w=s.advance(f.input,11,11);auto roots=train_roots(w,0,4);
  train_reject([&]{s.advance({},11,11);},"window capacity ignored");train_require(s.cut()==11&&s.retained_windows()==1,"capacity refusal mutated session");
  auto malformed=roots;malformed.outputs_connected=Tensor();train_reject([&]{s.backward({malformed});},"incomplete root pair accepted");
  malformed=roots;++malformed.token.session;train_reject([&]{s.backward({malformed});},"foreign session token accepted");
  roots.final.fill_(std::numeric_limits<float>::quiet_NaN());s.backward({roots});
  auto refusal=s.step();train_require(!refusal.applied&&refusal.refusal_code==20&&s.generation()==0,"nonfinite optimizer was not transactional");
  s.detach();auto rejected=s.checkpoint();
  train_require(at::equal(initial.state.values,rejected.state.values)&&at::equal(initial.state.first,rejected.state.first)
    &&at::equal(initial.state.steps,rejected.state.steps),"nonfinite refusal partially changed parameters/slots");
  auto later=s.advance({},11,11);train_reject([&]{s.backward({roots});},"detached root token remained valid");
  s.backward({train_roots(later,0,0)});train_require(s.step().applied,"None update failed after explicit detach");
  s.close();
  auto limited=limits;limited.forward.stages=1;
  ResidentTrainingSession failed(f.graph,f.model,f.initial,device,ResidentOptimizerKind::sgd,{},limited);
  train_reject([&]{failed.advance(f.input,11,11);},"stage overflow ignored");
  train_reject([&]{failed.checkpoint();},"failed forward exported incomplete cut");failed.close();
  // Implicit zero initial states are not caller-owned differentiable leaves.
  auto empty=f.initial;empty.states.clear();ResidentTrainingSession absent(f.graph,f.model,empty,device);
  auto aw=absent.advance(f.input,11,11);auto ag=absent.backward({train_roots(aw,0,4)});
  train_require(!ag.initial_connected.any().cpu().item<bool>()&&ag.initial.eq(0).all().cpu().item<bool>(),"absent initial states became leaves");absent.detach();
  // A selected trainable owner must retain all of its aliases, including Read.
  auto subset=f;subset.model=train_model(f.model,at::kFloat);
  for(auto& owner:subset.model.parameters(false).owners())owner.value.set_requires_grad(false);
  subset.model.nodes[0].bias.set_requires_grad(true);
  ResidentTrainingSession selected(subset.graph,subset.model,subset.initial,device);
  auto sw=selected.advance(subset.input,11,11);auto sg=selected.backward({train_roots(sw,0,4)});
  train_require(sg.names.size()==1&&sg.aliases[0].size()==2,"trainable subset lost shared Read alias");selected.step();
  for(auto& owner:subset.model.parameters(false).owners())owner.value.set_requires_grad(false);
  ResidentTrainingSession frozen(subset.graph,subset.model,subset.initial,device);
  auto fw=frozen.advance(subset.input,11,11);auto fg=frozen.backward({train_roots(fw,0,4)});
  train_require(fg.names.empty()&&!fg.connected.any().cpu().item<bool>()&&frozen.step().applied,"empty trainable registry fabricated owners");frozen.close();
  f.model.nodes[0].bias.add_(1);train_reject([&]{absent.advance({},12,12);},"mutated caller parameter accepted");absent.close();
}
} // namespace tide::device_online::test
