// Separate algebraic anchor: two detached groups sum once at the optimizer
// boundary. No normalization, implicit per-window step or public-buffer alias.
#include "weighted_training_check.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online::weighted_training_check {
void accumulation(at::Device device){
  at::NoGradGuard ng;auto graph=weighted_check::graph("fc","kv");auto model=weighted_check::model(graph,"kv",at::kCPU,at::kFloat);
  Continuation initial;initial.identity=graph.identity;initial.batch_size=2;OptimizerGroup group;
  for(const auto& owner:model.parameters().owners())group.parameters.push_back(owner.canonical);group.lr=.002;
  WeightedTrainingSession session(graph,model,initial,device,ResidentOptimizerKind::sgd,{group},limits());
  auto before=session.checkpoint();auto xs=weighted_check::inputs(graph,at::kCPU,at::kFloat);
  std::vector<External> head,tail;for(const auto& x:xs)(x.time<4?head:tail).push_back(x);
  auto one=session.advance(head,4);auto first=session.backward({seed(one,"outputs")});
  auto a=first.parameters.values.cpu().clone(),ac=first.parameters.connected.cpu().clone();session.accumulate();
  first.parameters.values.fill_(NAN);first.parameters.connected.zero_();
  auto two=session.advance(tail,10);auto second=session.backward({seed(two,"outputs")});
  auto b=second.parameters.values.cpu().clone(),bc=second.parameters.connected.cpu().clone();session.accumulate();
  auto expected=before.state.values.clone();auto expected_steps=at::zeros_like(before.state.steps);
  for(size_t i=0;i<second.parameters.names.size();++i){const auto size=model.parameters().value(second.parameters.names[i]).numel(),offset=second.parameters.offsets[i];
    auto sum=at::zeros({size},at::kFloat);if(ac[i].item<bool>())sum.add_(a.narrow(0,offset,size));if(bc[i].item<bool>())sum.add_(b.narrow(0,offset,size));
    if(ac[i].item<bool>()||bc[i].item<bool>()){expected.narrow(0,offset,size).add_(sum,-group.lr);expected_steps[i]=1;}}
  require(session.step().applied,"accumulated weighted step refused");auto after=session.checkpoint();
  require(!at::equal(after.state.values,before.state.values),"accumulation check performed no update");
  require(at::allclose(after.state.values,expected,1e-5,1e-6),"weighted groups did not sum at one update boundary");
  require(at::equal(after.state.steps,expected_steps),"weighted accumulated owner counters differ");session.close();
}
}
