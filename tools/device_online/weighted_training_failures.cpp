#include "weighted_training_check.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online::weighted_training_check {
namespace {
template<class F>void refuses(F f,const std::string& name){bool refused=false;
  try{f();}catch(const std::logic_error&){refused=true;}require(refused,"accepted invalid weighted training "+name);}
}
void failures(at::Device device,at::ScalarType dtype){
  at::NoGradGuard ng;auto g=weighted_check::graph("fc","kv");
  auto m=weighted_check::model(g,"kv",at::kCPU,dtype);Continuation q;q.identity=g.identity;q.batch_size=2;
  auto l=limits();l.windows=1;
  auto small=l;small.total_bytes=1;refuses([&]{WeightedTrainingSession bad(g,m,q,device,ResidentOptimizerKind::sgd,{},small);},"total budget");
  WeightedTrainingSession session(g,m,q,device,ResidentOptimizerKind::sgd,{},l);const auto initial=session.checkpoint();
  auto xs=weighted_check::inputs(g,at::kCPU,dtype);std::vector<External> first;for(const auto& x:xs)if(x.time<4)first.push_back(x);
  auto w=session.advance(first,4);const auto good=seed(w,"outputs");
  refuses([&]{session.checkpoint();},"checkpoint with live tape");
  refuses([&]{session.step();},"step before backward");
  refuses([&]{session.advance({},4);},"window budget");require(session.cut()==4,"capacity refusal advanced cut");
  auto bad=good;++bad.token.index;refuses([&]{session.backward({bad});},"stale token");
  bad=good;bad.fields["unknown"]={};refuses([&]{session.backward({bad});},"unknown field");
  bad=good;bad.fields.at("outputs").values=bad.fields.at("outputs").values.to(at::kHalf);refuses([&]{session.backward({bad});},"root dtype");
  bad=good;bad.fields.at("outputs").connected=at::ones_like(bad.fields.at("outputs").connected);refuses([&]{session.backward({bad});},"absent root");
  bad=good;bad.fields.at("outputs").values=at::full_like(bad.fields.at("outputs").values,NAN);refuses([&]{session.backward({bad});},"connected poison");
  auto gradients=session.backward({good});refuses([&]{session.backward({good});},"reused backward");
  gradients.parameters.values.add_(0);refuses([&]{session.step();},"modified public gradient");session.close();
  auto checkpoint=initial;checkpoint.mode="hard";refuses([&]{WeightedTrainingSession bad(g,m,checkpoint,device,l);},"checkpoint binding");
  checkpoint=initial;checkpoint.aliases.clear();refuses([&]{WeightedTrainingSession bad(g,m,checkpoint,device,l);},"checkpoint aliases");
  checkpoint=initial;checkpoint.state.values=checkpoint.state.values+1;refuses([&]{WeightedTrainingSession bad(g,m,checkpoint,device,l);},"checkpoint named/master mismatch");
  WeightedTrainingSession empty(g,m,initial,device,l);
  for(int repeat=0;repeat<2;++repeat){auto view=empty.advance({},0);auto grad=empty.backward({seed(view,"none")});
    require(!grad.parameters.connected.any().item<bool>(),"empty window introduced parameter connection");empty.accumulate();}
  refuses([&]{empty.checkpoint();},"checkpoint with accumulated gradients");require(empty.step().applied,"empty gradient update failed");
  auto after=empty.checkpoint();require(after.state.steps.eq(0).all().item<bool>(),"None gradients advanced owner counters");
  require(at::equal(after.state.values,initial.state.values),"None gradients changed weights");empty.close();
  WeightedTrainingSession guarded(g,m,initial,device,l);auto output=guarded.advance(first,4);
  output.fields.at("outputs").present.logical_not_();refuses([&]{guarded.backward({seed(output,"none")});},"modified view support");guarded.close();
}
}
