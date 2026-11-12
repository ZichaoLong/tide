#pragma once
#include "weighted_flow_fixture.h"
#include "tide/settle.h"
#include "tide/kernel.h"
namespace tide::device_online::block_public_check {
struct Fixture {SettleGraph graph;Model model;Continuation initial;};
inline Fixture fixture(){
  weighted::RegionalConfig r("regional_total");r.hidden_width=3;r.k=2;r.gamma_max=1.6;r.validate();
  auto [g,m]=weighted_flow_check::configured(r,true,true,false);
  // The public Model retains physically duplicated edges and same-class atoms.
  for(auto& e:g.edges)e.delay=1;g.compile();SettleGraph graph(g,{1,2},"ports",3);
  for(auto& w:m.nodes){const std::string name="aggregate_program.transforms.1.";
    for(const auto& role:{"a_left","a_right","b_left","b_right"}){const auto key=name+role;const auto& old=w.extra.at(key);
      const bool left=std::string(role).find("left")!=std::string::npos;auto shape=old.sizes().vec();shape[left?1:0]=2;
      w.extra[key]=at::full(shape,.03,old.options()).set_requires_grad(true);}}
  const auto shared=m.nodes[0].extra.at("aggregate_program.transforms.0.a");
  for(auto& w:m.nodes)w.extra["aggregate_program.transforms.0.a"]=shared;
  Continuation q;q.identity=g.identity;q.batch_size=2;q.cut=3;
  const auto& w=m.nodes[0];const auto h=at::tensor({.2f,.1f,-.2f});
  q.states[{0,0}]=w.kernel->step(w,w.kernel->initial(w),ContentView{h},1);
  q.history[{0,0}].last_time=1;q.ledger[{0,0}]={0,1};
  return {std::move(graph),std::move(m),std::move(q)};
}
}
