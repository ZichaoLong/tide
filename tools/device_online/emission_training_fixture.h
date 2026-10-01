#pragma once
#include "retained_fixture.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online::test {
// Ordinary graph/module parameters, consumed independently by both schedules.
// Mixed broadcast/projection, phases, reused parameter owners and zero physical
// scales exercise structural adjoints rather than only nonzero output values.
inline void emission_training_fixture(Fixture& f,int variant=0) {
  at::NoGradGuard guard;const auto width=f.model.width();Tensor shared_weight,shared_bias;
  for(size_t n=0;n<f.graph.nodes.size();++n) {
    auto& node=f.graph.nodes[n];if(node.identity||n%3==1)continue;
    node.emission="slot_affine";node.emit_period=3;node.emit_phases.clear();
    const auto slots=f.graph.outgoing_ports.offsets[n+1]-f.graph.outgoing_ports.offsets[n];
    for(Index slot=0;slot<slots;++slot) {
      node.emit_phases.push_back(variant==2?-2:variant==1?(slot%4==1?-2:slot%4==2?1:-1):-1);
      auto w=at::eye(width,at::kFloat)*(.375f+.03125f*slot);
      if(width>1)w[0][width-1].fill_(.125f);
      auto bias=at::arange(width,at::kFloat)*.0078125f+.015625f*(slot+1);
      if(!shared_weight.defined()){shared_weight=w;shared_bias=bias;}
      // Share across physical slots and nodes; preserve one canonical update.
      f.model.nodes[n].extra["emit_w_"+std::to_string(slot)]=slot%2?shared_weight:w;
      f.model.nodes[n].extra["emit_b_"+std::to_string(slot)]=slot%2?shared_bias:bias;
    }
  }
  if(!f.model.edge_scale.empty())f.model.edge_scale[0].zero_();
  if(!f.model.output_scale.empty())f.model.output_scale[0].zero_();
  f.graph.compile();f.initial.identity=f.graph.identity;
}
} // namespace tide::device_online::test
