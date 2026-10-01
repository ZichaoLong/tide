#pragma once
#include "retained_fixture.h"
#include <array>

namespace tide::device_online::test {
// Public input/parameter fixtures, shared by candidate and independent oracle.
// The scheduler sees ordinary legal feedback graphs, never a precomputed trace.
inline Fixture precision_graph_profile(int shape,int variant,Index width,int profile) {
  auto f=retained_fixture(shape,variant,width);
  if(profile==0)return f;
  if(profile<1||profile>16)throw std::invalid_argument("unknown precision graph profile");
  const std::array<std::string,5> aggregates{"sum","mean","weighted_mean","active_softmax","all_softmax"};
  const std::array<std::string,3> activations{"relu","silu","identity"},norms{"identity","rms","layer"};
  Tensor shared_norm,shared_gate;
  for(size_t n=0;n<f.graph.nodes.size();++n) {
    auto& node=f.graph.nodes[n];auto& w=f.model.nodes[n];if(node.identity)continue;
    if(profile<=5||profile==16) {
      const auto k=profile>=5?int(n%5):profile;node.aggregation=aggregates[k];
      if(k>=2)for(Index slot=0;slot<f.graph.source_counts[n];++slot)
        w.extra[(k==2?"agg_mass_":"agg_logit_")+std::to_string(slot)]=slot==0?f.model.input_scale[0]:
          at::full({},.125f*float(slot)-.25f,at::kFloat);
    }
    if(profile>=6) {
      const auto kind=profile==16?int(n%3==0?9:n%3==1?5:-1):profile-6;
      if(kind<0)continue; // Mixed tanh keeps its ordinary parameter banks.
      if(kind==9) {
        node.full="swiglu";auto eye=at::eye(width,at::kFloat);
        if(!shared_gate.defined())shared_gate=at::cat({eye*.25f,eye*-.125f},1);
        w.extra["ffn_gate"]=shared_gate;w.extra["ffn_up"]=shared_gate;
        w.extra["ffn_down"]=at::cat({eye*.375f,eye*.125f},0);
      } else {
        node.full="lh-"+activations[kind/3]+"-"+norms[kind%3]+"-v1";
        if(kind%3) {
          if(!shared_norm.defined())shared_norm=at::arange(width,at::kFloat).remainder(5)*.0625f+.75f;
          w.extra["lh_norm_weight"]=shared_norm;
        }
        if(kind%3==2)w.extra["lh_norm_bias"]=at::arange(width,at::kFloat).remainder(3)*.03125f;
      }
    }
  }
  f.graph.compile();f.initial.identity=f.graph.identity;return f;
}
} // namespace tide::device_online::test
