#include "state_reverse_view.h"
#include <limits>
#include <stdexcept>

namespace tide::device_online {
StateReverseLayout::StateReverseLayout(const std::vector<Node>& ns,const std::vector<int64_t>& counts,
    int64_t w,int64_t in,int64_t edges):nodes(ns.size()),width(w),inputs(in),sources(0),source_counts(counts) {
  if(nodes<1||w<1||in<0||edges<0||in>std::numeric_limits<int64_t>::max()-edges||counts.size()!=ns.size())
    throw std::invalid_argument("invalid compact state reverse layout");
  sources=in+edges;
  for(const auto n:counts)if(n<0)throw std::invalid_argument("negative compact source domain");
  event_offsets=event_parameter_offsets(ns,w);fiber_offsets=fiber_parameter_offsets(ns,counts,w);
}
namespace {
StateReverseLayout whole_layout(const ReverseTape& t) {
  if(!t.graph)throw std::invalid_argument("state reverse view requires validated graph metadata");
  return {t.graph->nodes,t.graph->source_counts,t.full.width,int64_t(t.graph->inputs.size()),int64_t(t.graph->edges.size())};
}
}
StateReverseView::StateReverseView(const ReverseTape& t)
    :layout(whole_layout(t)),state(t.state),fiber_meta(t.fiber_meta),fiber_values(t.fiber_values),sources(t.sources) {}
StateReverseView::StateReverseView(StateReverseLayout l,StateTape s,at::Tensor m,at::Tensor v,at::Tensor src)
    :layout(std::move(l)),state(std::move(s)),fiber_meta(std::move(m)),fiber_values(std::move(v)),sources(std::move(src)) {}
} // namespace tide::device_online
