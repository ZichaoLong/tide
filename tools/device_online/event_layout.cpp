#include "event_tape.h"
#include <limits>
#include <stdexcept>
namespace tide::device_online {
std::vector<int64_t> event_parameter_offsets(const Graph& g,int64_t width) {
  return event_parameter_offsets(g.nodes,width);
}
std::vector<int64_t> event_parameter_offsets(const std::vector<Node>& nodes,int64_t width) {
  std::vector<int64_t> out;int64_t offset=0;
  for(const auto& n:nodes) {
    out.push_back(offset);
    if(!n.identity&&n.memory=="attention") {
      if(width<1||n.query_heads<1||n.kv_heads<1||width%n.query_heads||n.query_heads%n.kv_heads)
        throw std::invalid_argument("invalid event attention parameter geometry");
      const long double next=offset+2.L*width*(width+width/n.query_heads*n.kv_heads);
      if(next>std::numeric_limits<int64_t>::max())throw std::invalid_argument("event parameter extent overflow");
      offset=static_cast<int64_t>(next);
    }
  }
  out.push_back(offset);return out;
}
} // namespace tide::device_online
