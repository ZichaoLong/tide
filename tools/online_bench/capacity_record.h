#pragma once
#include "capacity.h"
#include <ostream>
namespace tide_flow::capacity {
inline void array(std::ostream& out,const std::vector<I>& x){out<<'[';for(size_t i=0;i<x.size();++i){if(i)out<<',';out<<x[i];}out<<']';}
inline void fields(std::ostream& out,const std::map<std::string,I>& values) {
  out<<'{';bool first=true;for(const auto& [k,v]:values){if(!first)out<<',';first=false;out<<'"'<<k<<"\":"<<v;}out<<'}';
}
inline void record(std::ostream& out,const Plan& p) {
  out<<"{\"schema\":\"tide-consumer-capacity-v1\",\"scope\":\"resident Add/Attention complete consumer; conservative shape estimate, not a vendor allocation guarantee\",\"devices\":[";
  bool first=true;for(const auto& card:p.cards) {
    if(!first)out<<',';first=false;
    out<<"{\"index\":"<<card.index<<",\"estimated_peak_bytes\":"<<card.peak<<",\"budget_bytes\":"<<card.budget
      <<",\"usable_bytes\":"<<card.usable<<",\"headroom_bytes\":"<<card.headroom<<",\"phases\":";fields(out,card.phases);
    out<<",\"components\":";fields(out,card.components);out<<'}';
  }
  out<<"],\"full_owners\":";array(out,p.owners);out<<",\"state_owners\":";array(out,p.owners);
  out<<",\"canonical_elements\":";array(out,p.canonical);out<<",\"requested_chunks\":";fields(out,p.requested);
  out<<",\"effective_chunks\":";fields(out,p.effective);
  out<<",\"physical_reductions\":"<<p.reductions<<",\"policy\":\""<<(p.aggressive?"aggressive":"conservative")<<"\"}";
}
} // namespace tide_flow::capacity
