#include "weighted_flow_internal.h"
#include <limits>
namespace tide::device_online {
int64_t weighted_flow_preflight(const Graph& g,const Model& m,int64_t samples,const WeightedFlowLimits& l){
  const long double atom=l.queue*(49.L+4.L*m.width());
  long double persistent=2*atom+44.L+16.L*samples*g.regions.size();
  long double journal=2*atom+24.L+9.L*samples*g.regions.size()+8.L*(5.L*l.queue+1);
  long double boundary=49.L*l.queue+8;
  for(const auto& w:m.nodes){const auto b=weighted::node_binding(w);
    auto layout=std::visit([&](const auto& c)->WeightedRawLayout {
      using C=std::decay_t<decltype(c)>;
      if constexpr(std::is_same_v<C,weighted::KVConfig>)return WeightedRawLayout(c,l.cache);
      else return WeightedRawLayout(c,c.basic()?0:l.cache);
    },b.memory);
    const long double raw=samples*(4.L*layout.floating_width()+8.L*layout.integer_width());
    persistent+=3*raw+8.L*samples;
    boundary+=8.L*samples*(layout.integer_width()+1);
    const auto& r=b.regional;const auto descriptors=3+(r.regional()?*r.region_control_width:0)+(r.region_score=="att"?2**r.selection_width:0);
    journal+=2*raw+samples*(14.L+8.L*m.width()+4.L*(m.width()-1)+4.L*descriptors);
  }
  journal=journal*l.stages+boundary;
  if(persistent+journal>l.tensor_bytes||journal>std::numeric_limits<int64_t>::max())
    throw std::invalid_argument("weighted persistent state and window journal exceed total tensor budget before allocation");
  return static_cast<int64_t>(journal);
}
int64_t WeightedFlow::Impl::remaining(DeviceProgram& p)const{
  std::vector<at::Tensor> held{error,begin,stop,stages,external.coordinates,external.values,external.valid,
    pending->atoms().coordinates,pending->atoms().values,pending->atoms().valid,pending->stats()};
  for(const auto& n:nodes){for(const auto& raw:{n->initial,n->state,n->old}){held.push_back(raw.floating);held.push_back(raw.integer);}
    held.push_back(n->state_present);for(const auto& [_,role]:n->banks.roles())held.push_back(role.bank);}
  for(const auto& r:regions){held.push_back(r->history);for(const auto& [_,role]:r->banks.roles())held.push_back(role.bank);}
  auto programs=services->programs();
  if(peers){auto ps=peers->programs();programs.insert(programs.end(),ps.begin(),ps.end());
    auto xs=peers->tensors();held.insert(held.end(),xs.begin(),xs.end());}
  const auto used=p.retained_tensor_bytes(held,programs),reserved=journal_bytes-journal->allocated_bytes();
  if(used>limits.tensor_bytes-reserved)throw std::invalid_argument("complete weighted flow tensor budget exceeded");
  return limits.tensor_bytes-used-reserved;
}
}
