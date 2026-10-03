#include "tide/delivery.h"
#include "tide/transfer.h"
#include <algorithm>

namespace tide {
Delivery prepare_delivery(const Graph& g,const Model& m,const std::vector<Event>& events,
                          bool packed,std::map<std::string,Index>& stats) {
  Delivery result;
  using Key=std::tuple<std::string,std::string,at::ScalarType,std::vector<Index>>;
  std::map<Key,std::vector<size_t>> groups;
  for(const auto& e:events)if(e.active) {
    const auto start=g.outgoing_ports.offsets[e.node];
    for(const auto& emission:e.emitted) {
      const auto& binding=g.outgoing_ports.bindings[start+emission.slot];
      const auto id=binding.id;
      if(binding.kind==1) {
        const auto& edge=g.edges[id];
        if(e.time>std::numeric_limits<Index>::max()-edge.delay)throw std::overflow_error("logical time overflow");
        auto value=emission.value*m.edge_scale[id].to(emission.value.device());
        const auto target=m.nodes[edge.target].bias.device();
        if(value.device()!=target)
          groups[{value.device().str(),target.str(),value.scalar_type(),value.sizes().vec()}].push_back(result.messages.size());
        result.messages.push_back({e.batch,edge.target,e.time+edge.delay,1,id,e.time,value});
      } else result.outputs.push_back({e.batch,e.time,id,emission.value*m.output_scale[id].to(emission.value.device())});
    }
  }
  for(const auto& [key,ids]:groups) {
    const auto target=at::Device(std::get<1>(key));
    const auto& first=result.messages[ids[0]].value;
    const Index bytes=first.numel()*first.element_size();
    const auto limit=packed?std::max<Index>(1,transfer_pack_bytes/bytes):1;
    for(size_t start=0;start<ids.size();) {
      const auto count=std::min<size_t>(limit,ids.size()-start);
      std::vector<Tensor> rows;rows.reserve(count);
      for(size_t i=0;i<count;++i)rows.push_back(result.messages[ids[start+i]].value);
      auto values=copy_rows(rows,target);
      for(size_t i=0;i<count;++i)result.messages[ids[start+i]].value=values[i];
      ++stats["cross_device_copy_groups"];stats["cross_device_rows"]+=count;
      stats["cross_device_bytes"]+=count*bytes;
      stats["max_cross_device_batch"]=std::max<Index>(stats["max_cross_device_batch"],count);
      start+=count;
    }
  }
  return result;
}
} // namespace tide
