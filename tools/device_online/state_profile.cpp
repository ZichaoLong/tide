#include "device_backend.h"
#include "state_profile.h"
#include "content_profile.h"
#include <stdexcept>

namespace tide::device_online {
StateKernelProfile::StateKernelProfile(const ContentProfile& p)
    :nodes(p.graph.nodes),weights(p.model.nodes),regions(p.graph.regions),source_counts(p.graph.source_counts),
     width(p.width),input_count(p.graph.inputs.size()),dtype(p.dtype),all_content(p.all_content),
     sources(p.sources),read(p.read),read_modes(p.read_modes),read_kinds(p.read_kinds),
     decay(p.decay),retention(p.retention),clock_policy(p.clock_policy),config(p.config) {}
StateKernelProfile::StateKernelProfile(const ContentProfile& p,const std::vector<int64_t>& owned,at::Device d)
    :regions(p.graph.regions),width(p.width),input_count(p.graph.inputs.size()),dtype(p.dtype),all_content(true) {
  if(d.type()!=tide::device_online::resident_device_type||d.index()<0||!p.sources.device().is_cpu())
    throw std::invalid_argument("state shard requires deferred CPU profile and explicit NPU");
  int64_t previous=-1;
  for(const auto n:owned) {
    if(n<=previous||n>=int64_t(p.graph.nodes.size()))
      throw std::invalid_argument("state shard nodes must be unique, increasing and in range");
    previous=n;nodes.push_back(p.graph.nodes[n]);weights.push_back(p.model.nodes[n]);
    source_counts.push_back(p.graph.source_counts[n]);
    all_content&=regions[nodes.back().region].read_mode=="content";
  }
  auto ids=at::tensor(owned,at::kLong);
  auto upload=[&](const at::Tensor& x) {
    if(!x.device().is_cpu())throw std::invalid_argument("state shard parameter tables must remain on CPU until placement");
    return x.index_select(0,ids).contiguous().to(d);
  };
  read=upload(p.read);read_modes=upload(p.read_modes);read_kinds=upload(p.read_kinds);
  decay=upload(p.decay);retention=upload(p.retention);clock_policy=upload(p.clock_policy);config=upload(p.config);
  // This table is static source identity metadata, not a replicated parameter
  // bank. Attention consumes its unchanged logical-slot column using physical IDs.
  sources=p.sources.to(d);
}
Continuation state_shard_initial(const Continuation& q,const std::vector<int64_t>& owned) {
  Continuation out;out.batch_size=q.batch_size;out.cut=q.cut;
  int64_t previous=-1;
  for(size_t i=0;i<owned.size();++i) {
    const auto n=owned[i];if(n<=previous)throw std::invalid_argument("invalid state shard initial mapping");previous=n;
    for(int64_t b=0;b<q.batch_size;++b) {
      const auto it=q.states.find({b,n});if(it!=q.states.end())out.states[{b,int64_t(i)}]=it->second;
    }
  }
  return out;
}
} // namespace tide::device_online
