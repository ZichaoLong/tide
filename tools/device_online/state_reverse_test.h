#pragma once
#include "sharded_state_vjp.h"
#include "sharded_full_vjp.h"
#include <limits>
#include "sharded_state.h"
#include "retained_cache_fixture.h"
namespace tide::device_online::test {
inline std::unique_ptr<ContentFlow> reverse_candidate(const Fixture& f,at::Device d,ContentLimits limits,
    FullPlacement full,bool states) {
  if(!states)return std::make_unique<ContentFlow>(f.graph,f.model,f.initial,d,limits,std::move(full));
  auto placement=full;for(auto& n:placement.owners)n=(n+1)%placement.devices.size();
  return std::make_unique<ContentFlow>(f.graph,f.model,f.initial,d,limits,ModelPlacement{full,placement});
}
inline std::vector<std::vector<CacheCotangents>> owner_cache_roots(const ShardedReverseTape& t,int window,int mode) {
  std::vector<std::vector<CacheCotangents>> out;
  for(const auto& owner:t.states) {
    ReverseTape local{};local.attention=owner.attention;local.fiber=owner.fiber;local.source_scales=owner.state.decay;
    GraphCotangents roots;retained_cache_roots(roots,local,window,mode);out.push_back(std::move(roots.cache));
  }
  return out;
}
inline void compare_owner_cache(const ShardedGraphVjp& g,const ShardedReverseTape& t,const RetainedReference& ref,const Fixture& f,bool half) {
  if(!g.state){compare_retained_cache(g.coordinator,t.coordinator,ref,f,half);return;}
  // Assertion-only combined view: no dynamic values are moved or fed back into
  // a candidate. Each tensor is downloaded by the independent comparator.
  ReverseTape view{};GraphVjp gradient{};const auto owners=g.state->gradients();
  for(int kind:{0,1})for(size_t s=0;s<t.states.size();++s) {
    const auto& owner=t.states[s];
    if(kind==0)for(size_t i=0;i<owner.attention.size();++i) {
      auto a=owner.attention[i];for(auto& n:a.nodes)n=owner.global_nodes.at(n);
      view.attention.push_back(a);gradient.cache.push_back(owners[s].cache[i]);
    } else for(size_t i=0;i<owner.fiber.size();++i) {
      auto a=owner.fiber[i];for(auto& n:a.cache.nodes)n=owner.global_nodes.at(n);
      view.fiber.push_back(a);gradient.cache.push_back(owners[s].cache[owner.attention.size()+i]);
    }
  }
  compare_retained_cache(gradient,view,ref,f,half);
}
inline void poison_owner_cache(ShardedReverseTape& t) {
  for(auto& owner:t.states) {
    ReverseTape local{};local.attention=owner.attention;local.fiber=owner.fiber;poison_live_cache(local);
    for(const auto& x:{owner.state.decay,owner.state.retention,owner.read})x.fill_(std::numeric_limits<float>::quiet_NaN());
  }
}
} // namespace tide::device_online::test
