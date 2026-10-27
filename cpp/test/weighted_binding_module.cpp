// A second static-core instance, like a separately loaded resident client.
#include "tide/weighted.h"
extern "C" void tide_test_weighted_install(tide::Model* m,int family) {
  using namespace tide::weighted;
  RegionalConfig r;r.local_score="match";r.selection_width=3;r.alpha_match=0.4;r.alpha_res=0.2;
  tide::SourceClasses sources;sources.edges={2,2};sources.inputs={0,1};
  m->nodes.resize(1);m->regions.resize(1);
  if(family==0){KVConfig c;c.memory_read="qry";configure_node(m->nodes[0],c,r,sources,0.3);}
  else {RecurrentConfig c;c.memory_family=std::vector<std::string>{"affine","delta","gated_delta","ssd"}.at(family-1);
    configure_node(m->nodes[0],c,r,sources,0.3);}
  configure_region(m->regions[0],r);
}
extern "C" void tide_test_weighted_inspect(const tide::Model* m,int family) {
  using namespace tide::weighted;
  auto n=node_binding(m->nodes.at(0));auto r=region_binding(m->regions.at(0));
  const auto name=family==0?"kv":std::vector<std::string>{"affine","delta","gated_delta","ssd"}.at(family-1);
  auto actual=std::visit([](const auto& c)->std::string {
    using C=std::decay_t<decltype(c)>;
    if constexpr(std::is_same_v<C,KVConfig>)return "kv";else return c.memory_family;
  },n.memory);
  if(actual!=name||n.sources.edges!=std::vector<tide::Index>{2,2}||n.sources.inputs!=std::vector<tide::Index>{0,1}
      ||n.aggregate_alpha!=0.3||r.local_score!="match"||!same_configuration(n.regional,r))
    throw std::runtime_error("cross-module installed binding differs");
}
