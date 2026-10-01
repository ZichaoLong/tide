#pragma once
#include "full_placement.h"
#include <algorithm>

namespace tide::device_online::test {
inline void full_placement_check(const Model& fixture,at::Device device) {
  Graph g;Model m=fixture;g.nodes.resize(4);m.nodes.resize(4);
  for(auto& n:g.nodes){n.region=0;n.full="tanh";}
  g.edges={{0,1,1},{0,1,2},{1,2,1},{2,3,1},{2,3,2},{3,0,1}};
  for(auto& w:m.nodes){w.weight=m.nodes[0].weight;w.bias=m.nodes[0].bias;}
  std::vector<at::Device> devices{device,at::Device(device.type(),device.index()+1)};
  auto memory=place_full(g,m,devices,"memory"),local=place_full(g,m,devices,"locality");
  auto cut=[&](const auto& p){int64_t n=0;for(const auto& e:g.edges)n+=p.owners[e.source]!=p.owners[e.target];return n;};
  if(cut(memory)!=6||cut(local)!=2||local.owners!=place_full(g,m,devices,"locality").owners)
    throw std::runtime_error("Full locality partition lost deterministic balanced physical-edge cuts");
  for(const auto& p:{memory,local})for(int64_t s=0;s<2;++s)
    if(std::count(p.owners.begin(),p.owners.end(),s)!=2)throw std::runtime_error("Full locality changed balanced node load");
  for(int kind=0;kind<3;++kind) {
    auto invalid=local;
    if(kind==0)invalid.devices[1]=invalid.devices[0];
    else if(kind==1)invalid.owners[0]=-1;else std::fill(invalid.owners.begin(),invalid.owners.end(),0);
    bool refused=false;try{validate_full_placement(invalid,4,device);}catch(const std::invalid_argument&){refused=true;}
    if(!refused)throw std::runtime_error("invalid Full placement accepted");
  }
}
} // namespace tide::device_online::test
