#pragma once
#include "tide/resident_blocks.h"
#include <iostream>
namespace tide::device_online::block_public_check {
inline void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
inline void close(const Tensor& actual,const Tensor& expected,const std::string& name){
  if(actual.sizes()!=expected.sizes()||!at::allclose(actual.cpu(),expected.cpu(),8e-5,8e-6)){
    std::cerr<<name<<" actual="<<actual.cpu()<<" expected="<<expected.cpu()<<'\n';throw std::runtime_error("block public numerical difference: "+name);}}
inline void continuation(const Continuation& actual,const Continuation& expected){
  require(actual.identity==expected.identity&&actual.batch_size==expected.batch_size&&actual.cut==expected.cut&&actual.pending.empty()&&expected.pending.empty(),"public continuation cut differs");
  require(actual.ledger==expected.ledger&&actual.history.size()==expected.history.size()&&actual.states.size()==expected.states.size(),"public continuation presence/history/ledger differs");
  for(const auto& [key,h]:expected.history)require(actual.history.at(key).last_time==h.last_time,"public exact regional history differs");
  for(const auto& [key,state]:expected.states){const auto& x=actual.states.at(key);
    require(x.last_time==state.last_time&&x.observations==state.observations&&x.slots.size()==state.slots.size(),"public exact node state metadata differs");close(x.value,state.value,"raw value");
    for(const auto& [name,y]:state.slots)if(y.scalar_type()==at::kLong)require(at::equal(x.slots.at(name),y),"public exact raw clock differs");else close(x.slots.at(name),y,"raw "+name);}
}
inline void exact(const BlockCheckpoint& a,const BlockCheckpoint& b){
  require(a.identity==b.identity&&a.generation==b.generation&&a.next_token==b.next_token&&a.aliases==b.aliases&&a.schema==b.schema,"public restored identity differs");
  for(const auto& [name,x]:a.parameters)require(at::equal(x,b.parameters.at(name)),"public restored parameter differs");
  const std::vector<Tensor> x{a.last_positions,a.node_present,a.region_times,a.input_counts,a.input_times,a.slots.values,a.slots.first,a.slots.second,a.slots.maximum,a.slots.steps,a.slots.corrections},
    y{b.last_positions,b.node_present,b.region_times,b.input_counts,b.input_times,b.slots.values,b.slots.first,b.slots.second,b.slots.maximum,b.slots.steps,b.slots.corrections};
  for(size_t j=0;j<x.size();++j)require(at::equal(x[j],y[j]),"public restored slots/boundary differs");
  require(a.state.size()==b.state.size()&&a.bound_initial.size()==b.bound_initial.size(),"public restored raw groups differ");
  for(size_t r=0;r<a.state.size();++r)for(const auto& pair:std::vector<std::pair<Tensor,Tensor>>{{a.state[r].floating,b.state[r].floating},{a.state[r].integer,b.state[r].integer},
      {a.bound_initial[r].floating,b.bound_initial[r].floating},{a.bound_initial[r].integer,b.bound_initial[r].integer}})
    require(at::equal(pair.first,pair.second),"public restored raw buffers differ");
}
}
