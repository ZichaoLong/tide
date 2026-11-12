#include "block_selection_formula.h"
#include <cmath>

namespace tide::device_online::block_selection_check {
using at::Tensor;
Tensor values(at::IntArrayRef shape,float shift){
  int64_t size=1;for(auto n:shape)size*=n;
  return (at::arange(size,at::kFloat)*.13+shift).sin().reshape(shape)*.2;
}
std::vector<Tensor> roles(const WeightedHeadParameters& p){return {p.matrix,p.hidden_bias,p.projection,p.bias};}
std::vector<Tensor> roles(const WeightedKVQueryParameters& p){return {p.query,p.key,p.value};}
namespace {
weighted::KVConfig kv_config(const std::string& read){weighted::KVConfig c;c.width=2;c.summary_width=2;c.memory_read=read;
  c.epsilon=.13;c.beta_short=.2;if(read=="qry")c.read_frequencies={.7};c.position_scale=3;c.validate();return c;}
weighted::RegionalConfig region_config(){weighted::RegionalConfig c("regional_total");c.hidden_width=3;
  c.gamma_max=2;c.allocation_temperature=.8;c.validate();return c;}
}
Fixture::Fixture(const std::string& memory_read):kv(kv_config(memory_read)),regional(region_config()),
    plan(make_settle_block_plan({0,0,1},{0,0},{},{0,1,2},Clock,1,3,1<<20)),raw_layout(kv,Cache){
  input.mass=at::tensor({.2f,.3f,.15f,0.f,.8f,0.f,.35f,0.f,1.2f,0.f,0.f,0.f}).reshape({Frames,Nodes});
  input.content=values({Frames,Nodes,kv.width},.3);
  input.present=at::tensor({1,1,1,0,1,0,1,1,1,0,0,0},at::kInt).to(at::kBool).reshape({Frames,Nodes});
  input.times=(at::arange(Frames,at::kLong)*3+Clock).unsqueeze(1).repeat({1,Nodes});
  input.node_owners=at::tensor(std::vector<int64_t>{0,1,0},at::kLong).repeat({Frames,1});
  for(int64_t row=0;row<Rows;++row){
    const auto time=input.times.reshape({Rows})[row].item<int64_t>();State s;
    s.last_time=time-1;s.observations=Cache;s.slots["times"]=at::tensor({time-4,time-1},at::kLong);
    for(const auto& part:raw_layout.fields()){
      auto shape=part.shape;if(part.ragged)shape.insert(shape.begin(),Cache);
      auto data=values(shape,.2+row*.07);if(part.name=="weights")data=at::tensor({.2f,.35f});
      if(part.name=="value")s.value=at::zeros_like(data);else s.slots[part.name]=data;
    }states.push_back(std::move(s));
  }
  input.read=encode_weighted_raw_state(raw_layout,states);
  for(int64_t row=0;row<Rows;++row)if(!input.present.reshape({Rows})[row].item<bool>()){
    input.mass.reshape({Rows})[row].fill_(NAN);input.content.reshape({Rows,kv.width})[row].fill_(NAN);
    input.read.floating[row].fill_(NAN);input.read.integer[row].fill_(-99);
    input.times.reshape({Rows})[row].fill_(-99);input.node_owners.reshape({Rows})[row].fill_(-99);
  }
  if(kv.memory_read=="qry")weights.query={values({Banks,2,2},.2),values({Banks,2,2},.4),values({Banks,2,2},.6)};
  int role=0;for(const auto& [name,shape]:weighted_describe_parameter_shapes(Banks,2,2,regional))
    weights.describe[name]=values(shape,.15+.019*role++);
  weights.region.context={values({Banks,3,4},.3),values({Banks,3},.4),values({Banks,1,3},.5),values({Banks,1},.6)};
  weights.region.context_owners=at::tensor(std::vector<int64_t>{0,1},at::kLong);
}
}
