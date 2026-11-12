#include "block_settle_formula.h"
#include "block_kv_message_formula.h"
namespace tide::device_online::block_flow_check {
at::Tensor values(at::IntArrayRef shape,float shift){return block_selection_check::values(shape,shift);}
namespace {
SettleGraph graph(){Graph g;g.nodes={{0},{0},{1}};g.regions={{2},{1}};
  g.edges={{0,2,1},{1,2,1},{0,2,1}};g.inputs={0,1,1};g.outputs={2,2};g.compile();return SettleGraph(g,{0,1},"ports",2);}
}
Fixture::Fixture():graph(block_flow_check::graph()){
  block_message_check::Fixture seed;bindings.resize(2);roles.resize(2);states.resize(2);bound.resize(2);
  input.positions=at::tensor(std::vector<int64_t>{Start,Start+3,Start+10,Start+14},at::kLong).reshape({Batch,Length});
  input.payload=values({Batch,Length,3,3},.4);input.present=at::ones({Batch,Length,3},at::kBool);
  for(int64_t f=0;f<Frames;++f)for(int64_t p=0;p<3;++p)input.payload.reshape({Frames,3,3})[f][p][0]=.15+.05*p;
  input.payload[0][1][1][0]=0;input.payload[0][1][2][0]=0;input.payload[1][0][0].slice(0,1).zero_();
  input.present[0][1][0]=false;input.present[1][1].zero_();
  std::map<std::string,at::Tensor> canonical;
  for(int64_t r=0;r<2;++r){auto& b=bindings[r];b.memory=seed.selection.kv;b.regional=seed.selection.regional;b.read_mode="+";
    b.aggregate_alpha=.2;b.node_owners=r?std::vector<int64_t>{0}:std::vector<int64_t>{0,1};b.input_owners={0,1,1};b.edge_owners={0,1,0};
    b.parameters.message={seed.selection.weights,seed.full};b.parameters.message.read.region.context_owners=at::tensor({r},at::kLong);
    b.parameters.update={values({2,4,2},.2),values({2,4,2},.3),values({2,2,2},.4),values({2,2,2},.5)};
    b.parameters.source={values({2,3,2},.2),values({2,2,3},.3),values({2,3},.4)};
    for(const auto& [role,bank]:block_kv_local_parameters(b.parameters))for(int64_t i=0;i<bank.size(0);++i){
      // Shared canonical owners across regions and Read/Full, distinct equal
      // Full.out owners, and shared gate/up; every canonical owns its storage.
      std::string key=role;if(role=="full.query")key="read.query";if(role=="full.up")key="full.gate";
      key+="."+std::to_string(i);if(!canonical.count(key))canonical[key]=bank[i].clone();
      const auto name="r"+std::to_string(r)+"."+role+"."+std::to_string(i);registry.add(name,canonical.at(key));roles[r][role].push_back(name);
    }
    WeightedRawLayout layout(b.memory,Capacity);const auto count=int64_t(b.node_owners.size());
    for(int64_t row=0;row<Batch*count;++row){State s;const auto first=input.positions[row/count][0].item<int64_t>()*2+r;
      s.last_time=first-1;s.observations=2;s.value=at::zeros({3},at::kFloat);s.slots["times"]=at::tensor({first-4,first-1},at::kLong);
      for(const auto& field:layout.fields())if(field.name!="value"){
        auto shape=field.shape;shape.insert(shape.begin(),2);s.slots[field.name]=values(shape,.3+row*.03+r*.05);
        if(field.name=="weights")s.slots[field.name]=at::tensor({.2f,.3f});}
      states[r].push_back(s);auto initial=s;initial.slots["keys"]=s.slots["keys"]+.13;bound[r].push_back(initial);
    }
    input.initial.push_back(encode_weighted_raw_state(layout,states[r]));input.bound_initial.push_back(encode_weighted_raw_state(layout,bound[r]));
  }
}
std::vector<BlockSettleKVBinding> device_bindings(const Fixture& f,at::Device device){
  auto out=f.bindings;for(auto& b:out){Fields fields;for(const auto& [role,bank]:block_kv_local_parameters(b.parameters))fields[role]=bank.to(device);
    b.parameters=block_kv_local_parameters(fields,b.parameters);
    for(auto* x:{&b.parameters.message.read.region.context_owners,&b.parameters.message.read.region.query_owners})if(x->defined())*x=x->to(device);}
  return out;
}
BlockSettleFlowInput device_input(const Fixture& f,at::Device device){auto x=f.input;x.positions=x.positions.to(device);x.payload=x.payload.to(device);x.present=x.present.to(device);
  for(auto* collection:{&x.initial,&x.bound_initial})for(auto& s:*collection)s=copy_weighted_raw_state(s,device);return x;}
}
