#include "block_kv_message_formula.h"
#include <cmath>

namespace tide::device_online::block_message_check {
using namespace block_selection_check;
Fields parameter_roles(const BlockKVMessageParameters& w){
  Fields out{{"read.query",w.read.query.query},{"read.key",w.read.query.key},{"read.value",w.read.query.value},
    {"full.query",w.full.query},{"full.out",w.full.residual.out},{"full.gate",w.full.residual.gate},
    {"full.up",w.full.residual.up},{"full.down",w.full.residual.down}};
  for(const auto& [name,t]:w.read.describe)out["describe."+name]=t;
  auto h=w.read.region.context;out.insert({{"region.context.matrix",h.matrix},{"region.context.hidden_bias",h.hidden_bias},
    {"region.context.projection",h.projection},{"region.context.bias",h.bias}});return out;
}
Fixture::Fixture():proposal_layout(selection.kv,Cache+1){
  selection.kv.full_frequencies={{.23}};selection.kv.distance_slopes={.07};
  selection.kv.alpha_att=.3;selection.kv.alpha_ff=.4;selection.kv.validate();
  selection.input.content[1][1].zero_(); // A selected zero-valued G remains a message.
  full_owners=selection.input.node_owners.clone();
  for(int64_t row=0;row<Rows;++row)if(selection.input.present.reshape({Rows})[row].item<bool>())
    full_owners.reshape({Rows})[row]=1-selection.input.node_owners.reshape({Rows})[row].item<int64_t>();
  for(int64_t row=0;row<Rows;++row){
    const auto time=Clock+row/Nodes*3;State s;s.last_time=time;s.observations=Cache+1;
    s.slots["times"]=at::tensor({time-4,time-1,time},at::kLong);
    for(const auto& part:proposal_layout.fields()){
      auto shape=part.shape;if(part.ragged)shape.insert(shape.begin(),Cache+1);
      auto x=values(shape,.8+row*.03);if(part.name=="weights")x=at::tensor({.2f,.35f,.1f});
      if(row==4&&part.name=="values")x.zero_();
      if(part.name=="value")s.value=at::zeros_like(x);else s.slots[part.name]=x;
    }proposal_states.push_back(std::move(s));
  }
  proposal=encode_weighted_raw_state(proposal_layout,proposal_states);
  for(int64_t row=0;row<Rows;++row)if(!selection.input.present.reshape({Rows})[row].item<bool>()||row==7){
    proposal.floating[row].fill_(NAN);proposal.integer[row].fill_(-99);full_owners.reshape({Rows})[row].fill_(-99);
  }
  auto gate=values({Banks,selection.kv.ffn_width,2},.7);
  full={selection.weights.query.query,{selection.weights.query.query.clone(),gate,gate,
    values({Banks,2,selection.kv.ffn_width},1.1)}};
  // Full.query really aliases Read.query; Full.out has equal values but a
  // distinct owner. Full.gate/up share another physical parameter.
  std::map<std::pair<const void*,int64_t>,at::Tensor> identities;
  for(const auto& [name,t]:parameter_roles({selection.weights,full})){
    for(int64_t bank=0;bank<t.size(0);++bank){
      const auto key=std::make_pair(static_cast<const void*>(t.unsafeGetTensorImpl()),bank);
      if(!identities.count(key))identities[key]=t[bank];const auto path=name+"."+std::to_string(bank);
      registry.add(path,identities.at(key));roles[name].push_back(path);
    }
  }
}
BlockKVMessageInput device_input(const Fixture& f,at::Device device){
  const auto& x=f.selection.input;
  return {{copy_weighted_raw_state(x.read,device),x.mass.to(device),x.content.to(device),x.present.to(device),
    x.times.to(device),x.node_owners.to(device)},copy_weighted_raw_state(f.proposal,device),f.full_owners.to(device)};
}
BlockKVMessageParameters device_parameters(const Fixture& f,at::Device device){
  Fields fields;std::map<const void*,at::Tensor> identity;
  for(const auto& [name,value]:parameter_roles({f.selection.weights,f.full})){
    const auto* key=value.unsafeGetTensorImpl();if(!identity.count(key))identity[key]=value.to(device);fields[name]=identity.at(key);
  }
  BlockKVMessageParameters w;w.read.query={fields.at("read.query"),fields.at("read.key"),fields.at("read.value")};
  for(const auto& [name,_]:f.selection.weights.describe)w.read.describe[name]=fields.at("describe."+name);
  w.read.region.context={fields.at("region.context.matrix"),fields.at("region.context.hidden_bias"),
    fields.at("region.context.projection"),fields.at("region.context.bias")};
  w.read.region.context_owners=f.selection.weights.region.context_owners.to(device);
  w.full={fields.at("full.query"),{fields.at("full.out"),fields.at("full.gate"),fields.at("full.up"),fields.at("full.down")}};return w;
}
std::string parameter_name(const FormulaLabel& label){
  const auto prefix=label.kind=="query"?"read.":label.kind=="describe"?"describe.":label.kind=="context"?"region.context.":"";
  return prefix+label.name+"."+std::to_string(label.row);
}
}
