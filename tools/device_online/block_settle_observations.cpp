#include "block_settle_observations.h"
#include <set>
namespace tide::device_online {
namespace {
std::string region_name(size_t r){return "region."+std::to_string(r)+".";}
void raw_fields(BlockWindow& out,const std::string& prefix,const WeightedRawLayout& layout,const WeightedRawState& raw,const at::Tensor& present){
  const auto rows=raw.floating.size(0),capacity=layout.capacity();const auto valid=at::arange(capacity,raw.integer.options()).unsqueeze(0)<raw.integer.select(1,2).unsqueeze(1);
  out.metadata[prefix+"integer"]=raw.integer;
  for(const auto& f:layout.fields()){auto shape=f.shape;if(f.ragged)shape.insert(shape.begin(),capacity);shape.insert(shape.begin(),rows);
    const auto width=f.width*(f.ragged?capacity:1);
    out.fields[prefix+f.name]={raw.floating.narrow(1,f.offset,width).reshape(shape),present,f.ragged?valid:at::Tensor{}};}
}
}
BlockWindow block_settle_observations(const BlockSettleWindow& w,ResidentToken token){
  const auto& t=*w.tape;BlockWindow out;out.token=token;out.metadata["positions"]=t.positions;
  out.metadata["journal"]=t.dispatcher->journal();out.metadata["dispatch_ledger"]=t.dispatcher->ledger();
  out.metadata["node_present"]=w.boundary.node_present;out.metadata["region_times"]=w.boundary.region_times;
  out.metadata["input_counts"]=w.boundary.input_counts;out.metadata["input_times"]=w.boundary.input_times;
  auto physical=[&](const std::string& prefix,const BlockPhysicalColumns& x){
    out.fields[prefix+"strength"]={x.payload.select(2,0),x.present,{}};
    out.fields[prefix+"content"]={x.payload.slice(2,1),x.present,{}};out.metadata[prefix+"times"]=x.times;};
  physical("edge.",t.edges);physical("output.",t.outputs);
  for(size_t r=0;r<t.regions.size();++r){const auto& stage=t.regions[r];const auto& x=*stage.local;const auto p=region_name(r);
    const auto candidates=x.present.reshape({-1});const auto frames=x.messages.selection.frames,nodes=x.messages.selection.nodes;
    out.metadata[p+"nodes"]=stage.nodes;out.metadata[p+"physical_atoms"]=stage.physical_atoms;out.metadata[p+"source_mapping"]=stage.source_mapping;
    out.metadata[p+"candidate_times"]=x.messages.selection.node_times;out.metadata[p+"fiber_lengths"]=x.aggregate.lengths;
    out.fields[p+"received"]={x.aggregate.output,candidates,{}};
    out.fields[p+"controls"]={x.messages.selection.controls,x.present.reshape({frames,nodes}),{}};
    out.fields[p+"Full"]={x.messages.full.output.reshape({frames,nodes,-1}),x.messages.messages.present,{}};
    out.fields[p+"message.strength"]={x.messages.messages.strength,x.messages.messages.present,{}};
    out.fields[p+"message.content"]={x.messages.messages.content,x.messages.messages.present,{}};
    for(int channel=0;channel<2;++channel){const auto suffix=channel?"long":"short";
      out.fields[p+"Mem."+suffix]={x.messages.selection.query.memory.select(1,channel),candidates,{}};
      out.fields[p+"total."+suffix]={x.messages.selection.query.total.select(1,channel),candidates,{}};}
    for(const auto& [name,value]:x.messages.selection.describe.fields)out.fields[p+"Describe."+name]={value,candidates,{}};
    raw_fields(out,p+"old.",x.sequence.layout,x.sequence.old,candidates);raw_fields(out,p+"proposal.",x.sequence.layout,x.sequence.proposal,candidates);
    raw_fields(out,p+"final.",x.sequence.layout,x.sequence.final,w.boundary.node_present.index_select(1,stage.nodes).reshape({-1}));
  }
  return out;
}
BlockSettleFlowRoots block_settle_roots(const BlockWindow& view,const BlockRoots& given,const BlockSettleFlowTape& t){
  if(given.token.session!=view.token.session||given.token.index!=view.token.index||given.token.generation!=view.token.generation)
    throw std::invalid_argument("block root names a different owner/window/generation");
  for(const auto& [name,r]:given.fields){const auto found=view.fields.find(name);if(found==view.fields.end())throw std::invalid_argument("unknown block observation root: "+name);
    const auto& f=found->second;if(!r.values.defined()||!r.connected.defined()||r.values.device()!=f.values.device()||r.connected.device()!=f.present.device()||
        r.values.scalar_type()!=at::kFloat||r.connected.scalar_type()!=at::kBool||r.values.sizes()!=f.values.sizes()||r.connected.sizes()!=f.present.sizes()||r.values.requires_grad()||r.connected.requires_grad())
      throw std::invalid_argument("block observation root layout differs: "+name);}
  auto root=[&](const std::string& name){const auto found=given.fields.find(name);if(found==given.fields.end())return WeightedRowAdjoint{};
    const auto& r=found->second;return WeightedRowAdjoint{r.values.contiguous(),at::logical_and(r.connected,view.fields.at(name).present).contiguous()};};
  BlockSettleFlowRoots out;out.edges={root("edge.strength"),root("edge.content")};out.outputs={root("output.strength"),root("output.content")};
  for(size_t r=0;r<t.regions.size();++r){const auto& x=*t.regions[r].local;const auto p=region_name(r);BlockKVLocalRoots a;
    a.received=root(p+"received");a.messages.controls=root(p+"controls");a.messages.full=root(p+"Full");
    a.messages.message_strength=root(p+"message.strength");a.messages.message_content=root(p+"message.content");
    const auto rows=x.sequence.old.floating.size(0),ds=x.sequence.config.summary_width;
    auto channels=[&](const std::string& prefix,bool memory){
      WeightedRowAdjoint result;for(int channel=0;channel<2;++channel){const auto bar=root(p+prefix+(channel?"long":"short"));if(!bar.value.defined())continue;
        if(!result.value.defined()){result.value=at::zeros(memory?std::vector<int64_t>{rows,2,ds}:std::vector<int64_t>{rows,2},bar.value.options());result.connected=at::zeros({rows,2},bar.connected.options());}
        result.value.select(1,channel).copy_(bar.value);result.connected.select(1,channel).copy_(bar.connected);}return result;};
    a.messages.read.memory=channels("Mem.",true);a.messages.read.total=channels("total.",false);
    for(const auto& [name,value]:x.messages.selection.describe.fields){auto bar=root(p+"Describe."+name);if(bar.value.defined())a.messages.read.describe[name]=bar;}
    auto raw=[&](const std::string& prefix,const WeightedRawState& state){WeightedRowAdjoint result;int field=0;
      for(const auto& f:x.sequence.layout.fields()){const auto bar=root(p+prefix+f.name);if(bar.value.defined()){
          if(!result.value.defined()){result.value=at::zeros_like(state.floating);result.connected=at::zeros({state.floating.size(0),int64_t(x.sequence.layout.fields().size())},bar.connected.options());}
          const auto width=f.width*(f.ragged?x.sequence.layout.capacity():1);
          result.value.narrow(1,f.offset,width).copy_(bar.value.reshape({state.floating.size(0),width}));result.connected.select(1,field).copy_(bar.connected);}++field;}return result;};
    auto old=raw("old.",x.sequence.old),proposal=raw("proposal.",x.sequence.proposal),final=raw("final.",x.sequence.final);
    a.raw={old.value,old.connected,proposal.value,proposal.connected,final.value,final.connected};out.regions.push_back(std::move(a));}
  return out;
}
BlockGradients block_settle_gradients(const BlockSettleAdjoints& actual,const BlockSettleFlowTape& tape){
  BlockGradients out;out.parameters.offsets=actual.parameters.offsets;out.parameters.values=actual.parameters.values;out.parameters.connected=actual.parameters.connected;
  for(const auto& owner:actual.parameters.owners){out.parameters.names.push_back(owner.canonical);out.parameters.aliases.push_back(owner.aliases);}
  for(const auto& g:actual.windows)out.inputs.push_back({g.inputs,g.input_connected});
  for(size_t r=0;r<tape.regions.size();++r){const auto& t=tape.regions[r].local->sequence;const auto& g=actual.windows.front().regions[r].sequence;int index=0;
    for(const auto& field:t.layout.fields()){const auto rows=g.initial.size(0);auto shape=field.shape;if(field.ragged)shape.insert(shape.begin(),t.layout.capacity());shape.insert(shape.begin(),rows);
      const auto width=field.width*(field.ragged?t.layout.capacity():1);
      const auto name=region_name(r)+field.name;out.initial[name]={g.initial.narrow(1,field.offset,width).reshape(shape),g.initial_connected.select(1,index)};
      out.bound_initial[name]={actual.bound_initial[r].value.narrow(1,field.offset,width).reshape(shape),actual.bound_initial[r].connected.select(1,index)};++index;}}
  return out;
}
}
