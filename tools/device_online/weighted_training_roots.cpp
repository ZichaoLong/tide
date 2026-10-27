#include "weighted_training_view.h"
namespace tide::device_online {
void seed_weighted_cotangents(const WeightedWindowSnapshot& s,const WeightedTrainingWindow& window,
    const WeightedCotangents& supplied,WeightedWindowRoots& roots){
  validate_weighted_cotangents(window,supplied);s.validate();
  auto pair=[&](const std::string& name){const auto& shape=window.fields.at(name);const auto found=supplied.fields.find(name);
    if(found==supplied.fields.end()||!found->second.values.defined())return WeightedCotangent{at::zeros_like(shape.values),at::zeros_like(shape.present)};
    const auto& r=found->second;auto mask=r.connected;while(mask.dim()<r.values.dim())mask=mask.unsqueeze(-1);
    return WeightedCotangent{at::where(mask,r.values,at::zeros_like(r.values)),r.connected};};
  auto row=[&](const std::string& name,WeightedRowAdjoint& target){auto r=pair(name);target.value.copy_(r.values);target.connected.copy_(r.connected);};
  auto raw=[&](const std::string& prefix,const WeightedRawLayout& layout,WeightedRawAdjoint& target){
    std::vector<at::Tensor> values,flags;
    for(const auto& f:layout.fields()){auto r=pair(prefix+f.name);auto shape=r.connected.sizes().vec();shape.push_back(f.width*(f.ragged?layout.capacity():1));
      values.push_back(r.values.reshape(shape));flags.push_back(r.connected);}
    target.value.copy_(at::cat(values,-1));target.connected.copy_(at::stack(flags,-1));
  };
  for(size_t v=0;v<roots.nodes.size();++v){auto& r=roots.nodes[v];const auto prefix="node."+std::to_string(v)+".";
    row(prefix+"content",r.content);row(prefix+"control",r.control);row(prefix+"full",r.full);
    for(auto& [name,value]:r.describe)row(prefix+"describe."+name,value);
    raw(prefix+"old.",s.layouts[v],r.old);raw(prefix+"proposal.",s.layouts[v],r.proposal);raw(prefix+"final.",s.layouts[v],r.final);
  }
  row("messages",roots.messages);row("fibers",roots.fibers);row("pending",roots.pending);
  if(!s.graph.outputs.empty()){
    auto outputs=pair("outputs");
    // Static port bindings only. Each physical port contributes once, even
    // when several ports broadcast the same mathematical (c,G) pair.
    for(size_t port=0;port<s.graph.outputs.size();++port){auto& node=roots.nodes[s.graph.outputs[port]];auto value=outputs.values.select(2,port),connected=outputs.connected.select(2,port);
      node.control.value.add_(value.narrow(-1,0,1));node.full.value.add_(value.narrow(-1,1,value.size(-1)-1));
      node.control.connected.logical_or_(connected);node.full.connected.logical_or_(connected);
    }
  }
}
}
