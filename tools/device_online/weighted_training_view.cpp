#include "weighted_training_view.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online {
WeightedTrainingWindow weighted_training_view(const WeightedWindowSnapshot& s,ResidentToken token){
  if(at::GradMode::is_enabled())throw std::invalid_argument("weighted training view requires no-grad");s.validate();
  WeightedTrainingWindow out;out.token=token;out.start=s.start;out.stop=s.boundary.cut;
  const auto stages=s.limits.stages,samples=s.samples;const auto& j=*s.journal;
  auto stage=at::arange(stages,s.stages.options()).lt(s.stages).reshape({stages,1});
  out.metadata["stages"]=s.stages;out.metadata["counts"]=j.field("counts");
  for(size_t r=0;r<s.history.size();++r){const auto prefix="region."+std::to_string(r)+".";
    out.metadata[prefix+"history"]=s.history[r];out.metadata[prefix+"times"]=j.field(prefix+"times");
    out.metadata[prefix+"present"]=j.field(prefix+"present")&stage;}
  auto field=[&](const std::string& name,const at::Tensor& x,const at::Tensor& present,const at::Tensor& valid=at::Tensor{}){
    if(!out.fields.emplace(name,WeightedTrainingField{x,present,valid}).second)throw std::logic_error("duplicate weighted training field");};
  auto raw=[&](const std::string& prefix,const WeightedRawLayout& layout,const WeightedRawState& raw,const at::Tensor& present){
    out.metadata[prefix+"integer"]=raw.integer;
    auto length=raw.integer.select(-1,2);auto slots=at::arange(layout.capacity(),s.stages.options());
    for(const auto& f:layout.fields()){
      auto shape=raw.floating.sizes().vec();shape.pop_back();if(f.ragged)shape.push_back(layout.capacity());shape.insert(shape.end(),f.shape.begin(),f.shape.end());
      auto x=raw.floating.narrow(-1,f.offset,f.width*(f.ragged?layout.capacity():1)).reshape(shape);
      auto valid=f.ragged?slots.lt(length.unsqueeze(-1))&present.unsqueeze(-1):at::Tensor{};
      field(prefix+f.name,x,present,valid);
    }
  };
  for(size_t v=0;v<s.layouts.size();++v){const auto prefix="node."+std::to_string(v)+".";
    auto present=j.field(prefix+"present")&stage;auto active=j.field(prefix+"active")&stage;
    out.metadata[prefix+"times"]=j.field(prefix+"times");
    field(prefix+"content",j.field(prefix+"content"),present);
    field(prefix+"control",j.field(prefix+"control").unsqueeze(-1),present);
    field(prefix+"full",j.field(prefix+"full"),active);
    for(const auto& name:s.descriptors[v])field(prefix+"describe."+name,j.field(prefix+"describe."+name),present);
    raw(prefix+"old.",s.layouts[v],{j.field(prefix+"old.float"),j.field(prefix+"old.int")},present);
    raw(prefix+"proposal.",s.layouts[v],{j.field(prefix+"proposal.float"),j.field(prefix+"proposal.int")},present);
    raw(prefix+"final.",s.layouts[v],s.final[v],s.present[v].ne(0));
  }
  for(const std::string group:{"message","fiber"}){
    const auto key=group=="message"?"messages":"fibers";
    field(key,j.field(group+".values"),j.field(group+".present")&stage);out.metadata[std::string(key)+".coordinates"]=j.field(group+".meta");
  }
  field("pending",s.pending.values,s.pending.valid);out.metadata["pending.coordinates"]=s.pending.coordinates;
  if(!s.graph.outputs.empty()){
    std::vector<at::Tensor> values,present,coordinates;auto batches=at::arange(samples,s.stages.options()).reshape({1,samples}).expand({stages,samples});
    for(size_t port=0;port<s.graph.outputs.size();++port){const auto prefix="node."+std::to_string(s.graph.outputs[port])+".";
      values.push_back(j.field(prefix+"payload"));
      present.push_back(j.field(prefix+"active")&stage);
      coordinates.push_back(at::stack({batches,j.field(prefix+"times"),at::full_like(batches,port)},-1));
    }
    field("outputs",at::stack(values,2),at::stack(present,2));out.metadata["outputs.coordinates"]=at::stack(coordinates,2);
  }
  return out;
}
void validate_weighted_cotangents(const WeightedTrainingWindow& window,const WeightedCotangents& roots){
  if(at::GradMode::is_enabled()||roots.token.session!=window.token.session||roots.token.index!=window.token.index||roots.token.generation!=window.token.generation)
    throw std::invalid_argument("stale, foreign or out-of-order weighted window token");
  for(const auto& [name,r]:roots.fields){const auto found=window.fields.find(name);if(found==window.fields.end())throw std::invalid_argument("unknown weighted root field: "+name);
    const auto& field=found->second;
    if(r.values.defined()!=r.connected.defined())throw std::invalid_argument("incomplete weighted root pair");if(!r.values.defined())continue;
    for(const auto& pair:std::vector<std::pair<at::Tensor,at::Tensor>>{{r.values,field.values},{r.connected,field.present}})
      if(pair.first.device()!=pair.second.device()||pair.first.scalar_type()!=(pair.second.is_floating_point()?at::kFloat:pair.second.scalar_type())||pair.first.sizes()!=pair.second.sizes()||pair.first.requires_grad()||!pair.first.is_contiguous())
        throw std::invalid_argument("weighted root shape/device/dtype differs: "+name);
    if((r.connected&field.present.logical_not()).any().item<bool>())throw std::invalid_argument("weighted root connects an absent value: "+name);
    auto connected=r.connected;while(connected.dim()<r.values.dim())connected=connected.unsqueeze(-1);
    if(!at::isfinite(at::where(connected,r.values,at::zeros_like(r.values))).all().item<bool>())
      throw std::invalid_argument("weighted connected root is nonfinite: "+name);
    if(field.valid.defined()){
      auto invalid=field.valid.logical_not()&r.connected.unsqueeze(-1);
      while(invalid.dim()<r.values.dim())invalid=invalid.unsqueeze(-1);
      if(at::where(invalid,r.values,at::zeros_like(r.values)).ne(0).any().item<bool>())
        throw std::invalid_argument("weighted root has nonzero/nonfinite ragged padding: "+name);
    }
  }
}
}
