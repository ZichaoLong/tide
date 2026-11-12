// Independent proposal Full and structured emission formulas. It uses the
// separately constructed Read formula, never candidate tapes or state decode.
#include "block_kv_message_formula.h"
#include <ATen/core/grad_mode.h>
#include <cmath>

namespace tide::device_online::block_message_check {
using namespace block_selection_check;
using at::Tensor;
Formula formula(const Fixture& f){
  at::AutoGradMode enabled(true);Formula out;out.selection=block_selection_check::formula(f.selection);
  out.leaves=out.selection.leaves;out.labels=out.selection.labels;
  std::vector<Fields> raw(Rows),params(Banks);std::vector<Tensor> input(Rows);
  std::map<std::string,Tensor> canonical;
  for(size_t i=0;i<out.labels.size();++i){const auto& label=out.labels[i];
    if(label.kind=="content")input[label.row]=out.leaves[i];
    else if(label.kind=="query"||label.kind=="describe"||label.kind=="context")
      canonical[f.registry.canonical_name(parameter_name(label))]=out.leaves[i];
  }
  for(int64_t row=0;row<Rows;++row)for(const auto& part:f.proposal_layout.fields()){
    auto value=part.name=="value"?f.proposal_states[row].value:f.proposal_states[row].slots.at(part.name);
    auto leaf=value.clone().set_requires_grad(true);raw[row][part.name]=leaf;
    out.leaves.push_back(leaf);out.labels.push_back({"proposal",part.name,row});
  }
  for(const auto& [role,value]:parameter_roles({f.selection.weights,f.full}))if(role.rfind("full.",0)==0){
    for(int64_t bank=0;bank<Banks;++bank){const auto key=f.registry.canonical_name(role+"."+std::to_string(bank));
      if(!canonical.count(key)){auto leaf=value[bank].clone().set_requires_grad(true);canonical[key]=leaf;
        out.leaves.push_back(leaf);out.labels.push_back({"full",role,bank});}
      params[bank][role]=canonical.at(key);
    }
  }
  const auto& c=f.selection.kv;out.presence=at::zeros({Frames,Nodes},at::kBool);
  for(int64_t row=0;row<Rows;++row){
    const auto strength=out.selection.control_roots[row];auto content=at::zeros({c.width},at::kFloat);
    const bool active=strength.detach().item<float>()>0;
    if(active){
      out.presence.reshape({Rows})[row]=true;const auto owner=f.full_owners.reshape({Rows})[row].item<int64_t>();
      const auto& p=params[owner];const auto& s=raw[row];auto u=input[row];
      auto q=at::mv(p.at("full.query"),u/(u.square().mean()+c.epsilon).sqrt());
      const auto time=Clock+(row/Nodes)*3;std::vector<Tensor> logits;
      for(int64_t j=0;j<Cache+1;++j){
        const auto age=time-f.proposal_states[row].slots.at("times")[j].item<int64_t>();
        auto phase=at::scalar_tensor(-float(age)/float(c.position_scale)*float(c.full_frequencies[0][0]),at::kFloat);
        auto key=s.at("keys")[j],co=phase.cos(),si=phase.sin();
        auto rotated=at::stack({key[0]*co-key[1]*si,key[0]*si+key[1]*co});
        logits.push_back((q*rotated).sum()/std::sqrt(2.)-float(age)/float(c.position_scale)*float(c.distance_slopes[0])+s.at("weights")[j].log());
      }
      auto memory=s.at("weights").sum().clamp_max(1)*(at::stack(logits).softmax(0).unsqueeze(1)*s.at("values")).sum(0);
      auto x=u+c.alpha_att*at::mv(p.at("full.out"),memory),norm=x/(x.square().mean()+c.epsilon).sqrt();
      auto gate=at::mv(p.at("full.gate"),norm),up=at::mv(p.at("full.up"),norm);
      content=x+c.alpha_ff*at::mv(p.at("full.down"),gate*gate.sigmoid()*up);
    }
    out.full.push_back(content);out.message_content.push_back(content);
    out.message_strength.push_back(active?strength:at::zeros({},at::kFloat));
    out.projection.push_back(active?strength*content:at::zeros_like(content));
  }
  return out;
}
}
