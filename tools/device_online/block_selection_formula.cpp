// Literal KV §2–3 and allocation §2–4 formulas. No candidate execution,
// packing, resident operator, general RegionKernel or reference recurrence.
#include "block_selection_formula.h"
#include <ATen/core/grad_mode.h>
#include <cmath>

namespace tide::device_online::block_selection_check {
using at::Tensor;
namespace {
Tensor normalized(const Tensor& x,double epsilon){return x/(x.square().mean()+epsilon).sqrt();}
Tensor head(const Fields& p,const std::string& name,const Tensor& x){
  auto h=at::mv(p.at(name+".matrix"),x)+p.at(name+".hidden_bias");
  return at::mv(p.at(name+".projection"),h*h.sigmoid())+p.at(name+".bias");
}
Tensor two_candidate_quad(const Tensor& scores,const Tensor& caps,const Tensor& bound,double tau){
  const auto count=scores.numel();auto zero=at::zeros({count},scores.options());
  // Empty/zero/full-cap branches are distinct exact maps, including their roots.
  if(!count||bound.item<float>()==0||caps.sum().item<float>()==0)return zero;
  if(bound.item<float>()>=caps.sum().item<float>())return caps;
  const auto connected_zero=(scores.sum()+caps.sum()+bound)*0;
  if(count==1)return bound.reshape({1})+connected_zero;
  if(count!=2)throw std::invalid_argument("formula is limited to two candidates");
  Tensor first;
  if(caps[0].item<float>()==0)first=at::zeros({},scores.options());
  else if(caps[1].item<float>()==0)first=bound;
  else {
    auto unconstrained=(bound+(scores[0]-scores[1])/tau)/2;
    auto lower=(bound-caps[1]).clamp_min(0),upper=at::minimum(caps[0],bound);
    first=unconstrained.item<float>()<lower.item<float>()?lower:
      unconstrained.item<float>()>upper.item<float>()?upper:unconstrained;
  }
  return at::stack({first,bound-first})+connected_zero;
}
}
Formula formula(const Fixture& f){
  at::AutoGradMode enabled(true);Formula out;
  auto leaf=[&](const Tensor& value,std::string kind,std::string name,int64_t row){
    auto x=value.clone().set_requires_grad(true);out.leaves.push_back(x);out.labels.push_back({kind,name,row});return x;};
  std::vector<Fields> raw(Rows),params(Banks),region(Banks);
  std::vector<Tensor> mass,content;
  for(int64_t row=0;row<Rows;++row){
    for(const auto& part:f.raw_layout.fields())raw[row][part.name]=leaf(
      part.name=="value"?f.states[row].value:f.states[row].slots.at(part.name),"raw",part.name,row);
    mass.push_back(leaf(f.input.mass.reshape({Rows})[row],"mass","",row));
    content.push_back(leaf(f.input.content.reshape({Rows,2})[row],"content","",row));
  }
  const std::vector<std::string> qnames{"query","key","value"},hnames{"matrix","hidden_bias","projection","bias"};
  auto qroles=roles(f.weights.query),hroles=roles(f.weights.region.context);
  for(int64_t bank=0;bank<Banks;++bank){
    for(int j=0;j<3;++j)params[bank][qnames[j]]=leaf(qroles[j][bank],"query",qnames[j],bank);
    for(const auto& [name,value]:f.weights.describe)params[bank][name]=leaf(value[bank],"describe",name,bank);
    for(int j=0;j<4;++j)region[bank]["context."+hnames[j]]=leaf(hroles[j][bank],"context",hnames[j],bank);
  }
  std::vector<Tensor> memories,totals;
  for(int64_t row=0;row<Rows;++row){
    Fields desc{{"w",at::zeros({1},at::kFloat)},{"mu",at::zeros({1},at::kFloat)},
      {"score",at::zeros({1},at::kFloat)},{"evidence",at::zeros({2},at::kFloat)}};
    std::vector<Tensor> memory{at::zeros({2},at::kFloat),at::zeros({2},at::kFloat)};
    std::vector<Tensor> total{at::zeros({},at::kFloat),at::zeros({},at::kFloat)};
    if(f.input.present.reshape({Rows})[row].item<bool>()){
      const auto owner=f.input.node_owners.reshape({Rows})[row].item<int64_t>();const auto& p=params[owner];
      auto cur=normalized(content[row],f.kv.epsilon),q=at::mv(p.at("query"),cur);
      auto times=f.states[row].slots.at("times");const auto time=f.input.times.reshape({Rows})[row].item<int64_t>();
      std::vector<Tensor> logits,features;std::vector<int64_t> exact_age;
      for(int64_t j=0;j<times.numel();++j){
        const auto age=time-times[j].item<int64_t>();exact_age.push_back(age);
        auto key=at::mv(p.at("key"),raw[row].at("keys")[j]);
        auto phase=at::scalar_tensor(-float(age)/float(f.kv.position_scale)*float(f.kv.read_frequencies[0]),at::kFloat);
        auto cosine=phase.cos(),sine=phase.sin();
        auto rotated=at::stack({key[0]*cosine-key[1]*sine,key[0]*sine+key[1]*cosine});
        logits.push_back((q*rotated).sum()/std::sqrt(2.));features.push_back(at::mv(p.at("value"),raw[row].at("values")[j]));
      }
      for(int channel=0;channel<2;++channel){
        auto ages=at::tensor(exact_age,at::kLong).to(at::kFloat),writes=raw[row].at("weights");
        auto weight=channel?writes:writes*at::exp(-float(f.kv.beta_short)*ages);total[channel]=weight.sum();
        auto probability=(at::stack(logits)+weight.log()).softmax(0);
        memory[channel]=total[channel].clamp_max(1)*(probability.unsqueeze(1)*at::stack(features)).sum(0);
      }
      auto current=at::mv(p.at("summary.matrix"),cur);
      auto feature=at::cat({cur,mass[row].log1p().reshape({1}),memory[0],memory[1],
        total[0].log1p().reshape({1}),total[1].log1p().reshape({1}),current*memory[0],current*memory[1]});
      desc["w"]=mass[row].clamp_max(1).reshape({1});desc["mu"]=(f.regional.gamma_max*desc["w"]).clamp_max(1);
      desc["score"]=head(p,"score",feature);
      desc["evidence"]=normalized(at::mv(p.at("evidence.matrix"),feature)+p.at("evidence.bias"),f.kv.epsilon);
    }
    out.descriptions.push_back(desc);memories.push_back(at::stack(memory));totals.push_back(at::stack(total));
  }
  std::vector<Tensor> controls(Rows);for(auto& x:controls)x=at::zeros({},at::kFloat);
  for(int64_t frame=0;frame<Frames;++frame)for(int64_t rid=0;rid<2;++rid){
    std::vector<Tensor> ws,mus,ss,es;std::vector<int64_t> rows;
    for(int64_t node=0;node<Nodes;++node)if((node<2?0:1)==rid&&f.input.present[frame][node].item<bool>()){
      const auto row=frame*Nodes+node;const auto& d=out.descriptions[row];rows.push_back(row);
      ws.push_back(d.at("w").reshape({}));mus.push_back(d.at("mu").reshape({}));ss.push_back(d.at("score").reshape({}));es.push_back(d.at("evidence"));
    }
    if(rows.empty())continue;auto w=at::stack(ws),mu=at::stack(mus),sumw=w.sum(),summu=mu.sum();
    auto context=at::cat({(w.unsqueeze(1)*at::stack(es)).sum(0)/sumw.clamp_min(1),sumw.log1p().reshape({1}),summu.log1p().reshape({1})});
    const auto owner=f.weights.region.context_owners[rid].item<int64_t>();auto rho=head(region[owner],"context",context).sigmoid().reshape({});
    auto lower=sumw.clamp_max(1),upper=summu.clamp_max(*f.regional.k),bound=lower+rho*(upper-lower);
    auto c=two_candidate_quad(at::stack(ss),mu,bound,*f.regional.allocation_temperature);
    for(size_t j=0;j<rows.size();++j)controls[rows[j]]=c[j];
  }
  out.control_roots=controls;out.memory=at::stack(memories);out.total=at::stack(totals);
  out.controls=at::stack(controls).reshape({Frames,Nodes});return out;
}
}
