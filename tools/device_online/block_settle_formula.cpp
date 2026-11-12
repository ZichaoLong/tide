#include "block_settle_formula.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
namespace tide::device_online::block_flow_check {
using at::Tensor;
namespace {
Tensor norm(const Tensor& x,double epsilon){return x/(x.square().mean()+epsilon).sqrt();}
Tensor head(const Fields& p,const std::string& name,const Tensor& x){auto h=at::mv(p.at(name+"matrix"),x)+p.at(name+"hidden_bias");
  return at::mv(p.at(name+"projection"),h*h.sigmoid())+p.at(name+"bias");}
Tensor rotate(const Tensor& k,int64_t age,double frequency,double scale){auto phase=at::scalar_tensor(-float(age)/float(scale)*float(frequency),at::kFloat);
  auto c=phase.cos(),s=phase.sin();return at::stack({k[0]*c-k[1]*s,k[0]*s+k[1]*c});}
Tensor quad(const Tensor& score,const Tensor& caps,const Tensor& bound,double tau){
  if(bound.item<float>()==0||caps.sum().item<float>()==0)return at::zeros_like(caps);
  if(bound.item<float>()>=caps.sum().item<float>())return caps;
  const auto zero=(score.sum()+caps.sum()+bound)*0;if(caps.numel()==1)return bound.reshape({1})+zero;
  if(caps.numel()!=2)throw std::logic_error("independent block formula expects at most two candidates");
  Tensor first;if(caps[0].item<float>()==0)first=at::zeros({},caps.options());else if(caps[1].item<float>()==0)first=bound;
  else {auto x=(bound+(score[0]-score[1])/tau)/2,lo=(bound-caps[1]).clamp_min(0),hi=at::minimum(caps[0],bound);
    first=x.item<float>()<lo.item<float>()?lo:x.item<float>()>hi.item<float>()?hi:x;}
  return at::stack({first,bound-first})+zero;
}
State update(State s,const Tensor& h,int64_t time,const Fields& p,const weighted::KVConfig& c){
  if(h[0].detach().item<float>()<=0)return s;auto u=norm(h.slice(0,1),c.epsilon);
  auto kv=at::mv(p.at("update.base"),u)+at::mv(p.at("update.left"),at::mv(p.at("update.right"),u));
  auto summary=at::mv(p.at("update.summary"),u);auto keep=at::arange(s.slots.at("times").numel(),at::kLong);
  if(c.window)keep=at::nonzero((time-s.slots.at("times"))<c.window).flatten();
  s.slots["times"]=at::cat({s.slots.at("times").index_select(0,keep),at::tensor({time},at::kLong)});
  for(const auto& name:std::vector<std::string>{"keys","values","summaries","weights"}){
    auto x=name=="keys"?kv.slice(0,0,2):name=="values"?kv.slice(0,2,4):name=="summaries"?summary:h[0].clamp_max(1);
    s.slots[name]=at::cat({s.slots.at(name).index_select(0,keep),x.unsqueeze(0)});}
  s.last_time=time;++s.observations;return s;
}
void read(const State& s,const Tensor& h,int64_t time,const Fields& p,const BlockSettleKVBinding& b,
    Tensor& memory,Tensor& total,Fields& desc){
  const auto& c=b.memory;auto u=norm(h.slice(0,1),c.epsilon),q=at::mv(p.at("read.query"),u);
  std::vector<Tensor> logits,features;std::vector<int64_t> ages;
  const auto times=s.slots.at("times");for(int64_t j=0;j<times.numel();++j){const auto age=time-times[j].item<int64_t>();ages.push_back(age);
    auto k=at::mv(p.at("read.key"),s.slots.at("keys")[j]);
    logits.push_back((q*rotate(k,age,c.read_frequencies[0],c.position_scale)).sum()/std::sqrt(2.));
    features.push_back(at::mv(p.at("read.value"),s.slots.at("values")[j]));}
  std::vector<Tensor> memories,totals;for(int channel=0;channel<2;++channel){
    auto w=s.slots.at("weights");if(!channel)w=w*at::exp(-float(c.beta_short)*at::tensor(ages,at::kLong).to(at::kFloat));
    auto mass=w.sum();totals.push_back(mass);
    memories.push_back(mass.clamp_max(1)*((at::stack(logits)+w.log()).softmax(0).unsqueeze(1)*at::stack(features)).sum(0));}
  memory=at::stack(memories);total=at::stack(totals);auto cur=at::mv(p.at("describe.summary.matrix"),u);
  auto feature=at::cat({u,h[0].log1p().reshape({1}),memory[0],memory[1],total[0].log1p().reshape({1}),
    total[1].log1p().reshape({1}),cur*memory[0],cur*memory[1]});
  desc["w"]=h[0].clamp_max(1).reshape({1});desc["mu"]=(b.regional.gamma_max*desc["w"]).clamp_max(1);
  desc["score"]=head(p,"describe.score.",feature);
  desc["evidence"]=norm(at::mv(p.at("describe.evidence.matrix"),feature)+p.at("describe.evidence.bias"),c.epsilon);
}
Tensor full(const State& s,const Tensor& h,int64_t time,const Fields& p,const weighted::KVConfig& c){
  auto u=h.slice(0,1),q=at::mv(p.at("full.query"),norm(u,c.epsilon));std::vector<Tensor> logits;
  for(int64_t j=0;j<s.slots.at("times").numel();++j){const auto age=time-s.slots.at("times")[j].item<int64_t>();
    logits.push_back((q*rotate(s.slots.at("keys")[j],age,c.full_frequencies[0][0],c.position_scale)).sum()/std::sqrt(2.)
      -float(age)/float(c.position_scale)*float(c.distance_slopes[0])+s.slots.at("weights")[j].log());}
  auto memory=s.slots.at("weights").sum().clamp_max(1)*(at::stack(logits).softmax(0).unsqueeze(1)*s.slots.at("values")).sum(0);
  auto x=u+c.alpha_att*at::mv(p.at("full.out"),memory),v=norm(x,c.epsilon);
  auto gate=at::mv(p.at("full.gate"),v),up=at::mv(p.at("full.up"),v);
  return x+c.alpha_ff*at::mv(p.at("full.down"),gate*gate.sigmoid()*up);
}
}
Formula formula(const Fixture& f){
  at::AutoGradMode enabled(true);Formula o;o.old.resize(2);o.proposal.resize(2);o.final.resize(2);o.describe.resize(2);
  o.received.resize(2);o.memory.resize(2);o.total.resize(2);o.controls.resize(2);o.full.resize(2);
  o.messages.assign(Frames*Nodes,at::zeros({3},at::kFloat));o.present=at::zeros({Frames,Nodes},at::kBool);o.times=at::zeros({Frames,Nodes},at::kLong);
  auto leaf=[&](Tensor x,Label label){x=x.detach().clone().set_requires_grad(true);o.leaves.push_back(x);o.labels.push_back(label);return x;};
  std::vector<Tensor> external;for(int64_t row=0;row<Frames*3;++row)external.push_back(leaf(f.input.payload.reshape({Frames*3,3})[row],{"input","",0,row}));
  std::map<std::string,Tensor> canonical;for(const auto& owner:f.registry.owners())canonical[owner.canonical]=leaf(owner.value,{"parameter",owner.canonical});
  std::vector<std::vector<Fields>> params(2);std::vector<std::vector<State>> states=f.states,bound=f.bound;
  for(int64_t r=0;r<2;++r){params[r].resize(2);for(const auto& [role,names]:f.roles[r])for(size_t bank=0;bank<names.size();++bank)
      params[r][bank][role]=canonical.at(f.registry.canonical_name(names[bank]));
    for(int set=0;set<2;++set)for(size_t row=0;row<states[r].size();++row){auto& s=(set?bound:states)[r][row];
      s.value=leaf(s.value,{set?"bound":"initial","value",r,int64_t(row)});
      for(auto& [name,x]:s.slots)if(name!="times")x=leaf(x,{set?"bound":"initial",name,r,int64_t(row)});}
  }
  const auto& g=f.graph.graph();
  // Independent time-major scalar schedule; the candidate uses whole-region
  // B x L tasks and device dependency selection.
  for(int64_t frame=0;frame<Frames;++frame)for(int64_t r=0;r<2;++r){const auto& binding=f.bindings[r];const auto& c=binding.memory;
    const int64_t count=r?1:2,offset=r?2:0;std::vector<int64_t> active;
    std::vector<Tensor> weights,caps,scores,evidence;
    for(int64_t n=0;n<count;++n){const auto node=offset+n,row=frame*count+n,owner=(frame/Length)*count+n;
      std::vector<std::pair<int64_t,Tensor>> atoms;std::vector<int64_t> owners;
      for(size_t p=0;p<g.inputs.size();++p)if(g.inputs[p]==node&&f.input.present.reshape({Frames,3})[frame][p].item<bool>()){
        atoms.push_back({g.source_domain->input[p],external[frame*3+p]});owners.push_back(binding.input_owners[p]);}
      for(size_t e=0;e<g.edges.size();++e)if(g.edges[e].target==node&&o.present[frame][g.edges[e].source].item<bool>()){
        atoms.push_back({g.source_domain->edge_target[e],o.messages[frame*Nodes+g.edges[e].source]});owners.push_back(binding.edge_owners[e]);}
      std::vector<size_t> order(atoms.size());for(size_t j=0;j<order.size();++j)order[j]=j;
      std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return atoms[a].first<atoms[b].first;});
      auto h=at::zeros({3},at::kFloat);if(!atoms.empty()){
        std::vector<Tensor> mass,terms;for(auto j:order){auto x=atoms[j].second;mass.push_back(x[0].clamp_min(0));}
        auto total=at::stack(mass).sum(),u=at::zeros({2},at::kFloat);
        if(total.item<float>()>0){for(size_t k=0;k<order.size();++k){const auto j=order[k];auto z=atoms[j].second.slice(0,1);const auto& p=params[r][owners[j]];
            auto hidden=at::mv(p.at("source.a"),norm(z,c.epsilon))+p.at("source.embedding");
            auto transformed=z+binding.aggregate_alpha*at::mv(p.at("source.b"),hidden*hidden.sigmoid());terms.push_back(mass[k]*transformed);}
          u=at::stack(terms).sum(0)/total;}
        h=at::cat({total.reshape({1}),u});}
      o.received[r].push_back(h);o.old[r].push_back(states[r][owner]);const auto time=f.input.positions.reshape({Frames})[frame].item<int64_t>()*2+r;
      const auto bank=binding.node_owners[n];if(!atoms.empty())states[r][owner]=update(states[r][owner],h,time,params[r][bank],c);
      o.proposal[r].push_back(states[r][owner]);Tensor memory=at::zeros({2,2},at::kFloat),total=at::zeros({2},at::kFloat);
      Fields desc{{"w",at::zeros({1},at::kFloat)},{"mu",at::zeros({1},at::kFloat)},{"score",at::zeros({1},at::kFloat)},{"evidence",at::zeros({2},at::kFloat)}};
      if(!atoms.empty()){const auto& s=binding.read_mode=="+"?states[r][owner]:binding.read_mode=="-"?o.old[r].back():bound[r][owner];
        read(s,h,time,params[r][bank],binding,memory,total,desc);active.push_back(n);weights.push_back(desc.at("w").reshape({}));
        caps.push_back(desc.at("mu").reshape({}));scores.push_back(desc.at("score").reshape({}));evidence.push_back(desc.at("evidence"));o.times[frame][node]=time;}
      o.memory[r].push_back(memory);o.total[r].push_back(total);o.describe[r].push_back(desc);o.controls[r].push_back(at::zeros({},at::kFloat));o.full[r].push_back(at::zeros({2},at::kFloat));
    }
    if(active.empty())continue;auto w=at::stack(weights),cap=at::stack(caps),sw=w.sum(),sc=cap.sum();
    auto context=at::cat({(w.unsqueeze(1)*at::stack(evidence)).sum(0)/sw.clamp_min(1),sw.log1p().reshape({1}),sc.log1p().reshape({1})});
    const auto context_owner=binding.parameters.message.read.region.context_owners[0].item<int64_t>();
    auto rho=head(params[r][context_owner],"region.context.",context).sigmoid().reshape({});
    auto low=sw.clamp_max(1),high=sc.clamp_max(*binding.regional.k);auto controls=quad(at::stack(scores),cap,low+rho*(high-low),*binding.regional.allocation_temperature);
    for(size_t j=0;j<active.size();++j){const auto n=active[j],row=frame*count+n,node=offset+n;auto control=controls[j];o.controls[r][row]=control;
      if(control.item<float>()>0){auto result=full(o.proposal[r][row],o.received[r][row],o.times[frame][node].item<int64_t>(),params[r][binding.node_owners[n]],c);
        o.full[r][row]=result;o.messages[frame*Nodes+node]=at::cat({control.reshape({1}),result});o.present[frame][node]=true;}}
  }
  o.final=std::move(states);return o;
}
}
