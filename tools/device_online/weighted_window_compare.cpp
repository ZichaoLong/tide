#include "weighted_window_check.h"
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
namespace tide::device_online::window_check {
namespace {
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void gradient(const at::Tensor& actual,bool connected,const at::Tensor& expected,const std::string& label){
  require(connected==expected.defined(),label+": None/zero differs actual="+std::to_string(connected));
  if(!connected){require(actual.eq(0).all().item<bool>(),label+": disconnected nonzero");return;}
  require(actual.sizes()==expected.sizes(),label+": shape differs");
  if(!at::allclose(actual.cpu(),expected,3e-5,3e-6)){std::cerr<<label<<" actual "<<actual.cpu()<<" expected "<<expected<<'\n';throw std::runtime_error(label+": gradient differs");}
}
bool same(const at::Tensor& row,const Atom& a){const int64_t want[]{a.batch,a.node,a.time,a.kind,a.source,a.position};
  for(int64_t j=0;j<6;++j)if(row[j].item<int64_t>()!=want[j])return false;return true;}
}
void verify(const Graph& graph,const Model& model,const Continuation& initial,const std::vector<External>& inputs,
    const Result& result,WeightedWindowAdjoint& vjp,const Records& data,const std::string& mode){
  at::AutoGradMode enabled(true);std::vector<at::Tensor> outputs,bars,leaves;const auto& roots=vjp.roots();
  auto add=[&](const at::Tensor& x,const at::Tensor& bar){if(x.requires_grad()){outputs.push_back(x);bars.push_back(bar.cpu().reshape(x.sizes()));}};
  auto raw=[&](const State& s,const WeightedRawLayout& layout,const WeightedRawAdjoint& r){
    auto flags=r.connected.cpu();for(size_t j=0;j<layout.fields().size();++j)if(flags[j].item<bool>()){
      const auto& part=layout.fields()[j];const auto& x=field(s,part.name);add(x,r.value.narrow(0,part.offset,x.numel()));}};
  for(const auto& e:result.trace){const auto t=stage(data,e.node,e.batch,e.time),b=e.batch;const auto& r=roots.nodes[e.node];
    if(r.content.connected[t][b].item<bool>())add(e.content,r.content.value[t][b]);
    if(r.control.connected[t][b].item<bool>())add(e.control,r.control.value[t][b]);
    if(e.active&&r.full.connected[t][b].item<bool>())add(e.full.narrow(0,1,e.full.numel()-1),r.full.value[t][b]);
    for(const auto& [name,g]:r.describe)if(g.connected[t][b].item<bool>())add(e.descriptor_fields.at(name),g.value[t][b]);
    raw(e.old,vjp.nodes()[e.node]->layout,{r.old.value[t][b],r.old.connected[t][b]});
    raw(e.proposed_state,vjp.nodes()[e.node]->layout,{r.proposal.value[t][b],r.proposal.connected[t][b]});
  }
  for(const auto& [key,s]:result.continuation.states){const auto& r=roots.nodes[key.second].final;
    raw(s,vjp.nodes()[key.second]->layout,{r.value[key.first],r.connected[key.first]});}
  auto atoms=[&](const std::vector<Atom>& values,const at::Tensor& coordinates,const WeightedRowAdjoint& root){
    const auto flags=root.connected.cpu();for(int64_t j=0;j<flags.numel();++j)if(flags[j].item<bool>()){
      const Atom* found=nullptr;for(const auto& a:values)if(same(coordinates[j],a)){require(!found,"ambiguous oracle atom");found=&a;}
      require(found,"root absent from independently scheduled oracle");add(found->value,root.value[j]);}};
  for(int64_t t=0;t<data.at("counts").size(0);++t){
    atoms(result.messages,data.at("message.meta")[t],{roots.messages.value[t],roots.messages.connected[t]});
    std::vector<Atom> fibers;for(const auto& e:result.trace)fibers.insert(fibers.end(),e.fiber.begin(),e.fiber.end());
    atoms(fibers,data.at("fiber.meta")[t],{roots.fibers.value[t],roots.fibers.connected[t]});
  }
  atoms(result.continuation.pending,data.at("right.pending.meta"),roots.pending);
  auto registry=model.parameters();for(const auto& owner:registry.owners())leaves.push_back(owner.value);
  for(const auto& x:inputs)leaves.push_back(x.value);for(const auto& a:initial.pending)leaves.push_back(a.value);
  for(const auto& [key,s]:initial.states){leaves.push_back(s.value);for(const auto& [name,x]:s.slots)if(x.scalar_type()==at::kFloat)leaves.push_back(x);}
  auto expected=outputs.empty()?std::vector<at::Tensor>(leaves.size()):torch::autograd::grad(outputs,leaves,bars,true,false,true);
  size_t index=0;const auto& parameters=vjp.parameters();const auto parameter_values=parameters.values.cpu(),parameter_flags=parameters.connected.cpu();
  for(size_t j=0;j<parameters.owners.size();++j){const auto& owner=parameters.owners[j];const auto offset=parameters.offsets[j];
    auto actual=offset<0?at::zeros_like(owner.value):parameter_values.narrow(0,offset,owner.value.numel()).reshape(owner.value.sizes());
    gradient(actual,parameter_flags[j].item<bool>(),expected.at(index++),owner.canonical);}
  const auto& boundary=vjp.boundary();auto coords=boundary.coordinates.cpu(),present=boundary.present.cpu(),values=boundary.values.cpu(),flags=boundary.connected.cpu();
  auto input=[&](const Atom& atom){int64_t found=-1;for(int64_t i=0;i<present.numel();++i)if(present[i].item<bool>()&&same(coords[i],atom)){
      require(found<0,"ambiguous candidate boundary root");found=i;}require(found>=0,"missing boundary root");
    gradient(values[found],flags[found].item<bool>(),expected.at(index++),"boundary."+std::to_string(index));};
  for(const auto& x:inputs)input({x.batch,graph.inputs[x.port],x.time,0,x.port,x.position,x.value});
  for(const auto& a:initial.pending)input(a);
  for(const auto& [key,s]:initial.states){const auto b=key.first,v=key.second;const auto& layout=vjp.nodes()[v]->layout;
    const auto values=vjp.states()[v].value.cpu(),flags=vjp.states()[v].connected.cpu();
    auto one=[&](const std::string& name,const at::Tensor& x){size_t j=0;while(layout.fields()[j].name!=name)++j;const auto& f=layout.fields()[j];
      gradient(values[b].narrow(0,f.offset,x.numel()).reshape(x.sizes()),flags[b][j].item<bool>(),expected.at(index++),"left."+std::to_string(v)+"."+name);};
    one("value",s.value);for(const auto& [name,x]:s.slots)if(x.scalar_type()==at::kFloat)one(name,x);
  }
  for(size_t v=0;v<vjp.nodes().size();++v)for(int64_t b=0;b<initial.batch_size;++b)if(!initial.states.count({b,int64_t(v)}))
    require(!vjp.states()[v].connected[b].cpu().any().item<bool>(),"bound initial constant exported as a checkpoint leaf");
  require(index==expected.size(),"window roots not completely checked");
}
}
