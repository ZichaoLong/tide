#include "training_test.h"
#include "full_vjp_fixture.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>
#include <set>

namespace tide::device_online::test {
void train_require(bool b,const char* why){if(!b)throw std::runtime_error(why);}
void train_reject(const std::function<void()>& f,const char* why) {
  bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}train_require(rejected,why);
}
Model train_model(Model m,at::ScalarType dtype) {
  std::map<const void*,Tensor> copies;
  auto copy=[&](Tensor& x){const auto key=x.unsafeGetTensorImpl();auto it=copies.find(key);
    if(it==copies.end())it=copies.emplace(key,x.detach().to(dtype).clone().set_requires_grad(true)).first;x=it->second;};
  for(auto& w:m.nodes){copy(w.decay);copy(w.weight);copy(w.bias);copy(w.read);for(auto& [_,x]:w.extra)copy(x);}
  for(auto& w:m.regions)for(auto& [_,x]:w.extra)copy(x);
  for(auto* group:{&m.input_scale,&m.agg_scale,&m.edge_scale,&m.output_scale})for(auto& x:*group)copy(x);return m;
}
Continuation train_boundary(Continuation q,at::ScalarType dtype) {
  auto copy=[&](Tensor& x){x=x.detach().to(dtype).clone();};
  for(auto& [_,s]:q.states){copy(s.value);for(auto& [__,x]:s.slots)copy(x);}
  for(auto& [_,h]:q.history)for(auto& [__,x]:h.tensors)copy(x);
  for(auto& a:q.pending)copy(a.value);return q;
}
ResidentCotangents train_roots(const ResidentTrainingWindow& w,int window,int mode) {
  ResidentCotangents r;r.token=w.token;
  if(mode==4||mode==9) {
    r.outputs=at::full_like(w.outputs.values,.0625);r.outputs_connected=w.outputs.valid.clone();
    r.pending=at::full_like(w.pending_values,.015625);r.pending_connected=w.pending_valid.clone();
    r.final=at::full_like(w.state_values,.03125);r.final_connected=w.state_present.clone();
  } else if(mode==5&&window==3) {r.final=at::zeros_like(w.state_values);r.final_connected=w.state_present.clone();}
  if(mode==9||window==3&&(mode>=6&&mode<=8||mode==10))for(const auto& c:w.cache) {
    ResidentCacheCotangents a;
    if(mode!=7&&mode!=10){a.key=at::full_like(c.key,mode==8?0.:.0078125);a.key_connected=c.present.clone();}
    if(mode!=6&&mode!=10){a.value=at::full_like(c.value,mode==8?0.:-.015625);a.value_connected=c.present.clone();}
    if(c.log_bias.defined()&&(mode==8||mode==9||mode==10)) {
      a.log_bias=at::full_like(c.log_bias,mode==8?0.:.0234375);a.log_bias_connected=c.present.clone();
    }
    r.cache.push_back(a);
  }
  return r;
}
void train_gradients(const ResidentGradients& g,const RetainedReference& ref,const Fixture& f) {
  auto values=g.values.cpu(),on=g.connected.cpu(),initial=g.initial.cpu(),ic=g.initial_connected.cpu();
  const auto owners=f.model.parameters(true).owners();train_require(owners.size()==g.names.size(),"public trainable owner count differs");
  for(size_t i=0;i<owners.size();++i) {
    const auto& o=owners[i];train_require(o.canonical==g.names[i]&&o.aliases==g.aliases[i],"public owner alias layout differs");
    auto v=g.offsets[i]<0?at::zeros_like(o.value).to(at::kFloat):values.narrow(0,g.offsets[i],o.value.numel()).reshape(o.value.sizes());
    full_same(v,on[i],ref.gradients.at(o.canonical),o.canonical.c_str());
  }
  for(const auto& [o,_]:f.initial.states) {
    auto name="state/"+std::to_string(o.first)+"/"+std::to_string(o.second);
    full_same(initial[o.first][o.second],ic[o.first][o.second],ref.gradients.at(name),name.c_str());
  }
  for(const auto& a:g.initial_cache) {
    auto key=a.key.cpu(),value=a.value.cpu(),lengths=a.lengths.cpu(),kc=a.key_connected.cpu(),vc=a.value_connected.cpu();
    auto bias=a.log_bias.defined()?a.log_bias.cpu():Tensor{},bc=a.log_bias_connected.defined()?a.log_bias_connected.cpu():Tensor{};
    for(Index b=0;b<f.initial.batch_size;++b)for(size_t i=0;i<a.nodes.size();++i) {
      const Index n=a.nodes[i],owner=b*a.nodes.size()+i;const auto found=f.initial.states.find({b,n});
      if(found==f.initial.states.end()) {train_require(!kc[owner].item<bool>()&&!vc[owner].item<bool>()&&(!bc.defined()||!bc[owner].item<bool>()),"absent initial cache acquired gradient");continue;}
      const auto length=found->second.slots.at("key").size(0);train_require(lengths[owner].item<Index>()==length,"initial cache length changed");
      for(const auto& name:{std::string("key"),std::string("value")}) {
        const auto leaf="cache/"+name+"/"+std::to_string(b)+"/"+std::to_string(n);
        full_same((name=="key"?key:value)[owner].narrow(0,0,length),(name=="key"?kc:vc)[owner],ref.gradients.at(leaf),leaf.c_str());
      }
      if(bias.defined()) {
        const auto leaf="cache/log_bias/"+std::to_string(b)+"/"+std::to_string(n);
        full_same(bias[owner].narrow(0,0,length),bc[owner],ref.gradients.at(leaf),leaf.c_str());
      }
    }
  }
  std::set<std::string> seen;
  for(size_t w=0;w<g.boundaries.size();++w) {
    const auto& b=g.boundaries[w];auto meta=b.coordinates.cpu(),valid=b.valid.cpu(),v=b.values.cpu(),c=b.connected.cpu();
    for(Index i=0;i<valid.numel();++i)if(valid[i].item<bool>()) {
      const auto row=meta[i];if(w&&row[3].item<Index>()!=0)continue; // Later pending leaves link internally.
      Atom a{row[0].item<Index>(),row[1].item<Index>(),row[2].item<Index>(),row[3].item<Index>(),row[4].item<Index>(),row[5].item<Index>(),{}};
      auto name=boundary_name(a);train_require(seen.insert(name).second,"duplicate public boundary leaf");
      full_same(v[i],c[i],ref.gradients.at(name),name.c_str());
    }
  }
  for(const auto& [name,_]:ref.gradients)if(name.rfind("boundary/",0)==0)train_require(seen.count(name),"public backward lost boundary leaf");
}
void train_checkpoint(const ResidentTrainingCheckpoint& c,const Model& m,const NamedOptimizer& optimizer) {
  const auto registry=m.parameters(false),trainable=m.parameters(true);auto yes=at::ones({},at::kBool);
  train_require(c.aliases==registry.alias_partitions()&&c.trainable==trainable.names(),"checkpoint owner identity changed");
  for(const auto& o:registry.owners())full_same(c.parameters.at(o.canonical),yes,o.value,o.canonical.c_str());
  const auto owners=trainable.owners();
  for(size_t i=0;i<owners.size();++i) {
    auto found=optimizer.state().find(owners[i].canonical);if(found==optimizer.state().end())continue;
    const auto& slot=found->second;const auto offset=c.offsets[i],count=owners[i].value.numel();
    auto same=[&](const Tensor& packed,const Tensor& ref){if(ref.defined())full_same(packed.narrow(0,offset,count).reshape(ref.sizes()),yes,ref,"optimizer slot");};
    same(c.state.first,slot.momentum_buffer.defined()?slot.momentum_buffer:slot.exp_avg);
    same(c.state.second,slot.exp_avg_sq);same(c.state.maximum,slot.max_exp_avg_sq);
    if(c.optimizer==ResidentOptimizerKind::adamw)train_require(c.state.steps[i].item<Index>()==slot.step,"checkpoint optimizer counter differs");
  }
}
} // namespace tide::device_online::test
