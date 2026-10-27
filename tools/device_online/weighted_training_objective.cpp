// Public fields -> independent Greedy mathematical returns. No native adjoint
// helper or reference-derived execution schedule enters the candidate.
#include "weighted_training_check.h"
#include <cmath>
#include <iostream>
namespace tide::device_online::weighted_training_check {
void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
void gradient(const at::Tensor& value,bool connected,const at::Tensor& expected,const std::string& name){
  require(connected==expected.defined(),name+" None/zero mismatch");
  if(connected){const auto actual=value.cpu();
    if(!at::allclose(actual,expected,3e-5,3e-6)){
      const auto diff=(actual-expected).abs();
      throw std::runtime_error(name+" gradient differs; max_abs="+std::to_string(diff.max().item<double>())
        +" max_tolerance_ratio="+std::to_string((diff/(3e-6+3e-5*expected.abs())).max().item<double>()));}}
}
void equal_window(const WeightedTrainingWindow& a,const WeightedTrainingWindow& b){
  require(a.start==b.start&&a.stop==b.stop&&a.fields.size()==b.fields.size(),"restored window layout differs");
  require(at::equal(a.metadata.at("stages"),b.metadata.at("stages")),"restored stage count differs");
  for(const auto& [name,x]:a.fields){const auto& y=b.fields.at(name);
    require(at::equal(x.present,y.present),"restored field support differs: "+name);
    require(x.valid.defined()==y.valid.defined(),"restored ragged layout differs: "+name);
    auto mask=x.present;
    if(x.valid.defined()){require(at::equal(x.valid,y.valid),"restored raw field length differs: "+name);mask=x.valid&mask.unsqueeze(-1);}
    while(mask.dim()<x.values.dim())mask=mask.unsqueeze(-1);
    const auto actual=at::where(mask,x.values,at::zeros_like(x.values));
    const auto expected=at::where(mask,y.values,at::zeros_like(y.values));
    // All logical raw fields, including expired retained entries, are exact.
    // Unused journal rows and ragged capacity padding have no defined value.
    require(at::equal(actual,expected),"restored logical field values differ: "+name);
    if(!at::equal(x.values,y.values))std::cout<<"restored unused field padding differs: "<<name<<" logical_values_exact=true\n"<<std::flush;
  }
}
namespace {
bool atom(const at::Tensor& c,const Atom& a){return c[0].item<int64_t>()==a.batch&&c[1].item<int64_t>()==a.node
  &&c[2].item<int64_t>()==a.time&&c[3].item<int64_t>()==a.kind&&c[4].item<int64_t>()==a.source&&c[5].item<int64_t>()==a.position;}
const at::Tensor& raw(const State& s,const std::string& name){return name=="value"?s.value:s.slots.at(name);}
}
WeightedCotangents seed(const WeightedTrainingWindow& w,const std::string& mode){
  WeightedCotangents result;result.token=w.token;
  for(const auto& [name,f]:w.fields){
    bool use=mode=="all"||mode=="zero"||(mode=="outputs"&&name=="outputs")||(mode=="final"&&name.find(".final.")!=std::string::npos);
    if(!use)continue;auto connected=f.present.contiguous().clone();
    auto x=(.07*at::sin(at::arange(f.values.numel(),at::kFloat)*.19+double(name.size()))).reshape(f.values.sizes()).to(f.values.device());
    if(mode=="zero")x.zero_();
    if(f.valid.defined()){auto valid=f.valid;while(valid.dim()<x.dim())valid=valid.unsqueeze(-1);x=at::where(valid,x,at::zeros_like(x));}
    auto mask=connected;while(mask.dim()<x.dim())mask=mask.unsqueeze(-1);
    result.fields[name]={at::where(mask,x,at::full_like(x,NAN)),connected};
  }
  return result;
}
Objective objective(const WeightedTrainingWindow& w,const WeightedCotangents& seed,const Result& reference){
  Objective out;std::map<std::string,int64_t> seen;
  std::map<std::string,at::Tensor> values,flags,metadata;
  for(const auto& [name,f]:w.fields){values[name]=f.values.cpu();flags[name]=f.present.cpu();}
  for(const auto& [name,x]:w.metadata)metadata[name]=x.cpu();
  auto take=[&](const std::string& name,at::Tensor actual,at::Tensor value,at::Tensor bar){
    ++seen[name];require(at::allclose(actual.reshape(value.sizes()),value.detach(),1e-5,1e-6),"public field differs: "+name);
    if(value.requires_grad()){out.first.push_back(value);out.second.push_back(bar.reshape(value.sizes()));}
  };
  auto row=[&](const std::string& name,const at::Tensor& value,int64_t t,int64_t b,bool final=false){
    if(!seed.fields.count(name))return;auto actual=final?values.at(name)[b]:values.at(name)[t][b];
    auto r=seed.fields.at(name).values.cpu();auto bar=final?r[b]:r[t][b];
    if(w.fields.at(name).valid.defined()){actual=actual.narrow(0,0,value.size(0));bar=bar.narrow(0,0,value.size(0));}
    take(name,actual,value,bar);
  };
  for(const auto& e:reference.trace){auto prefix="node."+std::to_string(e.node)+".";int64_t stage=-1;
    auto present=w.fields.at(prefix+"content").present.cpu(),times=metadata.at(prefix+"times");
    for(int64_t t=0;t<present.size(0);++t)if(present[t][e.batch].item<bool>()&&times[t][e.batch].item<int64_t>()==e.time){require(stage<0,"duplicate public event");stage=t;}
    require(stage>=0,"missing public event");row(prefix+"content",e.content,stage,e.batch);row(prefix+"control",e.control.reshape({1}),stage,e.batch);
    if(e.active)row(prefix+"full",e.full.narrow(0,1,e.full.numel()-1),stage,e.batch);
    for(const auto& [name,x]:e.descriptor_fields)row(prefix+"describe."+name,x,stage,e.batch);
    for(const auto& item:std::vector<std::pair<std::string,State>>{{"old.",e.old},{"proposal.",e.proposed_state}}){
      row(prefix+item.first+"value",item.second.value,stage,e.batch);
      for(const auto& [name,x]:item.second.slots)if(x.is_floating_point())row(prefix+item.first+name,x,stage,e.batch);}
  }
  for(const auto& [key,state]:reference.continuation.states){const auto prefix="node."+std::to_string(key.second)+".final.";
    row(prefix+"value",state.value,0,key.first,true);for(const auto& [name,x]:state.slots)if(x.is_floating_point())row(prefix+name,x,0,key.first,true);}
  auto atoms=[&](const std::string& name,const std::vector<Atom>& items){if(!seed.fields.count(name))return;
    auto coordinates=metadata.at(name+".coordinates").reshape({-1,6}),valid=flags.at(name).reshape({-1});
    auto x=values.at(name).reshape({valid.numel(),-1}),bar=seed.fields.at(name).values.cpu().reshape(x.sizes());
    for(const auto& a:items){int64_t index=-1;for(int64_t i=0;i<valid.numel();++i)if(valid[i].item<bool>()&&atom(coordinates[i],a)){
        require(index<0,"duplicate public atom");index=i;}
      require(index>=0,"missing public atom");take(name,x[index],a.value,bar[index]);}
  };
  atoms("messages",reference.messages);atoms("pending",reference.continuation.pending);
  std::vector<Atom> fibers;for(const auto& e:reference.trace)fibers.insert(fibers.end(),e.fiber.begin(),e.fiber.end());atoms("fibers",fibers);
  if(seed.fields.count("outputs")){
    auto coordinates=metadata.at("outputs.coordinates").reshape({-1,3}),valid=flags.at("outputs").reshape({-1});
    auto x=values.at("outputs").reshape({valid.numel(),-1}),bar=seed.fields.at("outputs").values.cpu().reshape(x.sizes());
    for(const auto& o:reference.outputs){int64_t index=-1;for(int64_t i=0;i<valid.numel();++i)if(valid[i].item<bool>()){
        const auto c=coordinates[i];if(c[0].item<int64_t>()==o.batch&&c[1].item<int64_t>()==o.time&&c[2].item<int64_t>()==o.port){require(index<0,"duplicate physical output");index=i;}}
      require(index>=0,"missing physical output");take("outputs",x[index],o.value,bar[index]);}
  }
  for(const auto& [name,r]:seed.fields)require(seen[name]==r.connected.sum().item<int64_t>(),"public field root support differs: "+name);
  return out;
}
void check_gradients(const Graph& graph,const ParameterRegistry& registry,const Continuation& initial,
    const std::vector<std::vector<External>>& inputs,const std::vector<at::Tensor>& expected,const WeightedGradients& result){
  const auto& p=result.parameters;size_t cursor=0;
  std::vector<std::string> names;for(const auto& owner:registry.owners())names.push_back(owner.canonical);
  require(p.names==names&&p.aliases==registry.alias_partitions(),"public gradient owner names/aliases differ");
  auto values=p.values.cpu(),connected=p.connected.cpu();
  for(size_t i=0;i<p.names.size();++i){const auto x=registry.value(p.names[i]);gradient(values.narrow(0,p.offsets[i],x.numel()).reshape(x.sizes()),connected[i].item<bool>(),expected.at(cursor++),p.names[i]);}
  auto boundary=[&](size_t window,const Atom& a){const auto& g=result.boundaries.at(window);auto meta=g.coordinates.cpu(),valid=g.valid.cpu();int64_t found=-1;
    for(int64_t j=0;j<valid.numel();++j)if(valid[j].item<bool>()&&atom(meta[j],a)){require(found<0,"ambiguous gradient boundary");found=j;}
    require(found>=0,"missing gradient boundary");gradient(g.values[found],g.connected[found].item<bool>(),expected.at(cursor++),"public input/pending");};
  for(size_t i=0;i<inputs.size();++i)for(const auto& x:inputs[i])boundary(i,{x.batch,graph.inputs[x.port],x.time,0,x.port,x.position,x.value});
  for(const auto& a:initial.pending)boundary(0,a);
  for(const auto& [key,s]:initial.states){const auto& g=result.initial.at(key.second);
    auto one=[&](const std::string& name,const at::Tensor& x){const auto& field=g.fields.at(name);auto value=field.values[key.first];
      if(value.sizes()!=x.sizes())value=value.narrow(0,0,x.size(0));
      gradient(value,field.connected[key.first].item<bool>(),expected.at(cursor++),"public initial "+name);};
    one("value",s.value);for(const auto& [name,x]:s.slots)if(x.is_floating_point())one(name,x);}
  require(cursor==expected.size(),"public gradient roots incomplete");
}
}
