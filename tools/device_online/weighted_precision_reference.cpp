#include "weighted_precision_reference.h"
#include "tide/kernel.h"
#include "tide/full.h"
#include <torch/csrc/autograd/custom_function.h>
namespace tide::device_online::weighted_precision_check {
namespace {
class Store : public torch::autograd::Function<Store> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext*,const at::Tensor& x){
    auto y=x.to(at::kHalf).to(at::kFloat);
    if(!at::isfinite(y).all().item<bool>())throw std::runtime_error("reference half store overflow");return y;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext*,torch::autograd::variable_list g){return {g.at(0)};}
};
State store(State s){s.value=Store::apply(s.value);for(auto& [name,x]:s.slots)if(x.is_floating_point())x=Store::apply(x);return s;}
class StateStorage final : public StateKernel {
  std::shared_ptr<const StateKernel> inner_;
 public:
  explicit StateStorage(std::shared_ptr<const StateKernel> x):inner_(std::move(x)){}
  State initial(const NodeWeights& w)const override{return store(inner_->initial(w));}
  State step(const NodeWeights& w,const State& old,const ContentView& h,Index time)const override{return store(inner_->step(w,old,h,time));}
  std::set<std::string> integer_slots()const override{return inner_->integer_slots();}
  void validate_weights(const NodeWeights& w)const override{inner_->validate_weights(w);}
  void validate_state(const NodeWeights& w,const State& s)const override{inner_->validate_state(w,s);}
};
class PayloadStorage final : public FullKernel {
  std::shared_ptr<const FullKernel> inner_;
 public:
  explicit PayloadStorage(std::shared_ptr<const FullKernel> x):inner_(std::move(x)){}
  FullResult unrounded(const NodeWeights& w,const FullInput& x,Index slots,const Options& options)const{
    return inner_->step(w,x,slots,options);
  }
  FullResult step(const NodeWeights& w,const FullInput& x,Index slots,const Options& options)const override{
    auto out=inner_->step(w,x,slots,options);std::map<const void*,Tensor> saved;
    auto cast=[&](Tensor& value){const auto id=value.unsafeGetTensorImpl();if(!saved.count(id)){
        auto rounded=Store::apply(value);
        if(value[0].item<float>()>0&&rounded[0].item<float>()<=0)throw std::runtime_error("reference half support underflow");
        saved[id]=rounded;}value=saved.at(id);};
    cast(out.value);for(auto& slot:out.emitted)cast(slot.value);return out;
  }
  void validate_weights(const NodeWeights& w,Index slots)const override{inner_->validate_weights(w,slots);}
};
}
Model reference_model(const Model& source){
  auto model=source;std::map<const void*,Tensor> saved;
  auto copy=[&](Tensor& x){const auto id=x.unsafeGetTensorImpl();
    if(!saved.count(id))saved[id]=x.detach().to(at::kCPU,at::kFloat).clone().set_requires_grad(x.requires_grad());x=saved.at(id);};
  for(auto& w:model.nodes){for(auto x:{&w.decay,&w.weight,&w.bias,&w.read})copy(*x);for(auto& [_,x]:w.extra)copy(x);
    w.kernel=std::make_shared<StateStorage>(w.kernel);w.full_kernel=std::make_shared<PayloadStorage>(w.full_kernel);}
  for(auto& w:model.regions)for(auto& [_,x]:w.extra)copy(x);
  for(auto group:{&model.input_scale,&model.output_scale,&model.edge_scale,&model.agg_scale})for(auto& x:*group)copy(x);
  return model;
}
std::vector<External> reference_inputs(std::vector<External> xs){
  for(auto& x:xs)x.value=x.value.to(at::kFloat);return xs;
}
Continuation reference_continuation(Continuation q){
  for(auto& [_,s]:q.states){s.value=s.value.to(at::kFloat);for(auto& [name,x]:s.slots)if(x.is_floating_point())x=x.to(at::kFloat);}
  for(auto& a:q.pending)a.value=a.value.to(at::kFloat);return q;
}
Tensor unrounded_full(const Model& model,const Event& e){
  const auto& w=model.nodes.at(e.node);const auto* kernel=dynamic_cast<const PayloadStorage*>(w.full_kernel.get());
  if(!kernel)return e.full.slice(0,1);
  return kernel->unrounded(w,{&e.comparison_state,e.time,{e.content,e.sources,e.contributions},e.control},0,Options{}).value.slice(0,1);
}
void validate_storage(const Result& result,at::ScalarType dtype){
  auto check=[](const Tensor& x,at::ScalarType kind){
    if(x.defined()&&x.scalar_type()!=kind)throw std::runtime_error("weighted public storage dtype differs");};
  auto raw=[&](const State& s){check(s.value,dtype);for(const auto& [_,x]:s.slots)check(x,x.is_floating_point()?dtype:at::kLong);};
  auto atoms=[&](const std::vector<Atom>& xs){for(const auto& a:xs)check(a.value,dtype);};
  for(const auto& [_,s]:result.continuation.states)raw(s);atoms(result.continuation.pending);atoms(result.messages);
  for(const auto& o:result.outputs)check(o.value,dtype);
  for(const auto& e:result.trace){raw(e.old);raw(e.proposed_state);raw(e.comparison_state);raw(e.next_state);atoms(e.fiber);
    for(const auto& x:{e.proposal,e.comparison,e.next,e.full})check(x,dtype);
    check(e.content,at::kFloat);check(e.control,at::kFloat);
    for(const auto& [_,x]:e.descriptor_fields)check(x,at::kFloat);
    for(const auto& a:e.emitted)check(a.value,dtype);
    for(const auto& a:e.sources){check(a.atom.value,dtype);check(a.scale,dtype);}
  }
}
}
