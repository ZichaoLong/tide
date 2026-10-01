#include "precision_graph_fixture.h"
#include "tide/aggregate.h"
#include "tide/kernel.h"
#include "tide/full.h"
#include "tide/read.h"
#include "tide/counters.h"
#include "tide/ops.h"
#include <torch/csrc/autograd/custom_function.h>
#include <algorithm>
#include <stdexcept>

namespace tide::device_online::test {
namespace {
Tensor rounded(const Tensor& x,at::ScalarType type=at::kHalf) {
  // Exact quantized forward, usual identity cast VJP without half gradient
  // accumulation. All fixture values are finite.
  return x.detach().to(type).to(x.scalar_type())+(x-x.detach());
}
class HalfActivation : public torch::autograd::Function<HalfActivation> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor x,bool sigmoid) {
    auto value=x.to(at::kHalf);value=(sigmoid?value.sigmoid():value.tanh()).to(x.scalar_type());
    ctx->save_for_backward({value});ctx->saved_data["sigmoid"]=sigmoid;return value;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list g) {
    const auto y=ctx->get_saved_variables()[0];
    return {g[0]*(ctx->saved_data["sigmoid"].toBool()?y*(1-y):1-y*y),Tensor()};
  }
};
class HalfHst : public torch::autograd::Function<HalfHst> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor h,Tensor g,Tensor p,double zeta) {
    ctx->set_materialize_grads(false);ctx->save_for_backward({rounded(g-h)});ctx->saved_data["zeta"]=zeta;return g.clone();
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list u) {
    if(!u[0].defined())return {Tensor(),Tensor(),Tensor(),Tensor()};
    return {at::zeros_like(u[0]),u[0],(u[0]*ctx->get_saved_variables()[0]).sum()*ctx->saved_data["zeta"].toDouble(),Tensor()};
  }
};
class HalfState final : public StateKernel {
  std::string kind_;
 public:
  explicit HalfState(const Node& n):kind_(n.identity?"identity":n.memory) {
    if(kind_!="identity"&&kind_!="ema"&&kind_!="lh-add-repeat-v1")throw std::invalid_argument("half CPU graph reference state unavailable");
  }
  State initial(const NodeWeights& w) const override {return {at::zeros_like(w.bias)};}
  State step(const NodeWeights& w,const State& old,const ContentView& h,Index time) const override {
    if(kind_=="identity")return old;
    auto value=old.value;
    if(kind_=="ema")value=rounded(HalfActivation::apply(w.decay,true)*value);
    else for(Index tick=old.last_time;tick<time;++tick)value=rounded(value*w.extra.at("add_retention"));
    return {rounded(value+h.value),time,increment(old.observations)};
  }
  void validate_weights(const NodeWeights& w) const override {make_state_kernel(kind_)->validate_weights(w);}
  void validate_state(const NodeWeights& w,const State& s) const override {make_state_kernel(kind_)->validate_state(w,s);}
};
class HalfSum final : public AggregateKernel {
 public:
  AggregateResult step(const NodeWeights&,const AggregateInput& input) const override {
    AggregateResult result;
    for(const auto& source:input.sources) {
      // Delivery stores half before source scaling. Sum has FP32 products and
      // ordered FP32 accumulation; exported contributions round independently.
      auto product=rounded(rounded(source.atom.value)*source.scale,at::kFloat);
      result.value=result.value.defined()?rounded(result.value+product,at::kFloat):product;
      result.contributions.push_back({source.slot,rounded(product)});
    }
    result.value=rounded(result.value);
    std::sort(result.contributions.begin(),result.contributions.end(),[](const auto& a,const auto& b){return a.slot<b.slot;});
    return result;
  }
  void validate_weights(const NodeWeights&,Index) const override {}
};
class HalfFull final : public FullKernel {
  Node node_;
 public:
  explicit HalfFull(Node n):node_(std::move(n)) {
    if(node_.emission!="broadcast"||(node_.full!="identity"&&node_.full!="tanh"))
      throw std::invalid_argument("half CPU graph reference Full unavailable");
  }
  FullResult step(const NodeWeights& w,const FullInput& in,Index slots,const Options& options) const override {
    const auto h=in.content.value;auto fresh=h;
    if(!node_.identity&&node_.full=="tanh") {
      const auto c=in.comparison->value;
      auto product=at::matmul(c,w.weight);
      auto actual=at::matmul(c.detach().to(at::kHalf),w.weight.detach().to(at::kHalf)).to(c.scalar_type());
      product=actual+(product-product.detach());
      fresh=rounded(h+HalfActivation::apply(rounded(product+w.bias),false));
    }
    auto value=fresh;
    if(!node_.identity&&options.mode=="hst")value=HalfHst::apply(h,fresh,rounded(in.control),options.zeta);
    else if(!node_.identity&&options.mode=="softp")value=rounded(h+rounded(rounded(in.control)*rounded(fresh-h)));
    FullResult out{value,{}};
    for(Index slot=0;slot<slots;++slot) {
      const auto phase=node_.emit_phases.empty()?-1:node_.emit_phases[slot];
      if(phase==-1||(phase>=0&&in.time%node_.emit_period==phase))out.emitted.push_back({slot,value});
    }
    return out;
  }
  void validate_weights(const NodeWeights& w,Index slots) const override {make_full_kernel(node_)->validate_weights(w,slots);}
};
}
void fixture_dtype(Fixture& f,at::ScalarType dtype) {
  // Preserve parameter aliases across the dtype conversion.
  std::map<const void*,Tensor> copies;
  auto copy=[&](Tensor& x){auto key=x.unsafeGetTensorImpl();auto it=copies.find(key);
    if(it==copies.end())it=copies.emplace(key,x.to(dtype)).first;x=it->second;};
  for(auto& w:f.model.nodes){for(auto* x:{&w.decay,&w.weight,&w.bias,&w.read})copy(*x);for(auto& [_,x]:w.extra)copy(x);}
  for(auto* group:{&f.model.input_scale,&f.model.agg_scale,&f.model.edge_scale,&f.model.output_scale})for(auto& x:*group)copy(x);
  for(auto& [_,s]:f.initial.states){s.value=s.value.to(dtype);for(auto& [__,x]:s.slots)x=x.to(dtype);}
  for(auto& a:f.initial.pending)a.value=a.value.to(dtype);
  for(auto& x:f.input)x.value=x.value.to(dtype);
}
void configure_half_reference(Fixture& f) {
  for(size_t n=0;n<f.graph.nodes.size();++n) {
    const auto& spec=f.graph.nodes[n];auto& w=f.model.nodes[n];
    if(spec.aggregation!="sum")throw std::invalid_argument("half CPU graph reference Aggregate unavailable");
    w.kernel=std::make_shared<HalfState>(spec);w.aggregate_kernel=std::make_shared<HalfSum>();
    w.full_kernel=std::make_shared<HalfFull>(spec);
  }
}
void round_half_transport(Result& result) {
  for(auto& x:result.outputs)x.value=rounded(x.value);
  for(auto& x:result.messages)x.value=rounded(x.value);
  for(auto& x:result.continuation.pending)x.value=rounded(x.value);
  for(auto& e:result.trace) {
    for(auto& x:e.fiber)x.value=rounded(x.value);
    for(auto& x:e.sources)x.atom.value=rounded(x.atom.value);
    e.control=rounded(e.control);
  }
}
} // namespace tide::device_online::test
