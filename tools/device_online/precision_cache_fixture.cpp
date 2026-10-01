#include "precision_cache_fixture.h"
#include "tide/fiber_attention.h"
#include "tide/counters.h"
#include <algorithm>
#include <cmath>

namespace tide {
std::shared_ptr<const StateKernel> make_attention_kernel(Index,Index,Index);
}
namespace tide::device_online::test {
namespace {
Tensor rounded(const Tensor& x) {
  return x.detach().to(at::kHalf).to(x.scalar_type())+(x-x.detach());
}
Tensor half_matmul(const Tensor& a,const Tensor& b) {
  auto y=at::matmul(a,b);
  return at::matmul(a.detach().to(at::kHalf),b.detach().to(at::kHalf)).to(y.scalar_type())+(y-y.detach());
}
// Dense global normalization is independent of the candidate's key tiling.
// QK uses real half operands/results; softmax and the weighted sum are FP32.
Tensor attend(const Tensor& q,const Tensor& k,const Tensor& v,const Tensor& bias,double scale) {
  auto score=half_matmul(q,k.transpose(-1,-2)).to(at::kFloat)*float(scale);
  if(bias.defined())score=score+bias.to(at::kFloat);
  auto y=at::matmul(at::softmax(score,-1),v.to(at::kFloat));
  return rounded(y).to(q.scalar_type());
}
class HalfCache final : public StateKernel {
  Node node_;
  std::string pool_;
  std::shared_ptr<const StateKernel> plain_;
 public:
  HalfCache(Node n,Index inputs):node_(std::move(n)) {
    if(node_.memory=="attention")plain_=make_attention_kernel(node_.query_heads,node_.kv_heads,node_.window);
    else {
      plain_=make_fiber_attention_kernel(node_,inputs);
      const std::string prefix="lh-fiber-attention-",suffix="-repeat-v1";
      pool_=node_.memory.substr(prefix.size(),node_.memory.size()-prefix.size()-suffix.size());
    }
  }
  State initial(const NodeWeights& w) const override {return plain_->initial(w);}
  State reset(const State& s) const override {return plain_->reset(s);}
  void validate_weights(const NodeWeights& w) const override {plain_->validate_weights(w);}
  void validate_state(const NodeWeights& w,const State& s) const override {plain_->validate_state(w,s);}
  void validate_policy(const Node& n,Index inputs) const override {plain_->validate_policy(n,inputs);}
  State step(const NodeWeights& w,const State& old,const ContentView& content,Index time) const override {
    const Index width=w.bias.numel(),h=node_.query_heads,kh=node_.kv_heads,d=width/h;
    if(pool_.empty()) {
      auto q=half_matmul(content.value,w.extra.at("attn_q")).reshape({h,1,d});
      auto k=at::cat({old.slots.at("key"),half_matmul(content.value,w.extra.at("attn_k")).reshape({1,kh,d})});
      auto v=at::cat({old.slots.at("value"),half_matmul(content.value,w.extra.at("attn_v")).reshape({1,kh,d})});
      const Index drop=node_.window?std::max<Index>(0,k.size(0)-node_.window):0;
      k=k.narrow(0,drop,k.size(0)-drop).clone();v=v.narrow(0,drop,v.size(0)-drop).clone();
      auto mapping=at::arange(h,at::kLong).div(h/kh,"floor");
      auto output=attend(q,k.transpose(0,1).index_select(0,mapping),v.transpose(0,1).index_select(0,mapping),{},1./std::sqrt(double(d)));
      return {half_matmul(output.reshape({width}),w.extra.at("attn_out")),time,increment(old.observations),{{"key",k},{"value",v}}};
    }
    std::vector<const SourceInput*> sources;
    for(const auto& source:content.sources)sources.push_back(&source);
    std::stable_sort(sources.begin(),sources.end(),[](auto a,auto b){return a->slot<b->slot;});
    std::vector<Tensor> rows;std::vector<Index> slots;
    for(auto source:sources){rows.push_back(rounded(rounded(source->atom.value)*source->scale));slots.push_back(source->slot);}
    const Index count=rows.size();
    auto qkv=rounded(half_matmul(at::stack(rows),w.extra.at("fiber_qkv"))+w.extra.at("fiber_qkv_bias")).split(width,-1);
    auto q=rounded(qkv[0].reshape({count,h,d}).transpose(0,1)*float(1./std::sqrt(double(d))));
    auto k=at::cat({old.slots.at("key"),qkv[1].reshape({count,h,d})});
    auto v=at::cat({old.slots.at("value"),qkv[2].reshape({count,h,d})});
    auto bias=old.slots.at("log_bias");
    if(bias.numel())for(Index tick=old.last_time;tick<time;++tick)bias=rounded(bias-w.extra.at("fiber_decay"));
    bias=at::cat({bias,at::zeros({count},bias.options())});
    auto y=attend(q,k.transpose(0,1),v.transpose(0,1),bias,1.).transpose(0,1).reshape({count,width});
    Tensor coefficient;
    if(pool_!="sum"&&pool_!="mean") {
      const auto weights=w.extra.at("fiber_pool").to(at::kFloat),ids=at::tensor(slots,at::kLong);
      coefficient=pool_=="linear"?weights.index_select(0,ids):pool_=="active-softmax"?
        at::softmax(weights.index_select(0,ids),0):at::softmax(weights,0).index_select(0,ids);
    }
    // Ordered FP32 reduction, followed by one half storage boundary.
    Tensor pooled;
    for(Index row=0;row<count;++row) {
      auto term=y[row].to(at::kFloat);if(coefficient.defined())term=term*coefficient[row];
      pooled=pooled.defined()?pooled+term:term;
    }
    if(pool_=="mean")pooled=pooled*(1.f/float(count));
    pooled=rounded(pooled).to(content.value.scalar_type());
    auto value=rounded(half_matmul(pooled,w.extra.at("fiber_out"))+w.extra.at("fiber_out_bias"));
    return {value,time,increment(old.observations),{{"key",k},{"value",v},{"log_bias",bias}}};
  }
};
}
std::shared_ptr<const StateKernel> half_cache_kernel(const Node& n,Index inputs) {
  if(n.identity||(n.memory!="attention"&&!is_fiber_attention_profile(n.memory)))return {};
  return std::make_shared<HalfCache>(n,inputs);
}
} // namespace tide::device_online::test
