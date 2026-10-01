#include "retained_cache_fixture.h"
#include "full_vjp_fixture.h"
#include <algorithm>
#include <array>
#include <limits>
#include <set>

namespace tide::device_online::test {
Fixture retained_cache_fixture(int shape,int variant,Index width,int profile,bool clock) {
  if(profile<0||profile>6)throw std::invalid_argument("invalid retained cache profile");
  auto f=retained_fixture(shape,variant,width);auto& g=f.graph;
  if(variant) {
    for(size_t i=0;i<g.inputs.size();++i)g.source_domain->input[i]=g.source_counts[g.inputs[i]]-1-g.source_domain->input[i];
    for(size_t i=0;i<g.edges.size();++i)g.source_domain->edge_target[i]=g.source_counts[g.edges[i].target]-1-g.source_domain->edge_target[i];
    g.compile();
  }
  const std::array<std::string,5> pools{"sum","mean","linear","active-softmax","all-softmax"};
  const int pool=profile==6?3:profile-1;
  for(Index n:{0,2,3}) {
    const bool event=profile==0||(profile==6&&n==3);
    auto& node=g.nodes[n];auto& w=f.model.nodes[n];node.identity=false;
    node.query_heads=width==4?(n==3?2:4):1;node.kv_heads=event&&width==4?node.query_heads/2:node.query_heads;
    node.memory=event?"attention":"lh-fiber-attention-"+pools[pool]+"-repeat-v1";
    node.window=event?(n==0?(variant?1:3):n==2?0:2):0;
    node.clear=variant&&n!=3;auto eye=at::eye(width,at::kFloat);w.extra.clear();
    const Index kv=width/node.query_heads*node.kv_heads;
    if(event)w.extra={{"attn_q",eye*.25f},{"attn_k",eye.narrow(1,0,kv).clone()*.5f},
      {"attn_v",eye.narrow(1,0,kv).clone()*.75f},{"attn_out",eye*.5f}};
    else {
      w.extra={{"fiber_qkv",at::cat({eye*.25f,eye*.5f,eye*.75f},1)},
        {"fiber_qkv_bias",at::arange(3*width,at::kFloat).remainder(5)*.00390625f},
        {"fiber_out",eye*.5f},{"fiber_out_bias",at::full({width},.03125f)},
        {"fiber_decay",at::full({},.03125f)}};
      if(pool>=2)w.extra["fiber_pool"]=at::arange(g.source_counts[n],at::kFloat)*.125f-.125f;
    }
    for(Index b=0;b<f.initial.batch_size;++b) {
      const Index rows=b?0:event?(node.window==1?1:2):3;
      auto& s=f.initial.states.at({b,n});s.observations=std::max(s.observations,rows);
      s.slots={{"key",at::arange(rows*kv,at::kFloat).remainder(7).reshape({rows,node.kv_heads,width/node.query_heads})/32.},
        {"value",at::arange(rows*kv,at::kFloat).remainder(11).reshape({rows,node.kv_heads,width/node.query_heads})/16.}};
      if(!event)s.slots["log_bias"]=at::arange(rows,at::kFloat)/32.;
    }
  }
  if(profile==0) {
    f.model.nodes[2].extra["attn_q"]=f.model.nodes[0].extra.at("attn_q");
    f.model.nodes[2].extra["attn_v"]=f.model.nodes[0].extra.at("attn_k");
  } else {
    f.model.nodes[2].extra["fiber_qkv"]=f.model.nodes[0].extra.at("fiber_qkv");
    f.model.nodes[2].extra["fiber_out"]=f.model.nodes[0].weight;
    f.model.nodes[2].extra["fiber_decay"]=f.model.nodes[0].extra.at("fiber_decay");
  }
  if(clock) {
    g.nodes[0].state_clock={5,1,3};
    for(Index b=0;b<f.initial.batch_size;++b)f.initial.states.at({b,0}).last_time=-1;
    for(auto& x:f.input)if(x.port==0)x.time=x.time==0?1:x.time==1?3:6;
  }
  g.compile();f.initial.identity=g.identity;return f;
}
void retained_cache_roots(GraphCotangents& r,const ReverseTape& t,int window,int mode) {
  if(mode!=9&&!(window==3&&(mode>=6&&mode<=8||mode==10)))return;
  const float scale=t.source_scales.scalar_type()==at::kHalf?256.f:1.f;
  const auto groups=t.attention.size()+t.fiber.size();
  for(size_t i=0;i<groups;++i) {
    const bool fiber=i>=t.attention.size();
    const auto& a=fiber?t.fiber[i-t.attention.size()].cache:t.attention[i];
    const auto opts=a.values.options().dtype(at::kFloat);
    auto padding=at::arange(a.capacity,a.lengths.options()).unsqueeze(0)>=a.lengths.unsqueeze(1);
    CacheCotangents c;
    auto add=[&](Tensor& x,Tensor& on,const Tensor& shape,float value) {
      x=at::full(shape.sizes(),mode==8?0.f:value*scale,opts);
      x.masked_fill_(x.dim()==2?padding:padding.unsqueeze(-1).unsqueeze(-1),std::numeric_limits<float>::quiet_NaN());
      on=at::ones(a.lengths.sizes(),opts.dtype(at::kBool));
    };
    if(mode!=7&&mode!=10)add(c.key,c.key_connected,a.key,.0078125f);
    if(mode!=6&&mode!=10)add(c.value,c.value_connected,a.value,-.015625f);
    if(fiber&&(mode==8||mode==9||mode==10))add(c.bias,c.bias_connected,t.fiber[i-t.attention.size()].bias,.0234375f);
    r.cache.push_back(c);
  }
}
void compare_retained_cache(const GraphVjp& g,const ReverseTape& t,const RetainedReference& ref,const Fixture& f,bool half) {
  if(g.cache.size()!=t.attention.size()+t.fiber.size())throw std::runtime_error("retained cache gradient groups differ");
  std::set<std::string> seen;
  for(size_t i=0;i<g.cache.size();++i) {
    const auto& a=i<t.attention.size()?t.attention[i]:t.fiber[i-t.attention.size()].cache;
    const auto& c=g.cache[i];auto lengths=c.lengths.cpu();
    const std::array<std::pair<Tensor,Tensor>,3> pairs{{{c.key,c.key_connected},{c.value,c.value_connected},{c.bias,c.bias_connected}}};
    const std::array<std::string,3> names{"key","value","log_bias"};
    for(size_t which=0;which<3;++which) {
      if(which==2&&i<t.attention.size())continue;
      auto value=pairs[which].first.cpu(),on=pairs[which].second.cpu();
      if(value.scalar_type()!=at::kFloat)throw std::runtime_error("retained cache cotangent lost FP32 precision");
      for(Index b=0;b<f.initial.batch_size;++b)for(size_t n=0;n<a.nodes.size();++n) {
        const auto& s=f.initial.states.at({b,a.nodes[n]});const auto size=s.slots.at("key").size(0);
        const Index owner=b*a.nodes.size()+n;
        if(lengths[owner].item<Index>()!=size)throw std::runtime_error("retained cache length changed");
        const auto leaf="cache/"+names[which]+"/"+std::to_string(b)+"/"+std::to_string(a.nodes[n]);
        seen.insert(leaf);full_same_precision(value[owner].narrow(0,0,size),on[owner],ref.gradients.at(leaf),leaf.c_str(),half);
        if(value[owner].narrow(0,size,a.capacity-size).count_nonzero().item<Index>())throw std::runtime_error("cache padding acquired gradients");
      }
    }
  }
  for(const auto& [name,_]:ref.gradients)if(name.rfind("cache/",0)==0&&!seen.count(name))throw std::runtime_error("retained reverse lost cache leaf");
}
void poison_live_cache(ReverseTape& t) {
  const float nan=std::numeric_limits<float>::quiet_NaN();
  auto poison=[&](EventAttentionTape& a){for(auto x:{a.qkv,a.projection,a.values,a.key,a.value})x.fill_(nan);a.count.fill_(-1);};
  for(auto& a:t.attention)poison(a);
  for(auto& f:t.fiber){poison(f.cache);for(auto x:{f.bias,f.qkv_bias,f.projection_bias,f.decay,f.pool_weights})if(x.defined())x.fill_(nan);}
}
std::vector<Tensor> cache_gradient_values(const GraphVjp& g) {
  std::vector<Tensor> out;
  for(const auto& a:g.cache)for(const auto& x:{a.key,a.value,a.bias,a.key_connected,a.value_connected,a.bias_connected,a.lengths})
    if(x.defined())out.push_back(x.cpu().clone());
  return out;
}
} // namespace tide::device_online::test
