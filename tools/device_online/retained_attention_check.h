#pragma once
#include "retained_attention.h"
#include <ATen/core/grad_mode.h>

namespace tide::device_online::test {
inline void retained_attention_check(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard guard;
  auto require=[](bool yes){if(!yes)throw std::runtime_error("immutable attention snapshot check failed");};
  auto reject=[&](auto action){bool failed=false;try{action();}catch(const std::logic_error&){failed=true;}require(failed);};
  auto opts=at::TensorOptions().device(device).dtype(dtype);
  FiberParameterBanks live;live.qkv=at::ones({3,4,12},opts);live.projection=at::ones({3,4,4},opts)*2;
  live.qkv_bias=at::ones({3,12},opts)*3;live.projection_bias=at::ones({3,4},opts)*4;
  live.decay=at::ones({2},opts)*5;live.pool=at::ones({3,2},opts.dtype(at::kFloat))*6;
  auto grouped=[&] {
    // Fresh gathers on each call, as with interleaved mixed-head groups.
    auto ids=at::tensor({1,0,2},at::kLong).to(device);FiberAttentionTape f;
    f.cache.nodes={1,3};f.cache.heads=f.cache.kv_heads=2;f.cache.width=4;
    f.cache.qkv=live.qkv.index_select(0,ids);f.cache.projection=live.projection.index_select(0,ids);
    f.qkv_bias=live.qkv_bias.index_select(0,ids);f.projection_bias=live.projection_bias.index_select(0,ids);
    f.decay=live.decay.clone();f.pool_weights=live.pool.index_select(0,ids);
    f.cache.key=at::ones({2,3,2,2},opts);f.cache.value=at::ones_like(f.cache.key);f.bias=at::ones({2,3},opts);
    return std::vector<FiberAttentionTape>{f};
  };
  RetainedAttention cache;auto first=grouped();reject([&]{cache.capture({},first);});
  cache.bind({},live);require(cache.reusable_bytes({},first)==0);cache.capture({},first);
  auto second=grouped();cache.bind({},live);
  RetainedAttention::Copies a,b;cache.reuse(a,{},first);cache.reuse(b,{},second);
  require(a.at(first[0].cache.qkv.unsafeGetTensorImpl()).is_same(b.at(second[0].cache.qkv.unsafeGetTensorImpl())));
  require(!a.count(first[0].bias.unsafeGetTensorImpl())&&!a.count(first[0].cache.key.unsafeGetTensorImpl()));
  require(cache.reusable_bytes({},second)==RetainedAttention::bytes({},first));
  auto changed=second;changed[0].cache.nodes={0,3};reject([&]{cache.reuse(b,{},changed);});
  changed=second;changed[0].cache.qkv=changed[0].cache.qkv.to(dtype==at::kHalf?at::kFloat:at::kHalf);
  reject([&]{cache.reuse(b,{},changed);});
  auto replaced=live;replaced.qkv=live.qkv.clone();reject([&]{cache.bind({},replaced);});
  first[0].cache.qkv.fill_(77);require(a.at(first[0].cache.qkv.unsafeGetTensorImpl()).cpu().eq(1).all().item<bool>());
  live.qkv.fill_(9);reject([&]{cache.bind({},live);});reject([&]{cache.reuse(b,{},second);});
  // Dropping an update cache never mutates tapes already returned to callers.
  cache={};cache.bind({},live);auto third=grouped();cache.capture({},third);RetainedAttention::Copies c;cache.reuse(c,{},third);
  require(c.at(third[0].cache.qkv.unsafeGetTensorImpl()).cpu().eq(9).all().item<bool>());
  require(a.at(first[0].cache.qkv.unsafeGetTensorImpl()).cpu().eq(1).all().item<bool>());
  cache={};live={};require(b.at(second[0].cache.qkv.unsafeGetTensorImpl()).cpu().eq(1).all().item<bool>());
}
} // namespace tide::device_online::test
