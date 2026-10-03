#pragma once
#include "retained_attention.h"
#include <ATen/core/grad_mode.h>

namespace tide::device_online::test {
inline void retained_attention_check(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard guard;
  auto require=[](bool yes,const char* why="default snapshot"){if(!yes)throw std::runtime_error(std::string("immutable attention snapshot check failed: ")+why);};
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
  cache={};require(b.at(second[0].cache.qkv.unsafeGetTensorImpl()).cpu().eq(1).all().item<bool>());
  // A complete ordered group may use the private bank, while subset/mixed-head
  // gathers retain independent copies. Dynamic KV/bias never enters the cache.
  live.nodes={1,3};live.qkv=at::arange(144,opts).reshape({3,4,12});
  auto whole=grouped();whole[0].cache.qkv=live.qkv.clone();
  whole[0].cache.projection=live.projection.clone();whole[0].qkv_bias=live.qkv_bias.clone();
  whole[0].projection_bias=live.projection_bias.clone();whole[0].decay=live.decay.clone();whole[0].pool_weights=live.pool.clone();
  RetainedAttention borrowed;borrowed.bind({},live,true);borrowed.capture({},whole);
  RetainedAttention::Copies shared;borrowed.reuse(shared,{},whole);
  require(shared.at(whole[0].cache.qkv.unsafeGetTensorImpl()).is_same(live.qkv),"whole-bank QKV identity");
  require(borrowed.borrowed_bytes()==RetainedAttention::bytes({},whole),"whole-bank borrowed footprint");
  require(!shared.count(whole[0].cache.key.unsafeGetTensorImpl())&&!shared.count(whole[0].bias.unsafeGetTensorImpl()));
  reject([&]{borrowed.bind({},live);});
  auto changed_nodes=live;changed_nodes.nodes={3,1};reject([&]{borrowed.bind({},changed_nodes,true);});
  live.qkv.add_(1);reject([&]{borrowed.reuse(shared,{},whole);});
  borrowed={};borrowed.bind({},live,true);whole[0].cache.qkv=live.qkv.clone();
  borrowed.capture({},whole);RetainedAttention::Copies fresh;borrowed.reuse(fresh,{},whole);
  require(fresh.at(whole[0].cache.qkv.unsafeGetTensorImpl()).is_same(live.qkv));
  // Declare an actual subset so bank geometry/order cannot be substituted.
  borrowed={};auto subset=whole;subset[0].cache.nodes={3};
  auto subset_rows=at::tensor({1,2},at::kLong).to(device);
  subset[0].cache.qkv=live.qkv.index_select(0,subset_rows);subset[0].cache.projection=live.projection.index_select(0,subset_rows);
  subset[0].qkv_bias=live.qkv_bias.index_select(0,subset_rows);subset[0].projection_bias=live.projection_bias.index_select(0,subset_rows);
  subset[0].decay=live.decay.narrow(0,1,1).clone();subset[0].pool_weights=live.pool.index_select(0,subset_rows);
  borrowed.bind({},live,true);borrowed.capture({},subset);
  fresh.clear();borrowed.reuse(fresh,{},subset);
  require(!fresh.at(subset[0].cache.qkv.unsafeGetTensorImpl()).is_same(live.qkv)&&borrowed.borrowed_bytes()==0);
  subset[0].cache.qkv.zero_();require(fresh.at(subset[0].cache.qkv.unsafeGetTensorImpl()).cpu().ne(0).any().item<bool>());
  // Event matrices are already the frozen bank tensors; preserve their aliases.
  EventAttentionTape event;event.qkv=live.qkv;event.projection=live.projection;
  RetainedAttention event_borrow;event_borrow.bind({event},{},true);event_borrow.capture({event},{});
  fresh.clear();event_borrow.reuse(fresh,{event},{});
  require(fresh.at(event.qkv.unsafeGetTensorImpl()).is_same(live.qkv));
  require(event_borrow.borrowed_bytes()==RetainedAttention::bytes({event},{}));
  borrowed={};event_borrow={};shared.clear();fresh.clear();live={};
}
} // namespace tide::device_online::test
