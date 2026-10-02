#pragma once
#include "retained_full.h"
#include <ATen/core/grad_mode.h>

namespace tide::device_online::test {
inline void retained_full_check(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard guard;
  auto require=[](bool yes){if(!yes)throw std::runtime_error("immutable Full snapshot check failed");};
  auto reject=[&](auto action){bool failed=false;try{action();}catch(const std::logic_error&){failed=true;}require(failed);};
  const auto f=at::TensorOptions().device(device).dtype(dtype),l=f.dtype(at::kLong);
  FullTape live{};live.samples=2;live.width=4;live.has_tanh=true;
  live.kinds=at::ones({3},l);live.weights=at::ones({4,4,4},f);live.biases=at::ones({4,4},f)*2;
  live.extra.lh_kinds=at::ones({3},l);live.extra.lh_weights=at::ones({4,4},f)*3;
  live.extra.lh_biases=live.extra.lh_weights; // A repeated TensorImpl counts once.
  live.extra.swiglu_kinds=at::ones({3},l);live.extra.swiglu_mapping=at::ones({3},l);
  live.extra.gate=at::ones({2,4,8},f)*4;live.extra.up=at::ones({2,4,8},f)*5;live.extra.down=at::ones({2,8,4},f)*6;
  live.metadata=at::zeros({16,4},l);live.values=at::zeros({16,4},f);live.count=at::zeros({1},l);
  const int64_t expected=4*3*8+(4*4*4+4*4+4*4+3*2*4*8)*live.weights.element_size();
  require(RetainedFull::bytes(live)==expected);
  RetainedFull snapshot;require(snapshot.reusable_bytes(live)==0);snapshot.capture(live);
  RetainedFull::Copies first,second;snapshot.reuse(first,live);
  require(first.size()==10&&!first.count(live.values.unsafeGetTensorImpl())&&!first.count(live.metadata.unsafeGetTensorImpl())
    &&!first.count(live.count.unsafeGetTensorImpl()));
  auto next=live;next.values=at::ones_like(live.values);next.metadata=at::ones_like(live.metadata);next.count=at::ones_like(live.count);
  snapshot.validate(next);snapshot.capture(next);snapshot.reuse(second,next);
  require(snapshot.reusable_bytes(next)==expected);
  for(const auto& [key,value]:first)require(value.is_same(second.at(key)));
  require(!first.at(live.weights.unsafeGetTensorImpl()).is_same(live.weights));
  auto changed=next;changed.weights=next.weights.clone();reject([&]{snapshot.validate(changed);});
  changed=next;changed.extra.up=at::Tensor{};reject([&]{snapshot.reuse(second,changed);});
  changed=next;changed.samples=3;reject([&]{snapshot.capture(changed);});
  changed=next;changed.extra.gate=next.extra.gate.to(dtype==at::kHalf?at::kFloat:at::kHalf);reject([&]{snapshot.validate(changed);});
  live.weights.fill_(9);reject([&]{snapshot.validate(live);});reject([&]{snapshot.reuse(second,next);});
  require(first.at(live.weights.unsafeGetTensorImpl()).cpu().eq(1).all().item<bool>());
  snapshot={};snapshot.capture(live);RetainedFull::Copies refreshed;snapshot.reuse(refreshed,live);
  require(refreshed.at(live.weights.unsafeGetTensorImpl()).cpu().eq(9).all().item<bool>());
  const auto saved=first.at(live.weights.unsafeGetTensorImpl());snapshot={};live={};next={};
  require(saved.cpu().eq(1).all().item<bool>());
  FullTape identity{};identity.samples=1;identity.width=4;identity.kinds=at::zeros({2},l);
  snapshot.capture(identity);require(snapshot.reusable_bytes(identity)==16);
  snapshot.shard(0,2);reject([&]{snapshot.shard(0,3);});
}
} // namespace tide::device_online::test
