#pragma once
#include "retained_projection.h"
#include <ATen/core/grad_mode.h>

namespace tide::device_online::test {
inline void retained_projection_check(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard guard;
  auto require=[](bool ok){if(!ok)throw std::runtime_error("retained projection ownership check failed");};
  auto reject=[&](auto action){bool failed=false;try{action();}catch(const std::logic_error&){failed=true;}require(failed);};
  auto opts=at::TensorOptions().device(device).dtype(dtype);
  auto weights=at::ones({2,4,4},opts),bias=at::ones({2,4},opts)*2;
  RetainedProjection owned,borrowed;
  owned.capture(weights,bias);borrowed.capture(weights,bias,true);
  require(!owned.retained()[0].is_same(weights)&&!owned.retained()[1].is_same(bias));
  require(borrowed.retained()[0].is_same(weights)&&borrowed.retained()[1].is_same(bias));
  const auto bytes=RetainedProjection::bytes(weights,bias);
  require(owned.reusable_bytes(weights,bias)==bytes&&borrowed.reusable_bytes(weights,bias)==bytes);
  borrowed.capture(weights,bias,true);
  reject([&]{borrowed.capture(weights,bias);});
  reject([&]{owned.capture(weights,bias,true);});
  reject([&]{borrowed.capture(weights.clone(),bias,true);});
  std::map<const void*,at::Tensor> copies;borrowed.reuse(copies,weights,bias);
  require(copies.at(weights.unsafeGetTensorImpl()).is_same(weights));
  weights.add_(3);reject([&]{borrowed.reusable_bytes(weights,bias);});
  require(owned.retained()[0].cpu().eq(1).all().item<bool>());
  // New generation rebinds only after the previous private tape is released.
  borrowed={};copies.clear();borrowed.capture(weights,bias,true);
  require(borrowed.retained()[0].cpu().eq(4).all().item<bool>());
  RetainedProjection alias;alias.capture(weights,weights,true);
  require(alias.retained()[0].is_same(alias.retained()[1]));
  require(alias.reusable_bytes(weights,weights)==int64_t(weights.nbytes()));
  RetainedProjection root;auto& shard=root.shard(0,2);shard.capture(weights,bias,true);
  require(shard.retained()[0].is_same(weights));
  reject([&]{root.shard(0,3);});
  // Default snapshots remain independently valid after their source changes.
  auto independent=owned.retained();owned={};borrowed={};root={};alias={};
  weights.fill_(9);bias.zero_();
  require(independent[0].cpu().eq(1).all().item<bool>()&&independent[1].cpu().eq(2).all().item<bool>());
}
} // namespace tide::device_online::test
