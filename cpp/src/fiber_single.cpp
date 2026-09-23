#include "tide/operator_profile.h"
// One node-local query batch, with per-query owner gather and padded KV.
#include "fiber_packing.h"
#include "fiber_pool.h"
#include "fiber_rows.h"
#include "tide/counters.h"
#include "tide/operator_work.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tide {
PackedStates fiber_attention_single(const NodeWeights& w, Index heads, const std::string& pool,
                                    const std::vector<State>& old, const PackedSequence& batch,
                                    const std::string& pooling, const std::string& ownership, const std::string& layout) {
  batch.validate();
  if (old.size() != batch.owners.size()) throw std::invalid_argument("packed initial-state count mismatch");
  op_profile::Scope profile(op_profile::InputPack);
  const auto source = fiber_rows(batch.views);
  const auto& offsets = source.offsets; const auto& slots = source.slots;
  std::vector<Index> query_owners;
  const auto& x = source.values;
  const auto width = w.bias.numel(), d = width/heads;
  profile.phase(op_profile::Qkv);
  work::linear(work::QkvCalls, x.size(0), width, 3*width);
  auto qkv = at::linear(x, w.extra.at("fiber_qkv").t(), w.extra.at("fiber_qkv_bias")).split(width, -1);
  auto q = qkv[0].reshape({-1, heads, d})*(1/std::sqrt(double(d)));
  auto k = qkv[1].reshape({-1, heads, d}), v = qkv[2].reshape({-1, heads, d});
  profile.phase(op_profile::KvBuild);
  Index maximum = 0, valid_pairs = 0;
  for (size_t i = 0; i < old.size(); ++i) {
    const auto count = offsets[batch.offsets[i+1]]-offsets[batch.offsets[i]];
    maximum = std::max(maximum, old[i].slots.at("key").size(0)+count);
    query_owners.insert(query_owners.end(), count, i);
  }
  PackedStates result; result.states.resize(batch.times.size());
  std::vector<Tensor> keys, values, bias_rows;
  for (size_t i = 0; i < old.size(); ++i) {
    const auto a = batch.offsets[i], b = batch.offsets[i+1], begin = offsets[a], end = offsets[b];
    const auto cache = old[i].slots.at("key").size(0);
    auto ks = at::cat({old[i].slots.at("key"), k.slice(0, begin, end)});
    auto vs = at::cat({old[i].slots.at("value"), v.slice(0, begin, end)});
    auto padding = at::zeros({maximum-ks.size(0), heads, d}, x.options());
    keys.push_back(at::cat({ks, padding})); values.push_back(at::cat({vs, padding}));
    auto bias = old[i].slots.at("log_bias"); auto last = old[i].last_time;
    for (auto j = a; j < b; ++j) {
      const auto time = batch.times[j], count = offsets[j+1]-offsets[j];
      if (last < -1 || time <= last) throw std::invalid_argument("fiber attention requires increasing tick times");
      if (bias.numel()) for (auto tick = last; tick < time; ++tick) bias = bias-w.extra.at("fiber_decay");
      bias = at::cat({bias, at::zeros({count}, x.options())});
      if (work::enabled()) valid_pairs += count*bias.numel();
      // Both future-event keys and KV padding are invisible. No query padding.
      auto hidden = at::full({maximum-bias.numel()}, -std::numeric_limits<double>::infinity(), x.options());
      bias_rows.push_back(at::cat({bias, hidden}).unsqueeze(0).expand({count, maximum}));
      auto key = ks.slice(0, 0, cache+offsets[j+1]-begin);
      auto value = vs.slice(0, 0, cache+offsets[j+1]-begin);
      if (j == b-1 && ownership == "cloned") { key = key.clone(); value = value.clone(); }
      result.states[j] = {Tensor(), time, increment(old[i].observations, j-a+1),
                          {{"key", key}, {"value", value}, {"log_bias", bias}}};
      last = time;
    }
    result.max_length = std::max(result.max_length, b-a);
  }
  auto owner = at::tensor(query_owners, x.options().dtype(at::kLong));
  const bool head_layout = layout == "head";
  if (head_layout) {
    for (auto& key : keys) key = key.transpose(0, 1);
    for (auto& value : values) value = value.transpose(0, 1);
  }
  work::attention(width, heads, valid_pairs, x.size(0)*maximum);
  profile.phase(op_profile::KvGather);
  auto gathered_keys = at::stack(keys).index_select(0, owner);
  if (!head_layout) gathered_keys = gathered_keys.transpose(1, 2);
  profile.phase(op_profile::Attention);
  auto scores = at::matmul(gathered_keys, q.unsqueeze(-1));
  gathered_keys = Tensor();
  scores = scores+at::cat(bias_rows).unsqueeze(1).unsqueeze(-1);
  auto probabilities = at::softmax(scores, 2).transpose(2, 3);
  profile.phase(op_profile::KvGather);
  auto gathered_values = at::stack(values).index_select(0, owner);
  if (!head_layout) gathered_values = gathered_values.transpose(1, 2);
  profile.phase(op_profile::Attention);
  auto outputs = at::matmul(probabilities, gathered_values).reshape({x.size(0), width});
  gathered_values = Tensor();
  profile.phase(op_profile::StateOther);
  std::vector<Tensor> pooled;
  if (pooling == "event") for (size_t j = 0; j < batch.times.size(); ++j)
    pooled.push_back(fiber_pool_rows(w, pool, {slots.begin()+offsets[j], slots.begin()+offsets[j+1]},
                                     outputs.slice(0, offsets[j], offsets[j+1])));
  Tensor pooled_rows;
  if (pooling == "csr") pooled_rows = fiber_pool_csr(w, pool, slots, offsets, {outputs});
  profile.phase(op_profile::Output);
  if (!pooled_rows.defined()) pooled_rows = at::stack(pooled);
  work::linear(work::OutCalls, batch.times.size(), width, width);
  auto y = at::linear(pooled_rows, w.extra.at("fiber_out").t(), w.extra.at("fiber_out_bias"));
  profile.phase(op_profile::StateCommit);
  for (size_t j = 0; j < result.states.size(); ++j) result.states[j].value = y[j].clone();
  result.calls = 1; result.max_batch = old.size(); result.score_elements = scores.numel();
  return result;
}
}  // namespace tide
