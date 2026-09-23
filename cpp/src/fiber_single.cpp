// One node-local query batch, with per-query owner gather and padded KV.
#include "fiber_packing.h"
#include "fiber_pool.h"
#include "tide/counters.h"
#include "tide/operator_work.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tide {
PackedStates fiber_attention_single(const NodeWeights& w, Index heads, const std::string& pool,
                                    const std::vector<State>& old, const PackedSequence& batch) {
  batch.validate();
  if (old.size() != batch.owners.size()) throw std::invalid_argument("packed initial-state count mismatch");
  std::vector<Tensor> inputs;
  std::vector<Index> offsets{0}, slots, query_owners;
  for (const auto& view : batch.views) {
    if (view.sources.empty()) throw std::invalid_argument("fiber attention requires complete source rows");
    std::vector<const SourceInput*> sources;
    for (const auto& source : view.sources) sources.push_back(&source);
    std::sort(sources.begin(), sources.end(), [](auto a, auto b) { return a->slot < b->slot; });
    for (auto source : sources) { inputs.push_back(source->atom.value*source->scale); slots.push_back(source->slot); }
    offsets.push_back(inputs.size());
  }
  auto x = at::stack(inputs);
  const auto width = w.bias.numel(), d = width/heads;
  work::linear(work::QkvCalls, x.size(0), width, 3*width);
  auto qkv = at::linear(x, w.extra.at("fiber_qkv").t(), w.extra.at("fiber_qkv_bias")).split(width, -1);
  auto q = qkv[0].reshape({-1, heads, d})*(1/std::sqrt(double(d)));
  auto k = qkv[1].reshape({-1, heads, d}), v = qkv[2].reshape({-1, heads, d});
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
      if (j == b-1) { key = key.clone(); value = value.clone(); }
      result.states[j] = {Tensor(), time, increment(old[i].observations, j-a+1),
                          {{"key", key}, {"value", value}, {"log_bias", bias}}};
      last = time;
    }
    result.max_length = std::max(result.max_length, b-a);
  }
  auto owner = at::tensor(query_owners, x.options().dtype(at::kLong));
  work::attention(width, heads, valid_pairs, x.size(0)*maximum);
  auto scores = at::matmul(at::stack(keys).index_select(0, owner).transpose(1, 2), q.unsqueeze(-1));
  scores = scores+at::cat(bias_rows).unsqueeze(1).unsqueeze(-1);
  auto probabilities = at::softmax(scores, 2).transpose(2, 3);
  auto outputs = at::matmul(probabilities, at::stack(values).index_select(0, owner).transpose(1, 2)).reshape({x.size(0), width});
  std::vector<Tensor> pooled;
  for (size_t j = 0; j < batch.times.size(); ++j)
    pooled.push_back(fiber_pool_rows(w, pool, {slots.begin()+offsets[j], slots.begin()+offsets[j+1]},
                                     outputs.slice(0, offsets[j], offsets[j+1])));
  work::linear(work::OutCalls, pooled.size(), width, width);
  auto y = at::linear(at::stack(pooled), w.extra.at("fiber_out").t(), w.extra.at("fiber_out_bias"));
  for (size_t j = 0; j < result.states.size(); ++j) result.states[j].value = y[j].clone();
  result.calls = 1; result.max_batch = old.size(); result.score_elements = scores.numel();
  return result;
}
}  // namespace tide
