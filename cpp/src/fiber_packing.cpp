#include "tide/operator_work.h"
// Event/source offsets preserve all current-fiber keys without cross-sample scores.
#include "fiber_packing.h"
#include "fiber_pool.h"
#include "tide/counters.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tide {
namespace {
struct FiberRows {
  Tensor values;
  std::vector<Index> offsets{0};  // event -> source rows; original tags stay in views
  std::vector<Index> slots;
};
FiberRows flatten(const ContentViews& views) {
  FiberRows result; std::vector<Tensor> rows;
  for (const auto& view : views) {
    if (view.sources.empty()) throw std::invalid_argument("fiber attention requires complete source rows");
    std::vector<const SourceInput*> sources;
    for (const auto& source : view.sources) sources.push_back(&source);
    std::sort(sources.begin(), sources.end(), [](auto a, auto b) { return a->slot < b->slot; });
    for (auto source : sources) {
      rows.push_back(source->atom.value*source->scale); result.slots.push_back(source->slot);
    }
    result.offsets.push_back(rows.size());
  }
  result.values = at::stack(rows); return result;
}
Tensor repeat_bias(Tensor bias, const Tensor& rate, Index last, Index target) {
  if (last < -1 || target <= last) throw std::invalid_argument("fiber attention requires increasing tick times");
  if (bias.numel()) for (auto tick = last; tick < target; ++tick) bias = bias-rate;
  return bias;
}
}  // namespace
PackedStates fiber_attention_packed(const NodeWeights& w, Index heads, const std::string& pool, const std::vector<State>& old,
                                   const PackedSequence& batch) {
  batch.validate();
  if (old.size() != batch.owners.size()) throw std::invalid_argument("packed initial-state count mismatch");
  const auto source = flatten(batch.views); const auto& offsets = source.offsets;
  const auto width = w.bias.numel(), d = width/heads;
  work::linear(work::QkvCalls, source.values.size(0), width, 3*width);
  auto qkv = at::linear(source.values, w.extra.at("fiber_qkv").t(), w.extra.at("fiber_qkv_bias")).split(width, -1);
  auto q = qkv[0].reshape({-1, heads, d})*(1/std::sqrt(double(d)));
  auto k = qkv[1].reshape({-1, heads, d}), v = qkv[2].reshape({-1, heads, d});
  std::map<Owner, std::vector<Index>> groups;  // initial cache rows, total query rows
  for (size_t i = 0; i < old.size(); ++i)
    groups[{old[i].slots.at("key").size(0), offsets[batch.offsets[i+1]]-offsets[batch.offsets[i]]}].push_back(i);
  PackedStates result; result.states.resize(batch.times.size()); std::vector<Tensor> pooled(batch.times.size());
  for (const auto& [shape, ids] : groups) {
    const auto cache = shape.first, queries = shape.second, rows = static_cast<Index>(ids.size());
    std::vector<Tensor> qs, ks, vs, bs, ms;
    Index valid_pairs = 0;
    auto key_index = at::arange(cache+queries, source.values.options().dtype(at::kLong));
    for (auto i : ids) {
      const auto a = batch.offsets[i], b = batch.offsets[i+1], begin = offsets[a], end = offsets[b];
      qs.push_back(q.slice(0, begin, end));
      ks.push_back(at::cat({old[i].slots.at("key"), k.slice(0, begin, end)}));
      vs.push_back(at::cat({old[i].slots.at("value"), v.slice(0, begin, end)}));
      auto bias = old[i].slots.at("log_bias"); auto last = old[i].last_time;
      std::vector<Tensor> bias_rows, masks;
      for (auto j = a; j < b; ++j) {
        const auto time = batch.times[j], count = offsets[j+1]-offsets[j];
        bias = at::cat({repeat_bias(bias, w.extra.at("fiber_decay"), last, time), at::zeros({count}, source.values.options())});
        const auto full = cache+queries;
        if (work::enabled()) valid_pairs += count*bias.numel();
        auto padded = at::cat({bias, at::zeros({full-bias.numel()}, source.values.options())});
        bias_rows.push_back(padded.unsqueeze(0).expand({count, full}));
        masks.push_back((key_index < bias.numel()).unsqueeze(0).expand({count, full})); last = time;
        result.states[j] = {Tensor(), time, increment(old[i].observations, j-a+1), {{"log_bias", bias}}};
      }
      bs.push_back(at::cat(bias_rows)); ms.push_back(at::cat(masks));
      result.max_length = std::max(result.max_length, b-a);
    }
    auto keys = at::stack(ks), values = at::stack(vs);
    work::attention(width, heads, valid_pairs, rows*queries*(cache+queries));
    auto scores = at::matmul(at::stack(qs).transpose(1, 2), keys.permute({0, 2, 3, 1}));
    scores = scores+at::stack(bs).unsqueeze(1);
    auto probabilities = at::softmax(scores.masked_fill(~at::stack(ms).unsqueeze(1), -std::numeric_limits<double>::infinity()), -1);
    auto outputs = at::matmul(probabilities, values.transpose(1, 2)).transpose(1, 2).reshape({rows, queries, width});
    for (Index row = 0; row < rows; ++row) {
      const auto i = ids[row], a = batch.offsets[i], b = batch.offsets[i+1], start = offsets[a];
      for (auto j = a; j < b; ++j) {
        const auto begin = offsets[j]-start, end = offsets[j+1]-start;
        pooled[j] = fiber_pool_rows(w, pool, {source.slots.begin()+offsets[j], source.slots.begin()+offsets[j+1]},
                                   outputs[row].slice(0, begin, end));
        auto key_slot = keys[row].slice(0, 0, cache+end), value_slot = values[row].slice(0, 0, cache+end);
        if (j == b-1) { key_slot = key_slot.clone(); value_slot = value_slot.clone(); }
        result.states[j].slots.emplace("key", key_slot); result.states[j].slots.emplace("value", value_slot);
      }
    }
    ++result.calls; result.max_batch = std::max(result.max_batch, rows); result.score_elements += scores.numel();
  }
  work::linear(work::OutCalls, pooled.size(), width, width);
  auto y = at::linear(at::stack(pooled), w.extra.at("fiber_out").t(), w.extra.at("fiber_out_bias"));
  for (size_t j = 0; j < result.states.size(); ++j) result.states[j].value = y[j].clone();
  return result;
}
}  // namespace tide
