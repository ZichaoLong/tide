#include "tide/counters.h"
#include "tide/kernel.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tide {
namespace {
class AttentionKernel final : public StateKernel {
 public:
  AttentionKernel(Index query_heads, Index kv_heads, Index window)
      : query_heads_(query_heads), kv_heads_(kv_heads), window_(window) {}
  State initial(const NodeWeights& w) const override {
    auto shape = std::vector<Index>{0, kv_heads_, w.bias.numel() / query_heads_};
    return {at::zeros_like(w.bias), -1, 0, {{"key", at::zeros(shape, w.bias.options())},
                                          {"value", at::zeros(shape, w.bias.options())}}};
  }
  State step(const NodeWeights& w, const State& old, const ContentView& content, Index time) const override {
    return batch(w, {old}, content.value.unsqueeze(0), {time}, {content})[0];
  }
  std::vector<State> batch(const NodeWeights& w, const std::vector<State>& old, const Tensor& h,
                           const std::vector<Index>& times, const ContentViews& views) const override {
    PackedSequence p; p.contents = h; p.times = times; p.views = views;
    for (size_t i = 0; i < old.size(); ++i) { p.offsets.push_back(i + 1); p.owners.emplace_back(i, 0); }
    return packed_sequence(w, old, p).states;
  }
  bool exact_sequence() const override { return true; }
  bool joint_batch() const override { return true; }
  bool joint_sequence() const override { return true; }
  std::vector<State> sequence(const NodeWeights& w, const State& old, const Tensor& h,
                              const std::vector<Index>& times, const ContentViews& views) const override {
    PackedSequence p{h, {0, h.size(0)}, {{0, 0}}, times, views};
    return packed_sequence(w, {old}, p).states;
  }
  PackedStates packed_sequence(const NodeWeights& w, const std::vector<State>& old,
                                const PackedSequence& p) const override {
    p.validate();
    if (old.size() != p.owners.size()) throw std::invalid_argument("packed initial-state count mismatch");
    const auto width = w.bias.numel(), d = width / query_heads_;
    auto q = at::matmul(p.contents, w.extra.at("attn_q")).reshape({-1, query_heads_, d});
    auto k = at::matmul(p.contents, w.extra.at("attn_k")).reshape({-1, kv_heads_, d});
    auto v = at::matmul(p.contents, w.extra.at("attn_v")).reshape({-1, kv_heads_, d});
    std::map<Owner, std::vector<Index>> groups;  // cache length, sequence length
    for (size_t i = 0; i < old.size(); ++i)
      groups[{old[i].slots.at("key").size(0), p.offsets[i + 1] - p.offsets[i]}].push_back(i);
    PackedStates result; result.states.resize(p.times.size());
    for (const auto& [shape, ids] : groups) {
      const auto cache = shape.first, length = shape.second, rows = static_cast<Index>(ids.size());
      std::vector<Tensor> qs, ks, vs;
      for (auto i : ids) {
        const auto a = p.offsets[i], b = p.offsets[i + 1];
        qs.push_back(q.slice(0, a, b));
        ks.push_back(at::cat({old[i].slots.at("key"), k.slice(0, a, b)}));
        vs.push_back(at::cat({old[i].slots.at("value"), v.slice(0, a, b)}));
      }
      auto keys = at::stack(ks), values = at::stack(vs);
      const auto group = query_heads_ / kv_heads_;
      auto scores = at::matmul(at::stack(qs).transpose(1, 2),
                               at::repeat_interleave(keys, group, 2).permute({0, 2, 3, 1})) / std::sqrt(d);
      auto ends = cache + at::arange(length, q.options().dtype(at::kLong)).unsqueeze(1);
      auto indexes = at::arange(cache + length, q.options().dtype(at::kLong)).unsqueeze(0);
      auto mask = indexes <= ends;
      if (window_) mask = mask & (indexes > ends - window_);
      auto probabilities = at::softmax(scores.masked_fill(~mask, -std::numeric_limits<double>::infinity()), -1);
      auto value = at::matmul(probabilities, at::repeat_interleave(values, group, 2).transpose(1, 2));
      value = at::matmul(value.transpose(1, 2).reshape({rows, length, width}), w.extra.at("attn_out"));
      for (Index row = 0; row < rows; ++row) {
        const auto i = ids[row];
        for (Index t = 0; t < length; ++t) {
          const auto end = cache + t + 1, begin = window_ ? std::max<Index>(0, end - window_) : 0;
          auto key_slot = keys[row].slice(0, begin, end), value_slot = values[row].slice(0, begin, end);
          // Compact the persistent final cache; intermediate trace views share the block storage.
          if (t == length - 1) { key_slot = key_slot.clone(); value_slot = value_slot.clone(); }
          const auto j = p.offsets[i] + t;
          auto read_value = t == length - 1 ? value[row][t].clone() : value[row][t];
          result.states[j] = {read_value, p.times[j], increment(old[i].observations, t + 1),
                              {{"key", key_slot}, {"value", value_slot}}};
        }
      }
      ++result.calls; result.max_batch = std::max(result.max_batch, rows);
      result.max_length = std::max(result.max_length, length);
      result.score_elements += scores.numel();
    }
    return result;
  }
  State reset(const State& state) const override {
    auto s = state; s.value = s.value * 0;
    for (auto& [name, tensor] : s.slots) tensor = tensor.slice(0, 0, 0).clone();
    return s;
  }
  void validate_weights(const NodeWeights& w) const override {
    const auto d = w.bias.numel();
    if (query_heads_ < 1 || kv_heads_ < 1 || query_heads_ % kv_heads_ || d % query_heads_ || window_ < 0)
      throw std::invalid_argument("invalid attention heads/window/width");
    for (const auto& name : {"attn_q", "attn_k", "attn_v", "attn_out"}) {
      auto it = w.extra.find(name);
      const Index out = std::string(name) == "attn_k" || std::string(name) == "attn_v" ? d / query_heads_ * kv_heads_ : d;
      if (it == w.extra.end() || it->second.sizes() != at::IntArrayRef({d, out}))
        throw std::invalid_argument("invalid attention projection weights");
    }
  }
  void validate_state(const NodeWeights& w, const State& s) const override {
    if (s.slots.size() != 2 || !s.slots.count("key") || !s.slots.count("value"))
      throw std::invalid_argument("attention requires key/value slots");
    const auto& k = s.slots.at("key"); const auto& v = s.slots.at("value");
    if (k.dim() != 3 || k.size(1) != kv_heads_ || k.size(2) != w.bias.numel() / query_heads_
        || k.sizes() != v.sizes() || k.size(0) > s.observations || (window_ && k.size(0) > window_))
      throw std::invalid_argument("invalid attention cache shape/length");
  }
 private:
  Index query_heads_, kv_heads_, window_;
};
}  // namespace
std::shared_ptr<const StateKernel> make_attention_kernel(Index q, Index kv, Index window) {
  return std::make_shared<AttentionKernel>(q, kv, window);
}
}  // namespace tide
