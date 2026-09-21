#include "tide/fiber_attention.h"
#include "tide/counters.h"
#include "fiber_packing.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tide {
namespace {
Tensor advance_bias(Tensor bias, const Tensor& rate, Index last, Index target) {
  if (last < -1 || target < last) throw std::invalid_argument("invalid fiber attention tick clock");
  if (bias.numel()) for (auto tick = last; tick < target; ++tick) bias = bias-rate;
  return bias;
}
class FiberAttention final : public StateKernel {
  Index heads_;
 public:
  explicit FiberAttention(const Node& n) : heads_(n.query_heads) {
    if (heads_ < 1 || n.kv_heads != heads_ || n.window || n.aggregation != "sum")
      throw std::invalid_argument("LH fiber attention requires equal heads, no eviction and sum Aggregate");
  }
  State initial(const NodeWeights& w) const override {
    auto shape = std::vector<Index>{0, heads_, w.bias.numel()/heads_};
    return {at::zeros_like(w.bias), -1, 0, {{"key", at::zeros(shape, w.bias.options())},
            {"value", at::zeros(shape, w.bias.options())}, {"log_bias", at::zeros({0}, w.bias.options())}}};
  }
  State step(const NodeWeights& w, const State& old, const ContentView& content, Index time) const override {
    if (content.sources.empty()) throw std::invalid_argument("LH fiber attention requires complete source rows");
    if (time < 0 || old.last_time < -1 || time <= old.last_time)
      throw std::invalid_argument("fiber attention requires increasing tick times");
    auto observations = increment(old.observations);
    std::vector<const SourceInput*> sources;
    for (const auto& source : content.sources) sources.push_back(&source);
    std::sort(sources.begin(), sources.end(), [](auto a, auto b) { return a->slot < b->slot; });
    std::vector<Tensor> rows;
    for (auto s : sources) rows.push_back(s->atom.value*s->scale);
    auto x = at::stack(rows); const auto width = w.bias.numel(), d = width/heads_;
    auto qkv = at::linear(x, w.extra.at("fiber_qkv").t(), w.extra.at("fiber_qkv_bias")).split(width, -1);
    auto q = qkv[0].reshape({x.size(0), heads_, d}).transpose(0, 1)*(1/std::sqrt(double(d)));
    auto k = at::cat({old.slots.at("key"), qkv[1].reshape({x.size(0), heads_, d})});
    auto v = at::cat({old.slots.at("value"), qkv[2].reshape({x.size(0), heads_, d})});
    auto bias = advance_bias(old.slots.at("log_bias"), w.extra.at("fiber_decay"), old.last_time, time);
    bias = at::cat({bias, at::zeros({x.size(0)}, x.options())});
    auto scores = at::matmul(k.transpose(0, 1), q.transpose(1, 2)).transpose(1, 2)+bias;
    auto output = at::matmul(at::softmax(scores, -1), v.transpose(0, 1));
    auto pooled = output.transpose(0, 1).contiguous().reshape({x.size(0), width}).sum(0);
    auto value = at::linear(pooled, w.extra.at("fiber_out").t(), w.extra.at("fiber_out_bias"));
    return {value, time, observations, {{"key", k}, {"value", v}, {"log_bias", bias}}};
  }
  bool exact_sequence() const override { return true; }
  bool joint_batch() const override { return true; }
  bool joint_sequence() const override { return true; }
  std::vector<State> batch(const NodeWeights& w, const std::vector<State>& old, const Tensor& h,
                          const std::vector<Index>& times, const ContentViews& views) const override {
    if (old.empty()) return {};
    PackedSequence p; p.contents = h; p.times = times; p.views = views;
    for (size_t i = 0; i < old.size(); ++i) { p.offsets.push_back(i+1); p.owners.emplace_back(i, 0); }
    return packed_sequence(w, old, p).states;
  }
  std::vector<State> sequence(const NodeWeights& w, const State& old, const Tensor& h,
                             const std::vector<Index>& times, const ContentViews& views) const override {
    if (times.empty()) return {};
    return packed_sequence(w, {old}, PackedSequence{h, {0, h.size(0)}, {{0, 0}}, times, views}).states;
  }
  PackedStates packed_sequence(const NodeWeights& w, const std::vector<State>& old,
                               const PackedSequence& batch) const override {
    return fiber_attention_packed(w, heads_, old, batch);
  }
  State reset(const State& state) const override {
    auto result = state; result.value = state.value*0;
    for (auto& [name, value] : result.slots) value = value.slice(0, 0, 0).clone();
    return result;
  }
  void validate_weights(const NodeWeights& w) const override {
    const auto d = w.bias.numel();
    if (d % heads_) throw std::invalid_argument("invalid fiber attention head/width policy");
    const std::map<std::string, std::vector<Index>> shapes{{"fiber_qkv", {d, 3*d}}, {"fiber_qkv_bias", {3*d}},
      {"fiber_out", {d, d}}, {"fiber_out_bias", {d}}, {"fiber_decay", {}}};
    for (const auto& [name, shape] : shapes) {
      auto it = w.extra.find(name);
      if (it == w.extra.end() || !it->second.defined() || it->second.sizes() != at::IntArrayRef(shape)
          || it->second.scalar_type() != w.bias.scalar_type() || it->second.device() != w.bias.device()
          || !at::isfinite(it->second).all().item<bool>())
        throw std::invalid_argument("invalid fiber attention parameter");
    }
  }
  void validate_state(const NodeWeights& w, const State& state) const override {
    if (state.slots.size() != 3 || !state.slots.count("key") || !state.slots.count("value") || !state.slots.count("log_bias"))
      throw std::invalid_argument("fiber attention requires key/value/log_bias slots");
    const auto& k = state.slots.at("key"); const auto& v = state.slots.at("value"); const auto& bias = state.slots.at("log_bias");
    if (k.dim() != 3 || k.size(1) != heads_ || k.size(2) != w.bias.numel()/heads_ || v.sizes() != k.sizes()
        || bias.dim() != 1 || bias.size(0) != k.size(0)) throw std::invalid_argument("invalid fiber attention cache shape");
  }
};
}  // namespace
std::shared_ptr<const StateKernel> make_fiber_attention_kernel(const Node& n) { return std::make_shared<FiberAttention>(n); }
Tensor decode_fiber_bias(const NodeWeights& w, const State& state, Index cut) {
  if (!state.slots.count("key") || state.slots.at("key").dim() != 3)
    throw std::invalid_argument("invalid fiber attention cache shape");
  Node node{0}; node.query_heads = node.kv_heads = state.slots.at("key").size(1);
  FiberAttention kernel(node); kernel.validate_weights(w); kernel.validate_state(w, state);
  if (cut < 0 || state.last_time < -1 || state.last_time >= cut || state.observations < 0)
    throw std::invalid_argument("invalid fiber attention cut/state clock");
  if (!state.value.defined() || state.value.sizes() != w.bias.sizes())
    throw std::invalid_argument("invalid fiber attention state value shape");
  auto check = [&](const Tensor& t) {
    if (!t.defined() || t.scalar_type() != w.bias.scalar_type() || t.device() != w.bias.device() || !at::isfinite(t).all().item<bool>())
      throw std::invalid_argument("invalid fiber attention state tensor");
  };
  check(state.value); for (const auto& [name, t] : state.slots) check(t);
  return advance_bias(state.slots.at("log_bias"), w.extra.at("fiber_decay"), state.last_time, cut-1);
}
}  // namespace tide
