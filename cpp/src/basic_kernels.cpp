#include "tide/counters.h"
#include "tide/kernel.h"
#include "tide/lazy_add.h"
#include "tide/operator_work.h"
#include <algorithm>
#include <stdexcept>

namespace tide {
namespace {
class BasicKernel final : public StateKernel {
 public:
  explicit BasicKernel(std::string kind) : kind_(std::move(kind)) {}
  State initial(const NodeWeights& w) const override {
    State s{at::zeros_like(w.bias)};
    if (kind_ == "ssm") s.slots["memory"] = at::zeros_like(w.bias);
    return s;
  }
  State step(const NodeWeights& w, const State& old, const ContentView& content, Index time) const override {
    const auto& h = content.value;
    if (kind_ == "identity") return old;
    if (kind_ == "ema") return {at::sigmoid(w.decay) * old.value + h, time, increment(old.observations)};
    const auto [a, b] = coefficients(w, h);
    auto memory = a * old.slots.at("memory") + b;
    return {summary(w, h, memory), time, increment(old.observations), {{"memory", memory}}};
  }
  std::vector<State> batch(const NodeWeights& w, const std::vector<State>& old, const Tensor& h,
                           const std::vector<Index>& times, const ContentViews&) const override {
    if (kind_ == "identity") return old;
    std::vector<Tensor> previous;
    for (const auto& s : old) previous.push_back(kind_ == "ema" ? s.value : s.slots.at("memory"));
    Tensor memory, values;
    if (kind_ == "ema") values = at::sigmoid(w.decay) * at::stack(previous) + h;
    else {
      const auto [a, b] = coefficients(w, h); memory = a * at::stack(previous) + b;
      values = summary(w, h, memory);
    }
    std::vector<State> states;
    for (size_t i = 0; i < old.size(); ++i) {
      State s{values[i], times[i], increment(old[i].observations)};
      if (kind_ == "ssm") s.slots["memory"] = memory[i];
      states.push_back(std::move(s));
    }
    return states;
  }
  bool exact_sequence() const override { return true; }
  bool joint_batch() const override { return true; }
  bool joint_sequence() const override { return true; }
  std::vector<State> sequence(const NodeWeights& w, const State& old, const Tensor& h,
                              const std::vector<Index>& times, const ContentViews&) const override {
    if (kind_ == "identity") return std::vector<State>(times.size(), old);
    Tensor memory, values;
    if (kind_ == "ema") values = affine_scan(at::sigmoid(w.decay).expand_as(h), h, old.value);
    else {
      const auto [a, b] = coefficients(w, h); memory = affine_scan(a, b, old.slots.at("memory"));
      values = summary(w, h, memory);
    }
    std::vector<State> states;
    for (size_t i = 0; i < times.size(); ++i) {
      State s{values[i], times[i], increment(old.observations, static_cast<Index>(i) + 1)};
      if (kind_ == "ssm") s.slots["memory"] = memory[i];
      if (i + 1 == times.size()) {
        s.value = s.value.clone();
        for (auto& [name, tensor] : s.slots) tensor = tensor.clone();
      }
      states.push_back(std::move(s));
    }
    return states;
  }
  PackedStates packed_sequence(const NodeWeights& w, const std::vector<State>& old,
                                const PackedSequence& p) const override {
    if (kind_ == "identity") return StateKernel::packed_sequence(w, old, p);
    p.validate();
    if (old.size() != p.owners.size()) throw std::invalid_argument("packed initial-state count mismatch");
    std::map<Index, std::vector<Index>> groups;
    for (size_t i = 0; i < old.size(); ++i) groups[p.offsets[i + 1] - p.offsets[i]].push_back(i);
    PackedStates result; result.states.resize(p.times.size());
    for (const auto& [length, ids] : groups) {
      std::vector<Tensor> contents, initial;
      for (auto i : ids) {
        contents.push_back(p.contents.slice(0, p.offsets[i], p.offsets[i + 1]));
        initial.push_back(kind_ == "ema" ? old[i].value : old[i].slots.at("memory"));
      }
      // One scan over [time,batch,width], with no padding or cross-sample terms.
      auto h = at::stack(contents, 1); auto previous = at::stack(initial);
      Tensor memory, values;
      if (kind_ == "ema") values = affine_scan(at::sigmoid(w.decay).expand_as(h), h, previous);
      else {
        const auto [a, b] = coefficients(w, h); memory = affine_scan(a, b, previous);
        values = summary(w, h, memory);
      }
      for (size_t row = 0; row < ids.size(); ++row) {
        const auto i = ids[row];
        for (Index t = 0; t < length; ++t) {
          const auto j = p.offsets[i] + t;
          State s{values[t][row], p.times[j], increment(old[i].observations, t + 1)};
          if (kind_ == "ssm") s.slots["memory"] = memory[t][row];
          if (t + 1 == length) {
            s.value = s.value.clone();
            for (auto& [name, tensor] : s.slots) tensor = tensor.clone();
          }
          result.states[j] = std::move(s);
        }
      }
      ++result.calls; result.max_batch = std::max<Index>(result.max_batch, ids.size());
      result.max_length = std::max(result.max_length, length);
    }
    return result;
  }
  void validate_weights(const NodeWeights& w) const override {
    if (kind_ != "ssm") return;
    const auto d = w.bias.numel();
    for (const auto& name : {"ssm_dt", "ssm_b", "ssm_c", "ssm_a", "ssm_skip"}) {
      const auto it = w.extra.find(name);
      const bool matrix = std::string(name) == "ssm_dt" || std::string(name) == "ssm_b" || std::string(name) == "ssm_c";
      const std::vector<Index> shape = matrix ? std::vector<Index>{d, d} : std::vector<Index>{d};
      if (it == w.extra.end() || it->second.sizes() != at::IntArrayRef(shape)) throw std::invalid_argument("invalid SSM weights");
    }
  }
  void validate_state(const NodeWeights& w, const State& s) const override {
    if (kind_ != "ssm") {
      if (!s.slots.empty()) throw std::invalid_argument("unexpected state slots");
    } else if (s.slots.size() != 1 || !s.slots.count("memory") || s.slots.at("memory").sizes() != w.bias.sizes())
      throw std::invalid_argument("SSM requires a width-sized memory slot");
  }
 private:
  std::string kind_;
  static std::pair<Tensor, Tensor> coefficients(const NodeWeights& w, const Tensor& h) {
    const auto d = w.bias.numel();
    work::linear(work::StateCalls, h.numel()/d, d, d);
    work::linear(work::StateCalls, h.numel()/d, d, d);
    auto dt = at::softplus(at::matmul(h, w.extra.at("ssm_dt")));
    return {at::exp(-at::softplus(w.extra.at("ssm_a")) * dt), dt * at::matmul(h, w.extra.at("ssm_b"))};
  }
  static Tensor summary(const NodeWeights& w, const Tensor& h, const Tensor& memory) {
    const auto d = w.bias.numel();
    work::linear(work::StateCalls, h.numel()/d, d, d);
    return at::matmul(h, w.extra.at("ssm_c")) * memory + w.extra.at("ssm_skip") * h;
  }
};
}  // namespace
std::shared_ptr<const StateKernel> make_matrix_kernel(const std::string&);
std::shared_ptr<const StateKernel> make_state_kernel(const std::string& name) {
  if (name == "lh-add-repeat-v1") return make_add_repeat_kernel();
  if (name == "linear" || name == "delta" || name == "delta-rule-v1") return make_matrix_kernel(name);
  if (name != "ema" && name != "identity" && name != "ssm") throw std::invalid_argument("unknown state kernel: " + name);
  return std::make_shared<BasicKernel>(name);
}
}  // namespace tide
