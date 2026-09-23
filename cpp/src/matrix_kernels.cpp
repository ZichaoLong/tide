#include "tide/counters.h"
#include "tide/kernel.h"
#include <array>
#include <stdexcept>

namespace tide {
namespace {
Tensor matrix_scan(Tensor a, Tensor b, const Tensor& initial) {
  const auto n = b.size(0);
  for (Index stride = 1; stride < n; stride *= 2) {
    b = at::cat({b.slice(0, 0, stride), b.slice(0, stride) + at::matmul(a.slice(0, stride), b.slice(0, 0, n - stride))});
    a = at::cat({a.slice(0, 0, stride), at::matmul(a.slice(0, stride), a.slice(0, 0, n - stride))});
  }
  return b + at::matmul(a, initial);
}
class MatrixKernel final : public StateKernel {
 public:
  explicit MatrixKernel(const std::string& name) : linear_(name == "linear"), gated_(name == "delta") {}
  State initial(const NodeWeights& w) const override {
    const auto d = w.bias.numel();
    State s{at::zeros_like(w.bias), -1, 0, {{"matrix", at::zeros({d, d}, w.bias.options())}}};
    if (linear_) s.slots["normalizer"] = at::zeros_like(w.bias);
    return s;
  }
  State step(const NodeWeights& w, const State& old, const ContentView& content, Index time) const override {
    const auto& h = content.value;
    auto p = project(w, h); const auto& q = p[0]; const auto& k = p[1]; const auto& v = p[2];
    State s; s.last_time = time; s.observations = increment(old.observations);
    if (linear_) {
      s.slots["matrix"] = old.slots.at("matrix") + k.unsqueeze(-1) * v.unsqueeze(-2);
      s.slots["normalizer"] = old.slots.at("normalizer") + k;
      s.value = output(w, q, s.slots.at("matrix"), s.slots.at("normalizer"));
    } else {
      auto beta = at::sigmoid(at::matmul(h, w.extra.at("mem_beta")));
      auto decay = gated_ ? at::sigmoid(at::matmul(h, w.extra.at("mem_decay"))) : at::ones_like(beta);
      auto decayed = decay * old.slots.at("matrix");
      auto error = v - at::matmul(k, decayed);
      s.slots["matrix"] = decayed + beta * k.unsqueeze(-1) * error.unsqueeze(-2);
      s.value = output(w, q, s.slots.at("matrix"));
    }
    return s;
  }
  std::vector<State> batch(const NodeWeights& w, const std::vector<State>& old, const Tensor& h,
                           const std::vector<Index>& times, const ContentViews&) const override {
    auto p = project(w, h); const auto& q = p[0]; const auto& k = p[1]; const auto& v = p[2];
    std::vector<Tensor> ms, zs;
    for (const auto& s : old) { ms.push_back(s.slots.at("matrix")); if (linear_) zs.push_back(s.slots.at("normalizer")); }
    Tensor matrix, z, values;
    if (linear_) {
      matrix = at::stack(ms) + k.unsqueeze(-1) * v.unsqueeze(-2);
      z = at::stack(zs) + k; values = output(w, q, matrix, z);
    } else {
      auto beta = at::sigmoid(at::matmul(h, w.extra.at("mem_beta"))).unsqueeze(-1).unsqueeze(-1);
      auto decay = gated_ ? at::sigmoid(at::matmul(h, w.extra.at("mem_decay"))).unsqueeze(-1).unsqueeze(-1) : at::ones_like(beta);
      auto decayed = decay * at::stack(ms);
      auto error = v - at::matmul(k.unsqueeze(-2), decayed).squeeze(-2);
      matrix = decayed + beta * k.unsqueeze(-1) * error.unsqueeze(-2); values = output(w, q, matrix);
    }
    std::vector<State> states;
    for (size_t i = 0; i < old.size(); ++i) {
      State s{values[i], times[i], increment(old[i].observations), {{"matrix", matrix[i]}}};
      if (linear_) s.slots["normalizer"] = z[i]; states.push_back(std::move(s));
    }
    return states;
  }
  bool exact_sequence() const override { return true; }
  bool joint_batch() const override { return true; }
  bool joint_sequence() const override { return true; }
  std::vector<State> sequence(const NodeWeights& w, const State& old, const Tensor& h,
                              const std::vector<Index>& times, const ContentViews&) const override {
    auto p = project(w, h); const auto& q = p[0]; const auto& k = p[1]; const auto& v = p[2];
    Tensor matrix, z, values;
    if (linear_) {
      matrix = at::cumsum(k.unsqueeze(-1) * v.unsqueeze(-2), 0) + old.slots.at("matrix");
      z = at::cumsum(k, 0) + old.slots.at("normalizer"); values = output(w, q, matrix, z);
    } else {
      auto beta = at::sigmoid(at::matmul(h, w.extra.at("mem_beta"))).unsqueeze(-1).unsqueeze(-1);
      auto decay = gated_ ? at::sigmoid(at::matmul(h, w.extra.at("mem_decay"))).unsqueeze(-1).unsqueeze(-1) : at::ones_like(beta);
      auto a = decay * (at::eye(h.size(-1), h.options()) - beta * k.unsqueeze(-1) * k.unsqueeze(-2));
      auto b = beta * k.unsqueeze(-1) * v.unsqueeze(-2);
      matrix = matrix_scan(a, b, old.slots.at("matrix")); values = output(w, q, matrix);
    }
    std::vector<State> states;
    for (size_t i = 0; i < times.size(); ++i) {
      State s{values[i], times[i], increment(old.observations, static_cast<Index>(i) + 1), {{"matrix", matrix[i]}}};
      if (linear_) s.slots["normalizer"] = z[i]; states.push_back(std::move(s));
    }
    return states;
  }
  void validate_weights(const NodeWeights& w) const override {
    const auto d = w.bias.numel();
    for (const auto& name : {"mem_q", "mem_k", "mem_v", "mem_out"}) {
      auto it = w.extra.find(name);
      if (it == w.extra.end() || it->second.sizes() != at::IntArrayRef({d, d})) throw std::invalid_argument("invalid matrix-memory weights");
    }
    if (!linear_) for (const auto& name : {"mem_beta", "mem_decay"}) {
      if (std::string(name) == "mem_decay" && !gated_) continue;
      auto it = w.extra.find(name);
      if (it == w.extra.end() || it->second.sizes() != w.bias.sizes()) throw std::invalid_argument("invalid delta gate weights");
    }
  }
  void validate_state(const NodeWeights& w, const State& s) const override {
    const auto d = w.bias.numel();
    if (s.slots.size() != (linear_ ? 2 : 1) || !s.slots.count("matrix")
        || s.slots.at("matrix").sizes() != at::IntArrayRef({d, d})) throw std::invalid_argument("invalid matrix-memory slots");
    if (linear_ && (!s.slots.count("normalizer") || s.slots.at("normalizer").sizes() != w.bias.sizes()
                    || (s.slots.at("normalizer") < 0).any().item<bool>()))
      throw std::invalid_argument("invalid linear attention normalizer");
  }
 private:
  bool linear_, gated_;
  std::array<Tensor, 3> project(const NodeWeights& w, const Tensor& h) const {
    auto q = at::matmul(h, w.extra.at("mem_q")); auto k = at::matmul(h, w.extra.at("mem_k"));
    if (linear_) { q = at::elu(q) + 1; k = at::elu(k) + 1; }
    else {
      q = q / at::linalg_vector_norm(q, 2, {-1}, true).clamp_min(1e-6);
      k = k / at::linalg_vector_norm(k, 2, {-1}, true).clamp_min(1e-6);
    }
    return {q, k, at::matmul(h, w.extra.at("mem_v"))};
  }
  static Tensor output(const NodeWeights& w, const Tensor& q, const Tensor& matrix, const Tensor& z = {}) {
    auto value = at::matmul(q.unsqueeze(-2), matrix).squeeze(-2);
    if (z.defined()) value = value / ((q * z).sum(-1, true) + 1e-6);
    return at::matmul(value, w.extra.at("mem_out"));
  }
};
}  // namespace
std::shared_ptr<const StateKernel> make_matrix_kernel(const std::string& name) {
  return std::make_shared<MatrixKernel>(name);
}
}  // namespace tide
