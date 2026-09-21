#include "tide/read.h"
#include "tide/autograd.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace tide {
ReadInput read_input(const std::string& mode, const State& old, const State& proposal,
                     Index time, const ContentView& content) {
  if (mode != "content" && mode != "old" && mode != "proposal") throw std::invalid_argument("invalid region Read mode");
  return {mode == "content" ? nullptr : mode == "old" ? &old : &proposal, time, content};
}
std::vector<Tensor> ReadKernel::batch(const NodeWeights& w, const std::vector<ReadInput>& requests) const {
  std::vector<Tensor> result;
  for (const auto& r : requests) result.push_back(step(w, r));
  return result;
}
namespace {
class LinearRead final : public ReadKernel {
 public:
  explicit LinearRead(bool identity) : identity_(identity) {}
  Tensor step(const NodeWeights& w, const ReadInput& r) const override {
    if (identity_) return at::zeros({}, r.content.value.options());
    return ((r.state ? r.state->value : r.content.value) * w.read).sum(-1);
  }
  std::vector<Tensor> batch(const NodeWeights& w, const std::vector<ReadInput>& requests) const override {
    if (identity_) return ReadKernel::batch(w, requests);
    std::vector<Tensor> values;
    for (const auto& r : requests) values.push_back(r.state ? r.state->value : r.content.value);
    return (at::stack(values) * w.read).sum(-1).unbind();
  }
  bool joint_batch() const override { return true; }
  void validate_weights(const NodeWeights&) const override {}
 private:
  bool identity_;
};
void validate(const Tensor& value, const ReadInput& r) {
  if (!value.defined() || value.dim() != 0 || value.device() != r.content.value.device()
      || value.scalar_type() != r.content.value.scalar_type()) throw std::invalid_argument("Read returned incompatible scalar metadata");
  if (!at::isfinite(value).item<bool>()) throw std::invalid_argument("nonfinite selector score from Read");
}
}  // namespace
std::shared_ptr<const ReadKernel> make_read_kernel(const Node& node) {
  if (node.readout != "linear-v1") throw std::invalid_argument("unknown Read profile: " + node.readout);
  return std::make_shared<LinearRead>(node.identity);
}
void evaluate_read(const Graph& g, const Model& m, std::vector<Event>& events, const std::vector<size_t>& ids, bool packed) {
  if (ids.empty()) return;
  const auto node = events[ids.front()].node;
  const auto& w = m.nodes[node];
  const auto& mode = g.regions[g.nodes[node].region].read_mode;
  std::vector<ReadInput> requests;
  for (auto i : ids) {
    const auto& e = events[i];
    requests.push_back(read_input(mode, e.old, e.proposed_state, e.time, e.local_content()));
  }
  std::vector<Tensor> values;
  if (packed) { at::NoGradGuard guard; values = w.read_kernel->batch(w, requests); }
  else for (const auto& r : requests) values.push_back(w.read_kernel->step(w, r));
  if (values.size() != ids.size()) throw std::invalid_argument("Read batch changed event count");
  for (size_t j = 0; j < ids.size(); ++j) {
    validate(values[j], requests[j]);
    if (packed && at::GradMode::is_enabled()) {
      auto semantic = w.read_kernel->step(w, requests[j]);
      validate(semantic, requests[j]);
      values[j] = semantic_value(values[j], semantic);
    }
    events[ids[j]].descriptor = values[j];
  }
}
}  // namespace tide
