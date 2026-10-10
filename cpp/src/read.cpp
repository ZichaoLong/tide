#include "tide/operator_work.h"
#include "tide/operator_profile.h"
#include "tide/read.h"
#include "tide/autograd.h"
#include "tide/isolated_read.h"
#include "placement_internal.h"
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
std::vector<Tensor> ReadKernel::batch_grad(const NodeWeights&, const std::vector<ReadInput>&) const {
  throw std::invalid_argument("Read program has no batched VJP");
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
  bool identity() const { return identity_; }
  bool batched_autograd() const override { return true; }
  std::vector<Tensor> batch_grad(const NodeWeights& w,const std::vector<ReadInput>& requests) const override {
    if (identity_) return batch(w,requests);
    std::vector<Tensor> rows;
    for (const auto& r:requests) rows.push_back(r.state?r.state->value:r.content.value);
    return isolated_read(rows,w.read,false,rows.front().scalar_type(),rows.front().device());
  }
 private:
  bool identity_;
};
class NormRead final : public ReadKernel {
 public:
  explicit NormRead(at::ScalarType dtype) : dtype_(dtype) {}
  Tensor step(const NodeWeights&, const ReadInput& r) const override {
    auto value=r.state ? r.state->value : r.content.value;
    if(dtype_==at::kFloat)value=value.to(at::kFloat);
    return at::norm(value, 2, {-1}, false, dtype_);
  }
  std::vector<Tensor> batch(const NodeWeights&, const std::vector<ReadInput>& requests) const override {
    std::vector<Tensor> values;
    for (const auto& r : requests) values.push_back(r.state ? r.state->value : r.content.value);
    auto batch=at::stack(values);
    if(dtype_==at::kFloat)batch=batch.to(at::kFloat);
    return at::norm(batch, 2, {-1}, false, dtype_).unbind();
  }
  at::ScalarType descriptor_dtype(at::ScalarType) const override { return dtype_; }
  bool batched_autograd() const override { return true; }
  std::vector<Tensor> batch_grad(const NodeWeights& w,const std::vector<ReadInput>& requests) const override {
    std::vector<Tensor> rows;
    for (const auto& r:requests) rows.push_back(r.state?r.state->value:r.content.value);
    return isolated_read(rows,w.read,true,dtype_,rows.front().device());
  }
  bool joint_batch() const override { return true; }
  void validate_weights(const NodeWeights&) const override {}
 private:
  at::ScalarType dtype_;
};
void validate(const Tensor& value, at::Device device, at::ScalarType dtype) {
  if (!value.defined() || value.dim() != 0 || value.device() != device
      || value.scalar_type() != dtype) throw std::invalid_argument("Read returned incompatible scalar metadata");
}
void finite(const std::vector<Tensor>& values) {
  std::map<std::tuple<int,int,int>,std::vector<Tensor>> groups;
  for (const auto& value:values)
    groups[{int(value.device().type()),value.device().index(),int(value.scalar_type())}].push_back(value);
  for (const auto& [key,rows]:groups)
    if (!at::isfinite(at::stack(rows)).all().item<bool>())
      throw std::invalid_argument("nonfinite selector score from Read");
}
}  // namespace
bool placement_detail::builtin_read(const ReadKernel& kernel, const Node& n) {
  if (auto k = dynamic_cast<const LinearRead*>(&kernel)) return n.readout == "linear-v1" && n.identity == k->identity();
  if (auto k = dynamic_cast<const NormRead*>(&kernel))
    return !n.identity && ((n.readout == "norm-fp32-v1" && k->descriptor_dtype(at::kFloat) == at::kFloat)
                       || (n.readout == "norm-fp64-v1" && k->descriptor_dtype(at::kFloat) == at::kDouble));
  return false;
}
std::shared_ptr<const ReadKernel> make_read_kernel(const Node& node) {
  if (node.readout == "norm-fp64-v1" && !node.identity) return std::make_shared<NormRead>(at::kDouble);
  if (node.readout == "norm-fp32-v1" && !node.identity) return std::make_shared<NormRead>(at::kFloat);
  if (node.readout != "linear-v1") throw std::invalid_argument("unknown Read profile: " + node.readout);
  return std::make_shared<LinearRead>(node.identity);
}
void evaluate_read(const Graph& g, const Model& m, std::vector<Event>& events, const std::vector<size_t>& ids, bool packed) {
  if (ids.empty()) return;
  op_profile::Scope profile(op_profile::Read);
  const auto node = events[ids.front()].node;
  const auto& w = m.nodes[node];
  const auto& mode = g.regions[g.nodes[node].region].read_mode;
  std::vector<ReadInput> requests;
  for (auto i : ids) {
    const auto& e = events[i];
    requests.push_back(read_input(mode, e.old, e.proposed_state, e.time, e.local_content()));
  }
  std::vector<Tensor> values;
  const bool batched=packed && at::GradMode::is_enabled() && w.read_kernel->batched_autograd();
  if (batched) values=w.read_kernel->batch_grad(w,requests);
  else if (packed) { at::NoGradGuard guard; values = w.read_kernel->batch(w, requests); }
  else for (const auto& r : requests) values.push_back(w.read_kernel->step(w, r));
  if (values.size() != ids.size()) throw std::invalid_argument("Read batch changed event count");
  for (size_t j = 0; j < ids.size(); ++j) {
    auto dtype = w.read_kernel->descriptor_dtype(requests[j].content.value.scalar_type());
    auto device = w.read_kernel->descriptor_device(requests[j].content.value.device());
    if (dtype != requests[j].content.value.scalar_type() && dtype != at::kDouble && dtype != at::kFloat)
      throw std::invalid_argument("invalid Read precision policy");
    validate(values[j], device, dtype);
  }
  finite(values);
  if (packed && at::GradMode::is_enabled() && !batched) {
    std::vector<Tensor> references;
    for (size_t j=0;j<ids.size();++j) {
        work::StateReplayTimer replay_timer(work::ReadReplayNs);
      auto semantic = w.read_kernel->step(w, requests[j]);
      validate(semantic, values[j].device(), values[j].scalar_type());
      references.push_back(semantic);
      values[j] = semantic_value(values[j], semantic);
    }
    finite(references);
  }
  for (size_t j=0;j<ids.size();++j) events[ids[j]].descriptor = values[j];
}
}  // namespace tide
