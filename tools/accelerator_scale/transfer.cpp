#include "placement.h"
#include <stdexcept>

namespace accelerator_scale {
Counters transfers;
void Counters::reset() {
  host_to_device = 0; device_to_host = 0; copies = 0;
  device_to_device = 0; local_messages = 0; remote_messages = 0;
  metadata_host_to_device=0;metadata_device_to_host=0;
}
std::map<std::string, double> Counters::metrics() const {
  return {{"transfer/host_to_device_bytes", double(host_to_device.load())},
          {"transfer/device_to_host_bytes", double(device_to_host.load())},
          {"transfer/device_to_device_bytes", double(device_to_device.load())},
          {"transfer/local_message_bytes", double(local_messages.load())},
          {"transfer/remote_message_bytes", double(remote_messages.load())},
          {"transfer/copy_calls", double(copies.load())},
          {"transfer/metadata_host_to_device_bytes",double(metadata_host_to_device.load())},
          {"transfer/metadata_device_to_host_bytes",double(metadata_device_to_host.load())}};
}
void Transfer::add(const Tensor& value) {
  if (!value.defined() || value.device() == destination_ || index_.count(value.unsafeGetTensorImpl())) return;
  index_[value.unsafeGetTensorImpl()] = source_.size(); source_.push_back(value);
}
void Transfer::add(const State& state) {
  add(state.value); for (const auto& [name, value] : state.slots) add(value);
}
void Transfer::add(const ContentView& view) {
  add(view.value);
  for (const auto& source : view.sources) { add(source.atom.value); add(source.scale); }
  for (const auto& value : view.contributions) add(value.value);
  if (view.source_batch) add(view.source_batch->values);
}
void Transfer::execute() {
  result_.resize(source_.size());
  // Joining otherwise independent differentiable fields would turn absent
  // gradients into connected zeros through cat's backward. Preserve each root.
  if (at::GradMode::is_enabled()) {
    for (size_t i = 0; i < source_.size(); ++i) {
      result_[i] = source_[i].to(destination_);
      const auto bytes = source_[i].numel()*source_[i].element_size();
      if (source_[i].device().is_cpu() && !destination_.is_cpu()) transfers.host_to_device += bytes;
      if (!source_[i].device().is_cpu() && destination_.is_cpu()) transfers.device_to_host += bytes;
      if (!source_[i].device().is_cpu() && !destination_.is_cpu()) transfers.device_to_device += bytes;
      ++transfers.copies;
    }
    return;
  }
  std::map<std::pair<std::string, at::ScalarType>, std::vector<size_t>> groups;
  for (size_t i = 0; i < source_.size(); ++i)
    groups[{source_[i].device().str(), source_[i].scalar_type()}].push_back(i);
  for (const auto& [key, indices] : groups) {
    std::vector<Tensor> flat;
    for (auto i : indices) flat.push_back(source_[i].reshape({-1}));
    auto joined = at::cat(flat), moved = joined.to(destination_);
    const auto bytes = joined.numel()*joined.element_size();
    if (joined.device().is_cpu() && !destination_.is_cpu()) transfers.host_to_device += bytes;
    if (!joined.device().is_cpu() && destination_.is_cpu()) transfers.device_to_host += bytes;
    if (!joined.device().is_cpu() && !destination_.is_cpu()) transfers.device_to_device += bytes;
    ++transfers.copies;
    int64_t offset = 0;
    for (auto i : indices) {
      result_[i] = moved.slice(0, offset, offset+source_[i].numel()).reshape(source_[i].sizes());
      offset += source_[i].numel();
    }
  }
}
Tensor Transfer::get(const Tensor& value) const {
  if (!value.defined() || value.device() == destination_) return value;
  return result_.at(index_.at(value.unsafeGetTensorImpl()));
}
State Transfer::get(const State& source) const {
  auto value = source; value.value = get(source.value);
  for (auto& [name, tensor] : value.slots) tensor = get(tensor);
  return value;
}
ContentStorage::ContentStorage(const ContentView& source, const Transfer& transfer)
    : value(transfer.get(source.value)), sources(source.sources.begin(), source.sources.end()),
      contributions(source.contributions.begin(), source.contributions.end()), row(source.source_row) {
  for (auto& s : sources) { s.atom.value = transfer.get(s.atom.value); s.scale = transfer.get(s.scale); }
  for (auto& v : contributions) v.value = transfer.get(v.value);
  if (source.source_batch) {
    batch = std::make_shared<SourceBatch>(*source.source_batch);
    batch->values = transfer.get(batch->values);
  }
}
ContentView ContentStorage::view() const { return {value, sources, contributions, batch, row}; }
Tensor host(const Tensor& value) {
  Transfer transfer(at::Device(at::kCPU)); transfer.add(value); transfer.execute(); return transfer.get(value);
}
Tensor embed(const Tensor& weight, const Tensor& ids, bool host_result) {
  Transfer transfer(weight.device()); transfer.add(ids); transfer.execute();
  auto value = weight.index_select(0, transfer.get(ids));
  return host_result ? host(value) : value;
}
Tensor project(const Tensor& hidden, const Tensor& weight, bool host_result, DenseLinear* head) {
  Transfer transfer(weight.device()); transfer.add(hidden); transfer.execute();
  auto value = head ? head->run(transfer.get(hidden), weight) : at::linear(transfer.get(hidden), weight);
  return host_result ? host(value) : value;
}
}  // namespace accelerator_scale
