#include "placement_internal.h"
#include "tide/device.h"
#include "tide/kernel.h"
#include "tide/ops.h"
#include "tide/ownership.h"
#include <stdexcept>

namespace tide {
namespace {
at::Device resolve(const std::string& request, const std::string& fallback, at::Device payload) {
  const auto value = request == "auto" ? fallback : request;
  if (value == "payload") return payload;
  auto device = at::Device(value);
  if (device.is_cpu()) return at::Device(at::kCPU);
  if (device.type() == payload.type() && (!device.has_index() || device.index() == payload.index())) return payload;
  throw std::invalid_argument("placement must use CPU or the model payload device");
}
}
ResolvedPlacement resolve_placement(const ExecutionPlacement& p, at::Device payload) {
  std::string read = "payload", control = "payload", selection = "cpu", events = "cpu";
  if (p.preset == "cpu") {
    if (!payload.is_cpu()) throw std::invalid_argument("CPU preset requires CPU payload tensors");
    read = control = "cpu";
  } else if (p.preset == "mixed-a" || p.preset == "mixed-b" || p.preset == "mixed-c" || p.preset == "resident") {
    if (payload.is_cpu()) throw std::invalid_argument("accelerator preset requires accelerator payload tensors");
    if (p.preset == "mixed-a") read = control = "cpu";
    if (p.preset == "mixed-c" || p.preset == "resident") selection = "payload";
    if (p.preset == "resident") events = "payload";
  } else if (p.preset != "native") throw std::invalid_argument("unknown execution placement preset");
  if (p.scoring_dtype != "profile" && p.scoring_dtype != "payload" &&
      p.scoring_dtype != "float32" && p.scoring_dtype != "float64")
    throw std::invalid_argument("unknown scoring precision");
  const auto backend = std::string(execution_backend());
  if (!payload.is_cpu() && !((payload.is_cuda() && backend == "cuda") ||
      (payload.type() == c10::DeviceType::PrivateUse1 && backend == "npu")))
    throw std::invalid_argument("payload backend is unavailable in this build");
  return {payload, resolve(p.read, read, payload), resolve(p.control, control, payload),
          resolve(p.selection, selection, payload), resolve(p.events, events, payload), p.scoring_dtype};
}
std::map<std::string, std::string> ResolvedPlacement::record() const {
  return {{"payload",payload.str()},{"read",read.str()},{"control",control.str()},
          {"selection",selection.str()},{"events",events.str()},{"scoring_dtype",scoring_dtype}};
}
Model place_model(const Graph& graph, const Model& model, const ExecutionPlacement& request) {
  if (model.nodes.empty()) throw std::invalid_argument("placement requires a model");
  const auto placement = resolve_placement(request, model.nodes[0].bias.device());
  if (!placement.events.is_cpu())
    throw std::invalid_argument("host model placement cannot provide device-resident event progression");
  if (model.nodes.size() != graph.nodes.size() || (!model.regions.empty() && model.regions.size() != graph.regions.size()))
    throw std::invalid_argument("placement model/graph size mismatch");
  Model result = model;
  if (result.regions.empty()) result.regions.resize(graph.regions.size());
  for (size_t i = 0; i < result.nodes.size(); ++i)
    result.nodes[i].read_kernel = placement_detail::read_kernel(graph.nodes[i], model.nodes[i],
      resolve_placement(request, model.nodes[i].bias.device()));
  for (size_t i = 0; i < result.regions.size(); ++i)
    result.regions[i].kernel = placement_detail::region_kernel(graph.regions[i], result.regions[i],
      resolve_placement(request, region_reference(graph, result, i).device()));
  configure_model(graph, result);
  validate_model(graph, result);
  return result;
}
}  // namespace tide
